"""
Mix multiple OpenPI JAX checkpoints (Orbax/OCDBT) with weighted averaging.

For PyTorch checkpoints (model.safetensors), use arithmetic_torch.py instead.

Example usage:
python model_arithmetic/arithmetic.py \
    --config pi05_hang_cloth \
    --data-path hang_cloth_1125_v6-5_data.pkl \
    --checkpoints /path/to/ckpt1/90000 /path/to/ckpt2/90000 \
    --output /path/to/mixed \
    --optimize_method inverse_loss \
    --use_gpu
"""
import argparse
import gc
import logging
import os
import time
from functools import partial
from pathlib import Path
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flax import nnx
import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
import openpi.shared.normalize as _normalize
from openpi.training import config as _config

from common import (
    compute_optimal_weights,
    load_norm_stats,
    mix_norm_stats,
    mix_params,
    save_norm_stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("xla").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def resolve_ckpt_path(path: str) -> str:
    """Resolve checkpoint path to params directory (JAX Orbax)."""
    p = Path(path).resolve()
    # Support both step dir (e.g. .../90000) and params subdir
    if (p / "_METADATA").exists():
        return str(p)
    elif (p / "_CHECKPOINT_METADATA").exists() and (p / "params" / "_METADATA").exists():
        return str(p / "params")
    elif (p.name == "params") and (p.parent / "_CHECKPOINT_METADATA").exists():
        return str(p)
    else:
        raise FileNotFoundError(f"Invalid JAX checkpoint path: {p}")


def load_jax_params(checkpoint_path: str):
    """Load parameters from a JAX checkpoint. Returns flat dict for mixing."""
    resolved = resolve_ckpt_path(checkpoint_path)
    params = _model.restore_params(resolved, restore_type=np.ndarray)
    return flax.traverse_util.flatten_dict(params, sep="/")



def save_jax_params(flat_params, output_dir):
    """Save mixed parameters to OCDBT checkpoint format (step 0)."""
    nested = flax.traverse_util.unflatten_dict(flat_params, sep="/")
    os.makedirs(output_dir, exist_ok=True)
    # Write as Orbax checkpoint so JAX can load it
    mngr = ocp.CheckpointManager(
        output_dir,
        item_handlers={"params": ocp.PyTreeCheckpointHandler(use_ocdbt=True)},
        options=ocp.CheckpointManagerOptions(max_to_keep=None, create=True),
    )
    mngr.save(0, {"params": {"params": nested}})
    mngr.wait_until_finished()
    print(f"✓ Saved JAX checkpoint to {output_dir}/0/")


def compute_checkpoint_losses(checkpoints, config, data_samples_list):
    """Compute mean loss per checkpoint on validation batches (for inverse_loss weights)."""
    losses = []

    for ckpt_path in checkpoints:
        ckpt_root = os.path.dirname(ckpt_path)
        # Prefer norm_stats next to checkpoint
        if os.path.exists(Path(ckpt_root) / "norm_stats.json"):
            norm_stats = _normalize.load(ckpt_root)
        else:
            norm_stats = _normalize.load(ckpt_path)
        policy = _policy_config.create_trained_policy(config, ckpt_path, norm_stats=norm_stats)
        ckpt_losses = []
        for data_samples in tqdm(
            data_samples_list, desc="Computing checkpoint losses"
        ):
            loss = policy._model.compute_loss(jax.random.key(0), data_samples[0], data_samples[1])
            ckpt_losses.append(float(jnp.mean(loss)))
        print(f"Checkpoint losses for {ckpt_path}: {ckpt_losses}")
        avg_loss = float(np.mean(ckpt_losses))
        losses.append(avg_loss)
        del policy, norm_stats
    print(f"Computed losses: {losses}")
    return losses


def optimize_weights_with_gradient_descent(
    checkpoints, config, data_samples_list,
    num_iterations=50, learning_rate=0.1, print_every=1
):
    """Optimize mixing weights using gradient descent."""
    print("\n" + "=" * 60)
    print("Optimizing weights with gradient descent...")
    print("=" * 60)

    ckpt_root = os.path.dirname(checkpoints[0])
    norm_stats = _normalize.load(ckpt_root)
    policy = _policy_config.create_trained_policy(
        config, checkpoints[0], norm_stats=norm_stats
    )
    # Load all checkpoints as flat params on CPU (mixing on GPU would OOM)
    params_list_cpu = []
    for ckpt_path in tqdm(checkpoints, desc="Loading checkpoints"):
        resolved = resolve_ckpt_path(ckpt_path)
        params = _model.restore_params(resolved, restore_type=np.ndarray)
        params_list_cpu.append(flax.traverse_util.flatten_dict(params, sep="/"))

    cpu_device = jax.devices("cpu")[0]
    params_list_jax_cpu = jax.device_put(params_list_cpu, cpu_device)

    n_checkpoints = len(checkpoints)
    # Optimize in log-space so weights stay on simplex after softmax
    log_weights = jnp.zeros(n_checkpoints)
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=num_iterations, alpha=0.01
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(log_weights)

    @partial(jax.jit, static_argnames=["policy"])
    def compute_loss_wrt_params(flat_params, policy, data_samples):
        model = policy._model
        nested_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")
        nnx.update(model, nnx.State(nested_params))
        loss = model.compute_loss(jax.random.key(0), data_samples[0], data_samples[1])
        return jnp.mean(loss)

    @partial(jax.jit, backend="cpu")
    def mix_params_cpu(params_list, weights):
        def weighted_sum(*args):
            res = jnp.zeros_like(args[0])
            for p, w in zip(args, weights):
                res += p * w
            return res
        return jax.tree.map(weighted_sum, *params_list)

    @partial(jax.jit, backend="cpu")
    def project_grads_cpu(grads, params_list):
        # Project param gradient onto each checkpoint's params (for weight gradient)
        dots = []
        for p_k in params_list:
            term_dots = jax.tree.map(lambda g, p: jnp.sum(g * p), grads, p_k)
            dots.append(jax.tree_util.tree_reduce(jnp.add, term_dots))
        return jnp.array(dots)

    best_loss = float("inf")
    best_weights = None
    gpu_device = jax.devices("gpu")[0]

    for iteration in range(num_iterations):
        current_weights = jax.nn.softmax(log_weights)
        # Mix params with current weights, run forward on GPU
        mixed_params_cpu = mix_params_cpu(params_list_jax_cpu, current_weights)
        mixed_params_gpu = jax.device_put(mixed_params_cpu, gpu_device)

        loss_value, param_grads_gpu = jax.value_and_grad(compute_loss_wrt_params)(
            mixed_params_gpu,
            policy,
            data_samples_list[iteration % len(data_samples_list)],
        )
        param_grads_cpu = jax.device_put(param_grads_gpu, cpu_device)
        # d(loss)/d(weight_k) = sum over params of (grad * theta_k); then convert to d/d(log_weights)
        g_k = project_grads_cpu(param_grads_cpu, params_list_jax_cpu)
        # Gradient of loss w.r.t. log_weights (on simplex)
        g_k_np = np.array(g_k)
        weights_np = np.array(current_weights)
        g_bar = np.sum(g_k_np * weights_np)
        grad_log_weights = weights_np * (g_k_np - g_bar)

        updates, opt_state = optimizer.update(
            jnp.array(grad_log_weights), opt_state
        )
        log_weights = optax.apply_updates(log_weights, updates)
        loss_val_float = float(loss_value)

        if loss_val_float < best_loss:
            best_loss = loss_val_float
            best_weights = weights_np.copy()
        if (iteration + 1) % print_every == 0 or iteration == 0:
            print(f"Iter {iteration + 1}/{num_iterations}: loss={loss_val_float:.6f}, weights={weights_np}")
        del mixed_params_cpu, mixed_params_gpu, param_grads_gpu, param_grads_cpu, g_k, current_weights, updates

    print(f"\nBest loss: {best_loss:.6f}, Best weights: {best_weights}")
    result = [float(w) for w in (best_weights if best_weights is not None else jax.nn.softmax(log_weights))]
    del params_list_cpu, params_list_jax_cpu, policy, norm_stats, optimizer, opt_state, log_weights
    jax.clear_caches()
    gc.collect()
    return result


def optimize_weights_with_adaptive_gradient_descent(
    checkpoints, config, data_samples_list,
    num_iterations=50, learning_rate=0.1, print_every=1
):
    """Optimize mixing weights with adaptive gradient descent."""
    print("\n" + "=" * 60)
    print("Optimizing weights with adaptive gradient descent...")
    print("=" * 60)

    ckpt_root = os.path.dirname(checkpoints[0])
    norm_stats = _normalize.load(ckpt_root)
    policy = _policy_config.create_trained_policy(
        config, checkpoints[0], norm_stats=norm_stats
    )
    params_list_cpu = []
    for ckpt_path in tqdm(checkpoints, desc="Loading checkpoints"):
        resolved = resolve_ckpt_path(ckpt_path)
        params = _model.restore_params(resolved, restore_type=np.ndarray)
        params_list_cpu.append(flax.traverse_util.flatten_dict(params, sep="/"))

    cpu_device = jax.devices("cpu")[0]
    params_list_jax_cpu = jax.device_put(params_list_cpu, cpu_device)
    n_checkpoints = len(checkpoints)
    log_weights = jnp.zeros(n_checkpoints)
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=num_iterations, alpha=0.01
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(log_weights)

    @partial(jax.jit, static_argnames=["policy"])
    def compute_loss_wrt_params(flat_params, policy, data_samples):
        model = policy._model
        nested_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")
        nnx.update(model, nnx.State(nested_params))
        loss = model.compute_loss(
            jax.random.key(0), data_samples[0], data_samples[1]
        )
        return jnp.mean(loss)

    @partial(jax.jit, backend="cpu")
    def mix_params_cpu(params_list, weights):
        def weighted_sum(*args):
            res = jnp.zeros_like(args[0])
            for p, w in zip(args, weights):
                res += p * w
            return res
        return jax.tree.map(weighted_sum, *params_list)

    @partial(jax.jit, backend="cpu")
    def project_grads_cpu(grads, params_list):
        # Project param gradient onto each checkpoint's params (for weight gradient)
        dots = []
        for p_k in params_list:
            term_dots = jax.tree.map(
                lambda g, p: jnp.sum(g * p), grads, p_k
            )
            dots.append(jax.tree_util.tree_reduce(jnp.add, term_dots))
        return jnp.array(dots)

    @partial(jax.jit, backend="cpu")
    def compute_weight_gradient(g_k, weights):
        g_bar = jnp.sum(g_k * weights)
        return weights * (g_k - g_bar)

    @partial(jax.jit, backend="cpu")
    def optimizer_step(log_weights, opt_state, grad_log_weights, loss_val):
        # Scale gradient by loss so steps are adaptive
        scale = (loss_val / 0.05) ** 2
        scaled_grads = grad_log_weights * scale
        updates, new_opt_state = optimizer.update(scaled_grads, opt_state)
        new_log_weights = optax.apply_updates(log_weights, updates)
        return new_log_weights, new_opt_state

    best_loss = float("inf")
    best_weights = None
    gpu_device = jax.devices("gpu")[0]

    for iteration in range(num_iterations):
        current_weights = jax.nn.softmax(log_weights)
        mixed_params_cpu = mix_params_cpu(params_list_jax_cpu, current_weights)
        mixed_params_gpu = jax.device_put(mixed_params_cpu, gpu_device)
        loss_value, param_grads_gpu = jax.value_and_grad(compute_loss_wrt_params)(
            mixed_params_gpu, policy, data_samples_list[iteration % len(data_samples_list)]
        )

        param_grads_cpu = jax.device_put(param_grads_gpu, cpu_device)
        g_k = project_grads_cpu(param_grads_cpu, params_list_jax_cpu)
        grad_log_weights = compute_weight_gradient(g_k, current_weights)
        loss_val_float = float(loss_value)
        log_weights, opt_state = optimizer_step(log_weights, opt_state, grad_log_weights, loss_val_float)
        weights_np = np.array(current_weights)

        if loss_val_float < best_loss:
            best_loss = loss_val_float
            best_weights = weights_np.copy()
        if (iteration + 1) % print_every == 0 or iteration == 0:
            print(f"Iter {iteration + 1}/{num_iterations}: loss={loss_val_float:.6f}, weights={weights_np}")
        del mixed_params_cpu, mixed_params_gpu, param_grads_gpu, param_grads_cpu, g_k, current_weights

    print(f"\nBest loss: {best_loss:.6f}, Best weights: {best_weights}")
    result = [float(w) for w in (best_weights if best_weights is not None else jax.nn.softmax(log_weights))]
    del params_list_cpu, params_list_jax_cpu, policy, norm_stats, optimizer, opt_state, log_weights
    jax.clear_caches()
    gc.collect()
    return result


def optimize_weights_greedy(checkpoints, config, data_samples_list):
    """Greedy optimization: best single checkpoint, then iteratively add best next."""
    print("\n" + "=" * 60)
    print("Optimizing weights with greedy strategy...")
    print("=" * 60)

    ckpt_root = os.path.dirname(checkpoints[0])
    norm_stats = _normalize.load(ckpt_root)
    policy = _policy_config.create_trained_policy(
        config, checkpoints[0], norm_stats=norm_stats
    )
    params_list_cpu = []
    for ckpt_path in tqdm(checkpoints, desc="Loading checkpoints"):
        resolved = resolve_ckpt_path(ckpt_path)
        params = _model.restore_params(resolved, restore_type=np.ndarray)
        params_list_cpu.append(flax.traverse_util.flatten_dict(params, sep="/"))

    cpu_device = jax.devices("cpu")[0]
    gpu_device = jax.devices("gpu")[0]
    params_list_jax_cpu = jax.device_put(params_list_cpu, cpu_device)

    @partial(jax.jit, static_argnames=["policy"])
    def compute_loss_wrt_params(flat_params, policy, data_samples):
        model = policy._model
        nested_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")
        nnx.update(model, nnx.State(nested_params))
        loss = model.compute_loss(jax.random.key(0), data_samples[0], data_samples[1])
        return jnp.mean(loss)

    @partial(jax.jit, backend="cpu")
    def mix_params_cpu(params_list, weights):
        def weighted_sum(*args):
            res = jnp.zeros_like(args[0])
            for p, w in zip(args, weights):
                res += p * w
            return res
        return jax.tree.map(weighted_sum, *params_list)

    def evaluate_combination(indices):
        """Average loss when using only checkpoints at indices (equal weights)."""
        n_selected = len(indices)
        weights = np.zeros(len(checkpoints))
        weights[indices] = 1.0 / n_selected
        weights_jax = jnp.array(weights)
        mixed_params_cpu = mix_params_cpu(params_list_jax_cpu, weights_jax)
        mixed_params_gpu = jax.device_put(mixed_params_cpu, gpu_device)
        total_loss = 0.0
        for batch_data in data_samples_list:
            loss = compute_loss_wrt_params(mixed_params_gpu, policy, batch_data)
            total_loss += float(loss)

        del mixed_params_gpu
        return total_loss / len(data_samples_list)

    n_checkpoints = len(checkpoints)
    remaining_indices = list(range(n_checkpoints))
    selected_indices = []
    best_loss = float("inf")
    # Phase 1: pick best single checkpoint
    print("\nEvaluating individual checkpoints...")
    for i in remaining_indices:
        loss = evaluate_combination([i])
        print(f"  Checkpoint {i+1}: loss={loss:.6f}")
        if loss < best_loss:
            best_loss = loss
            selected_indices = [i]
    remaining_indices.remove(selected_indices[0])
    print(f"-> Selected best start: Checkpoint {selected_indices[0]+1} (loss={best_loss:.6f})")
    # Phase 2: greedily add checkpoints that improve loss
    while remaining_indices:
        print(f"\nSearching for best addition to {[i+1 for i in selected_indices]}...")
        iteration_best_loss = best_loss
        best_candidate = -1
        for i in remaining_indices:
            loss = evaluate_combination(selected_indices + [i])
            print(f"  + Checkpoint {i+1}: loss={loss:.6f}")
            if loss < iteration_best_loss:
                iteration_best_loss = loss
                best_candidate = i
        if best_candidate != -1:
            best_loss = iteration_best_loss
            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)
            print(f"-> Improvement found! Added Checkpoint {best_candidate+1}. New loss: {best_loss:.6f}")
            jax.clear_caches()
            gc.collect()
        else:
            print("-> No improvement found. Stopping.")
            break

    final_weights = np.zeros(n_checkpoints)
    final_weights[selected_indices] = 1.0 / len(selected_indices)
    print(f"\nFinal greedy weights: {final_weights}")
    del params_list_cpu, params_list_jax_cpu, policy, norm_stats
    gc.collect()
    return final_weights.tolist()


def test_mixed_checkpoint_jax(config, checkpoint_path, data_samples_list):
    """Test mixed JAX checkpoint and compute average loss."""
    norm_stats = _normalize.load(checkpoint_path)
    ckpt_dir = os.path.join(checkpoint_path, "0")
    policy = _policy_config.create_trained_policy(config, ckpt_dir, norm_stats=norm_stats)
    avg_loss = 0.0
    for data_samples in data_samples_list:
        loss = policy._model.compute_loss(jax.random.key(0), data_samples[0], data_samples[1])
        avg_loss += float(jnp.mean(loss))
    avg_loss /= len(data_samples_list)
    del policy, norm_stats
    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Mix OpenPI JAX checkpoints (Orbax) with weighted averaging. Use arithmetic_torch.py for PyTorch."
    )
    parser.add_argument("--config", required=True, help="Config name")
    parser.add_argument("--data-path", required=True, help="Test data pickle file")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint directories")
    parser.add_argument("--weights", nargs="+", type=float, help="Manual weights")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--optimize_method",
        type=str,
        default="gradient_descent",
        choices=["inverse_loss", "gradient_descent", "adaptive_gradient_descent", "greedy"],
    )
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--memory_fraction", type=float, default=0.8)
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.memory_fraction)
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

    config = _config.get_config(args.config)
    with open(args.data_path, "rb") as f:
        data_samples_list = pickle.load(f)

    # Compute weights: by optimization or use provided
    losses = []
    if args.weights is None:
        if args.optimize_method == "gradient_descent":
            args.weights = optimize_weights_with_gradient_descent(
                args.checkpoints, config, data_samples_list,
                num_iterations=args.num_iterations, learning_rate=args.learning_rate
            )
        elif args.optimize_method == "adaptive_gradient_descent":
            args.weights = optimize_weights_with_adaptive_gradient_descent(
                args.checkpoints, config, data_samples_list,
                num_iterations=args.num_iterations, learning_rate=args.learning_rate
            )
        elif args.optimize_method == "inverse_loss":
            # Weight by inverse loss: worse loss -> smaller weight
            losses = compute_checkpoint_losses(args.checkpoints, config, data_samples_list)
            args.weights = compute_optimal_weights(losses)
        elif args.optimize_method == "greedy":
            args.weights = optimize_weights_greedy(args.checkpoints, config, data_samples_list)
        else:
            raise ValueError(f"Invalid optimization method: {args.optimize_method}")
        print(f"\n✓ Optimized weights: {args.weights}")
    else:
        print(f"\nUsing provided weights: {args.weights}")
        losses = compute_checkpoint_losses(args.checkpoints, config, data_samples_list)

    if len(args.weights) != len(args.checkpoints):
        raise ValueError("Number of weights must match number of checkpoints")

    print("\n" + "=" * 60)
    print("Results:")
    if losses:
        for i, (ckpt, loss) in enumerate(zip(args.checkpoints, losses)):
            print(f"  Ckpt {i+1}: {loss:.6f} (w={args.weights[i]:.4f})")
    print("=" * 60)

    # Weighted average of all checkpoint params
    print("\nMixing parameters...")
    params_list = [load_jax_params(p) for p in args.checkpoints]
    mixed = mix_params(params_list, args.weights)
    del params_list
    gc.collect()
    save_jax_params(mixed, args.output)

    del mixed
    gc.collect()

    # Optionally mix and save normalization stats, then eval mixed ckpt
    print("\nMixing norm_stats...")
    norm_stats_paths = []
    for ckpt_path in args.checkpoints:
        ckpt_root = os.path.dirname(ckpt_path) if not ckpt_path.endswith("/params") else os.path.dirname(os.path.dirname(ckpt_path))
        norm_stats_path = os.path.join(ckpt_root, "norm_stats.json")
        if os.path.exists(norm_stats_path):
            norm_stats_paths.append(norm_stats_path)
    if len(norm_stats_paths) == len(args.checkpoints):
        norm_stats_list = [load_norm_stats(p) for p in norm_stats_paths]
        mixed_norm_stats = mix_norm_stats(norm_stats_list, weights=args.weights)
        save_norm_stats(mixed_norm_stats, os.path.join(args.output, "norm_stats.json"))
        print("\nCleaning GPU memory...")
        jax.clear_caches()
        gc.collect()
        time.sleep(2)
        print("\nTesting mixed checkpoint...")
        mixed_loss = test_mixed_checkpoint_jax(config, args.output, data_samples_list)
        print("\n" + "=" * 60)
        print("Results:")
        if losses:
            for i, (ckpt, loss) in enumerate(zip(args.checkpoints, losses)):
                print(f"  Ckpt {i+1}: {loss:.6f} (w={args.weights[i]:.4f})")
        print(f"  Mixed:  {mixed_loss:.6f}")
        print("=" * 60)
    else:
        logger.warning("Incomplete norm_stats files, skipping test")
    print("\n✓ Completed successfully!")


if __name__ == "__main__":
    main()
