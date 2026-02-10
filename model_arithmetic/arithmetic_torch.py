"""
Mix multiple OpenPI PyTorch checkpoints (model.safetensors) with weighted averaging.

For JAX checkpoints (Orbax/OCDBT), use arithmetic.py instead.

Example usage:
python model_arithmetic/arithmetic_torch.py \
    --config pi05_hang_cloth \
    --data-path hang_cloth_val.pkl \
    --checkpoints /path/to/torch_ckpt1 /path/to/torch_ckpt2 \
    --output /path/to/mixed \
    --optimize_method gradient_descent \
    --use_gpu
"""
import argparse
import gc
import logging
import os
import time
from pathlib import Path
import pickle

import numpy as np
import torch
from tqdm import tqdm

try:
    import safetensors.torch
except ImportError:
    safetensors = None

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
logger = logging.getLogger(__name__)


def resolve_torch_ckpt_path(path: str) -> str:
    """Resolve checkpoint path to directory containing model.safetensors."""
    p = Path(path).resolve()
    # Accept either dir with model.safetensors or .../params
    if (p / "model.safetensors").exists():
        return str(p)
    if p.name == "params" and (p.parent / "model.safetensors").exists():
        return str(p.parent)
    raise FileNotFoundError(f"Invalid PyTorch checkpoint path (no model.safetensors): {p}")


def load_torch_params(checkpoint_path: str) -> dict:
    """Load PyTorch state_dict from model.safetensors as dict[str, np.ndarray] for mixing."""
    if safetensors is None:
        raise ImportError("safetensors required. Install with: pip install safetensors")
    resolved = resolve_torch_ckpt_path(checkpoint_path)
    state = safetensors.torch.load_file(
        str(Path(resolved) / "model.safetensors")
    )
    return {k: v.cpu().numpy() for k, v in state.items()}



def save_torch_params(flat_params: dict, output_dir: str) -> None:
    """Save mixed parameters to model.safetensors (OpenPI PyTorch format)."""
    if safetensors is None:
        raise ImportError("safetensors required. Install with: pip install safetensors")
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / "model.safetensors"
    tensors = {
        k: torch.from_numpy(np.asarray(v, dtype=np.float32))
        for k, v in flat_params.items()
    }
    safetensors.torch.save_file(tensors, str(out_path))
    print(f"✓ Saved PyTorch checkpoint to {out_path}")



def _to_torch_batch(data_samples, device):
    """Convert (obs_dict, actions) batch from numpy/jax to torch on device."""
    obs_dict, actions = data_samples[0], data_samples[1]
    obs_torch = {}

    for k, v in obs_dict.items():
        if isinstance(v, dict):
            obs_torch[k] = {
                kk: torch.from_numpy(np.asarray(vv)).to(device=device)
                for kk, vv in v.items()
            }
        else:
            obs_torch[k] = torch.from_numpy(np.asarray(v)).to(device=device)
    actions_torch = torch.from_numpy(np.asarray(actions)).to(device=device)
    return obs_torch, actions_torch


def _get_torch_norm_stats(checkpoints, config, device):
    """Load norm_stats for first PyTorch checkpoint."""
    ckpt_dir = resolve_torch_ckpt_path(checkpoints[0])
    if os.path.exists(Path(ckpt_dir) / "norm_stats.json"):
        return _normalize.load(ckpt_dir)
    ckpt_root = os.path.dirname(ckpt_dir.rstrip("/"))
    if os.path.exists(Path(ckpt_root) / "norm_stats.json"):
        return _normalize.load(ckpt_root)
    from openpi.training import checkpoints as _checkpoints
    data_config = config.data.create(config.assets_dirs, config.model)
    return _checkpoints.load_norm_stats(Path(ckpt_dir) / "assets", data_config.asset_id)


def compute_checkpoint_losses_torch(checkpoints, config, data_samples_list, device="cuda"):
    """Compute mean loss per checkpoint on validation batches (for inverse_loss weights)."""
    losses = []

    for ckpt_path in checkpoints:
        ckpt_dir = resolve_torch_ckpt_path(ckpt_path)
        ckpt_root = os.path.dirname(ckpt_dir.rstrip("/"))
        if os.path.exists(Path(ckpt_dir) / "norm_stats.json"):
            norm_stats = _normalize.load(ckpt_dir)
        elif os.path.exists(Path(ckpt_root) / "norm_stats.json"):
            norm_stats = _normalize.load(ckpt_root)
        else:
            from openpi.training import checkpoints as _checkpoints
            data_config = config.data.create(config.assets_dirs, config.model)
            norm_stats = _checkpoints.load_norm_stats(Path(ckpt_dir) / "assets", data_config.asset_id)

        policy = _policy_config.create_trained_policy(
            config, ckpt_dir, norm_stats=norm_stats, pytorch_device=device
        )
        ckpt_losses = []

        for data_samples in tqdm(data_samples_list, desc="Computing torch checkpoint losses"):
            obs_torch, actions_torch = _to_torch_batch(data_samples, device)
            observation = _model.Observation.from_dict(obs_torch)
            with torch.no_grad():
                loss_per_element = policy._model.forward(observation, actions_torch)
            ckpt_losses.append(float(loss_per_element.mean().cpu().numpy()))

        print(f"Checkpoint losses for {ckpt_path}: {ckpt_losses}")
        avg_loss = float(np.mean(ckpt_losses))
        losses.append(avg_loss)
        del policy, norm_stats
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"Computed losses: {losses}")
    return losses


def optimize_weights_with_gradient_descent_torch(
    checkpoints, config, data_samples_list, device="cuda",
    num_iterations=50, learning_rate=0.1, print_every=1
):
    """Optimize mixing weights with gradient descent (PyTorch checkpoints)."""
    print("\n" + "=" * 60)
    print("Optimizing weights with gradient descent (PyTorch)...")
    print("=" * 60)

    norm_stats = _get_torch_norm_stats(checkpoints, config, device)
    ckpt_dir = resolve_torch_ckpt_path(checkpoints[0])
    policy = _policy_config.create_trained_policy(
        config, ckpt_dir, norm_stats=norm_stats, pytorch_device=device
    )
    model = policy._model
    params_list = [
        load_torch_params(p) for p in tqdm(checkpoints, desc="Loading checkpoints")
    ]
    n = len(checkpoints)
    keys = list(params_list[0].keys())
    # Optimize in log-space so weights stay on simplex
    log_weights = torch.zeros(
        n, device=device, dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.Adam([log_weights], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations, eta_min=learning_rate * 0.01
    )
    best_loss = float("inf")
    best_weights = None

    for iteration in range(num_iterations):
        weights = torch.softmax(log_weights, dim=0)
        mixed = {}
        for k in keys:
            mixed[k] = sum(
                weights[i].detach()
                * torch.from_numpy(params_list[i][k]).to(device=device)
                for i in range(n)
            )
        # Load mixed params into model and run forward
        model.load_state_dict(mixed, strict=False)
        data_samples = data_samples_list[
            iteration % len(data_samples_list)
        ]
        obs_torch, actions_torch = _to_torch_batch(data_samples, device)
        observation = _model.Observation.from_dict(obs_torch)
        loss = model.forward(observation, actions_torch).mean()
        model.zero_grad()
        loss.backward()
        # Project param gradients onto each ckpt's params -> d(loss)/d(weight_k), then to d/d(log_weights)
        g_k = [0.0] * n
        for name, param in model.named_parameters():
            if param.grad is not None and name in params_list[0]:
                for i in range(n):
                    p_i = torch.from_numpy(params_list[i][name]).to(
                        device=device
                    )
                    g_k[i] += (param.grad.detach() * p_i).sum().item()
        g_k = np.array(g_k, dtype=np.float32)
        weights_np = weights.detach().cpu().numpy()
        g_bar = np.sum(weights_np * g_k)
        grad_log_weights = weights_np * (g_k - g_bar)

        if log_weights.grad is not None:
            log_weights.grad.zero_()

        log_weights.grad = torch.from_numpy(grad_log_weights).to(device=device)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        loss_val = float(loss.detach().cpu().numpy())

        if loss_val < best_loss:
            best_loss = loss_val
            best_weights = weights_np.copy()
        if (iteration + 1) % print_every == 0 or iteration == 0:
            print(f"Iter {iteration + 1}/{num_iterations}: loss={loss_val:.6f}, weights={weights_np}")
        del mixed, loss
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nBest loss: {best_loss:.6f}, Best weights: {best_weights}")
    result = [float(w) for w in best_weights]
    del params_list, policy, norm_stats, model, optimizer, scheduler, log_weights
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def optimize_weights_with_adaptive_gradient_descent_torch(
    checkpoints, config, data_samples_list, device="cuda",
    num_iterations=50, learning_rate=0.1, print_every=1
):
    """Optimize mixing weights with adaptive gradient descent (PyTorch)."""
    print("\n" + "=" * 60)
    print("Optimizing weights with adaptive gradient descent (PyTorch)...")
    print("=" * 60)

    norm_stats = _get_torch_norm_stats(checkpoints, config, device)
    ckpt_dir = resolve_torch_ckpt_path(checkpoints[0])
    policy = _policy_config.create_trained_policy(
        config, ckpt_dir, norm_stats=norm_stats, pytorch_device=device
    )
    model = policy._model
    params_list = [
        load_torch_params(p) for p in tqdm(checkpoints, desc="Loading checkpoints")
    ]
    n = len(checkpoints)
    keys = list(params_list[0].keys())
    log_weights = torch.zeros(
        n, device=device, dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.Adam([log_weights], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations, eta_min=learning_rate * 0.01
    )
    best_loss = float("inf")
    best_weights = None

    for iteration in range(num_iterations):
        weights = torch.softmax(log_weights, dim=0)
        mixed = {}
        for k in keys:
            mixed[k] = sum(
                weights[i].detach()
                * torch.from_numpy(params_list[i][k]).to(device=device)
                for i in range(n)
            )
        # Load mixed params into model and run forward
        model.load_state_dict(mixed, strict=False)
        data_samples = data_samples_list[
            iteration % len(data_samples_list)
        ]
        obs_torch, actions_torch = _to_torch_batch(data_samples, device)
        observation = _model.Observation.from_dict(obs_torch)
        loss = model.forward(observation, actions_torch).mean()
        model.zero_grad()
        loss.backward()

        g_k = [0.0] * n
        for name, param in model.named_parameters():
            if param.grad is not None and name in params_list[0]:
                for i in range(n):
                    p_i = torch.from_numpy(params_list[i][name]).to(
                        device=device
                    )
                    g_k[i] += (param.grad.detach() * p_i).sum().item()
        g_k = np.array(g_k, dtype=np.float32)
        weights_np = weights.detach().cpu().numpy()
        g_bar = np.sum(weights_np * g_k)
        grad_log_weights = weights_np * (g_k - g_bar)
        # Adaptive scale by loss magnitude so step size tracks loss
        scale = (float(loss.detach().cpu().numpy()) / 0.05) ** 2
        grad_log_weights = grad_log_weights * scale
        if log_weights.grad is not None:
            log_weights.grad.zero_()
        log_weights.grad = torch.from_numpy(grad_log_weights).to(device=device)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        loss_val = float(loss.detach().cpu().numpy())

        if loss_val < best_loss:
            best_loss = loss_val
            best_weights = weights_np.copy()
        if (iteration + 1) % print_every == 0 or iteration == 0:
            print(f"Iter {iteration + 1}/{num_iterations}: loss={loss_val:.6f}, weights={weights_np}")
        del mixed, loss
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nBest loss: {best_loss:.6f}, Best weights: {best_weights}")
    result = [float(w) for w in best_weights]
    del params_list, policy, norm_stats, model, optimizer, scheduler, log_weights
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def optimize_weights_greedy_torch(checkpoints, config, data_samples_list, device="cuda"):
    """Greedy optimization for PyTorch checkpoints."""
    print("\n" + "=" * 60)
    print("Optimizing weights with greedy strategy (PyTorch)...")
    print("=" * 60)

    norm_stats = _get_torch_norm_stats(checkpoints, config, device)
    ckpt_dir = resolve_torch_ckpt_path(checkpoints[0])
    policy = _policy_config.create_trained_policy(
        config, ckpt_dir, norm_stats=norm_stats, pytorch_device=device
    )
    model = policy._model
    params_list = [
        load_torch_params(p) for p in tqdm(checkpoints, desc="Loading checkpoints")
    ]
    n = len(checkpoints)
    keys = list(params_list[0].keys())

    def evaluate_combination(indices):
        """Average loss when using only checkpoints at indices (equal weights)."""
        n_sel = len(indices)
        w = np.zeros(n, dtype=np.float32)
        for i in indices:
            w[i] = 1.0 / n_sel
        mixed = {}
        for k in keys:
            mixed[k] = sum(
                w[i]
                * torch.from_numpy(params_list[i][k]).to(device=device)
                for i in range(n)
            )
        model.load_state_dict(mixed, strict=False)
        total_loss = 0.0
        with torch.no_grad():
            for data_samples in data_samples_list:
                obs_torch, actions_torch = _to_torch_batch(data_samples, device)
                observation = _model.Observation.from_dict(obs_torch)
                loss_per = model.forward(observation, actions_torch).mean()
                total_loss += float(loss_per.cpu().numpy())

        return total_loss / len(data_samples_list)

    remaining = list(range(n))
    selected = []
    best_loss = float("inf")
    # Phase 1: pick best single checkpoint
    print("\nEvaluating individual checkpoints...")
    for i in remaining:
        loss = evaluate_combination([i])
        print(f"  Checkpoint {i+1}: loss={loss:.6f}")
        if loss < best_loss:
            best_loss = loss
            selected = [i]

    remaining.remove(selected[0])
    print(f"-> Selected best start: Checkpoint {selected[0]+1} (loss={best_loss:.6f})")
    # Phase 2: greedily add checkpoints that improve loss
    while remaining:
        print(f"\nSearching for best addition to {[i+1 for i in selected]}...")
        iter_best = best_loss
        best_candidate = -1
        for i in remaining:
            loss = evaluate_combination(selected + [i])
            print(f"  + Checkpoint {i+1}: loss={loss:.6f}")
            if loss < iter_best:
                iter_best = loss
                best_candidate = i

        if best_candidate != -1:
            best_loss = iter_best
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            print(f"-> Improvement found! Added Checkpoint {best_candidate+1}. New loss: {best_loss:.6f}")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            print("-> No improvement found. Stopping.")
            break

    final_weights = np.zeros(n)
    final_weights[selected] = 1.0 / len(selected)
    print(f"\nFinal greedy weights: {final_weights}")
    del params_list, policy, norm_stats, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return final_weights.tolist()


def test_mixed_checkpoint_torch(config, checkpoint_path, data_samples_list, device="cuda"):
    """Test mixed PyTorch checkpoint and compute average loss."""
    ckpt_dir = str(Path(checkpoint_path).resolve())
    if os.path.exists(Path(checkpoint_path) / "norm_stats.json"):
        norm_stats = _normalize.load(checkpoint_path)
    else:
        norm_stats = _normalize.load(ckpt_dir)

    policy = _policy_config.create_trained_policy(
        config, ckpt_dir, norm_stats=norm_stats, pytorch_device=device
    )
    avg_loss = 0.0

    for data_samples in data_samples_list:
        obs_torch, actions_torch = _to_torch_batch(data_samples, device)
        observation = _model.Observation.from_dict(obs_torch)
        with torch.no_grad():
            loss_per_element = policy._model.forward(observation, actions_torch)
        avg_loss += float(loss_per_element.mean().cpu().numpy())

    avg_loss /= len(data_samples_list)
    del policy, norm_stats
    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Mix OpenPI PyTorch checkpoints (model.safetensors). Use arithmetic.py for JAX."
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
        choices=["average", "inverse_loss", "gradient_descent", "adaptive_gradient_descent", "greedy"],
    )
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    config = _config.get_config(args.config)
    with open(args.data_path, "rb") as f:
        data_samples_list = pickle.load(f)

    # Compute weights by optimization or use provided
    losses = []
    if args.weights is None:
        if args.optimize_method == "average":
            n = len(args.checkpoints)
            args.weights = [1.0 / n] * n
            print(f"\n✓ Average weights (1/{n} each): {args.weights}")
        elif args.optimize_method == "gradient_descent":
            args.weights = optimize_weights_with_gradient_descent_torch(
                args.checkpoints, config, data_samples_list, device=device,
                num_iterations=args.num_iterations, learning_rate=args.learning_rate
            )
        elif args.optimize_method == "adaptive_gradient_descent":
            args.weights = optimize_weights_with_adaptive_gradient_descent_torch(
                args.checkpoints, config, data_samples_list, device=device,
                num_iterations=args.num_iterations, learning_rate=args.learning_rate
            )
        elif args.optimize_method == "inverse_loss":
            # Weight by inverse loss: worse loss -> smaller weight
            losses = compute_checkpoint_losses_torch(
                args.checkpoints, config, data_samples_list, device=device
            )
            args.weights = compute_optimal_weights(losses)
        elif args.optimize_method == "greedy":
            args.weights = optimize_weights_greedy_torch(
                args.checkpoints, config, data_samples_list, device=device
            )
        else:
            raise ValueError(f"Invalid optimization method: {args.optimize_method}")
        print(f"\n✓ Optimized weights: {args.weights}")
    else:
        print(f"\nUsing provided weights: {args.weights}")
        losses = compute_checkpoint_losses_torch(
            args.checkpoints, config, data_samples_list, device=device
        )

    if len(args.weights) != len(args.checkpoints):
        raise ValueError("Number of weights must match number of checkpoints")

    print("\n" + "=" * 60)
    print("Results:")
    if losses:
        for i, (ckpt, loss) in enumerate(zip(args.checkpoints, losses)):
            print(f"  Ckpt {i+1}: {loss:.6f} (w={args.weights[i]:.4f})")
    print("=" * 60)

    # Weighted average of all checkpoint params, then save
    print("\nMixing parameters...")
    params_list = [load_torch_params(p) for p in args.checkpoints]
    mixed = mix_params(params_list, args.weights)
    del params_list
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    save_torch_params(mixed, args.output)
    del mixed
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Optionally mix norm_stats and run validation on mixed checkpoint
    print("\nMixing norm_stats...")
    norm_stats_paths = []
    for ckpt_path in args.checkpoints:
        ckpt_root = resolve_torch_ckpt_path(ckpt_path)
        norm_stats_path = os.path.join(ckpt_root, "norm_stats.json")
        if os.path.exists(norm_stats_path):
            norm_stats_paths.append(norm_stats_path)
    if len(norm_stats_paths) == len(args.checkpoints):
        norm_stats_list = [load_norm_stats(p) for p in norm_stats_paths]
        mixed_norm_stats = mix_norm_stats(norm_stats_list, weights=args.weights)
        save_norm_stats(mixed_norm_stats, os.path.join(args.output, "norm_stats.json"))
        gc.collect()
        time.sleep(2)
        print("\nTesting mixed checkpoint...")
        mixed_loss = test_mixed_checkpoint_torch(config, args.output, data_samples_list, device=device)
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
