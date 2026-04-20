from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger(__name__)

BasePolicy: TypeAlias = _base_policy.BasePolicy

# Reserved transport-layer key in the observation dict for served clients to override
# sample_kwargs (e.g. pass a deterministic noise sample). Leading underscore signals
# "transport-layer field, not a model observation input" — avoids collisions with future
# models that legitimately use observation field names like "noise".
_RESERVED_SAMPLE_KWARGS_KEY = "_sample_kwargs"
_ALLOWED_TRANSPORT_SAMPLE_KWARGS = frozenset({"noise"})


def _normalize_and_pad_prev_chunk(
    raw: np.ndarray,
    *,
    norm_stats: dict[str, _transforms.NormStats],
    use_quantile_norm: bool,
    action_horizon: int,
) -> np.ndarray:
    """Normalize a client-supplied ``prev_action_chunk`` into model space and pad to ``action_horizon``.

    The model's RTC ``sample_actions`` consumes ``prev_action_chunk`` in **model space**
    (post-Normalize), but websocket clients send a raw ``(d, state_dim)`` slice of their
    deploy-space execution buffer. Without this helper the guidance term operates on
    un-normalized inputs — a silent train-deploy contract break.

    Delegates to the same ``transforms.Normalize`` instance the serving pipeline uses so the
    formula (z-score vs quantile) cannot drift. Pads the chunk to the model's
    ``action_horizon`` because the JAX/PyTorch RTC implementations require that shape.
    """
    state_dim = raw.shape[-1]
    action_stats = norm_stats["actions"]
    if state_dim > action_stats.mean.shape[-1]:
        raise ValueError(
            f"prev_action_chunk state_dim={state_dim} exceeds norm_stats['actions'] width "
            f"{action_stats.mean.shape[-1]}; client is sending more joints than the checkpoint knows about."
        )
    normalizer = _transforms.Normalize({"actions": action_stats}, use_quantiles=use_quantile_norm)
    normalized = normalizer({"actions": raw})["actions"]
    d = normalized.shape[0]
    if d < action_horizon:
        pad = np.zeros((action_horizon - d, state_dim), dtype=np.float32)
        normalized = np.concatenate([normalized, pad], axis=0)
    elif d > action_horizon:
        normalized = normalized[:action_horizon]
    return normalized.astype(np.float32, copy=False)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantile_norm: bool = False,
        action_horizon: int | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        if norm_stats is not None and action_horizon is None:
            raise ValueError(
                "Policy(norm_stats=...) requires action_horizon to also be provided; "
                "without it, server-side prev_action_chunk normalization cannot pad to the model's horizon."
            )
        self._norm_stats = norm_stats
        self._use_quantile_norm = use_quantile_norm
        self._action_horizon = action_horizon
        self._rtc_log_emitted = False

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)

        # RTC cheap-path guidance. Client sends a raw (d, state_dim) slice of its blended queue head
        # along with inference_delay; we normalize with the same stats+mode as the serving Normalize
        # transform and pad horizon to the model's action_horizon before forwarding. Both fields must
        # be present together — forwarding prev_action_chunk without inference_delay would silently
        # trip the cheap-path gate with d=0, running the eager loop with no prefix conditioning.
        has_prev = "prev_action_chunk" in obs
        has_delay = "inference_delay" in obs
        if has_prev and not has_delay:
            logger.warning(
                "[rtc_cheap_path] obs has prev_action_chunk but not inference_delay; skipping cheap-path "
                "forwarding to avoid silent d=0 activation. Client must send both fields together."
            )
        elif has_prev and has_delay:
            raw_prev = np.asarray(obs["prev_action_chunk"], dtype=np.float32)
            if self._norm_stats is not None and self._action_horizon is not None:
                prev_chunk = _normalize_and_pad_prev_chunk(
                    raw_prev,
                    norm_stats=self._norm_stats,
                    use_quantile_norm=self._use_quantile_norm,
                    action_horizon=self._action_horizon,
                )
                sample_kwargs["prev_action_chunk"] = prev_chunk
                log_fn = logger.info if not self._rtc_log_emitted else logger.debug
                log_fn(
                    "[rtc] forwarded prev_action_chunk d=%d ah=%d quantile=%s",
                    raw_prev.shape[0], self._action_horizon, self._use_quantile_norm,
                )
                self._rtc_log_emitted = True
            else:
                sample_kwargs["prev_action_chunk"] = raw_prev
            sample_kwargs["inference_delay"] = obs["inference_delay"]
        if "execute_horizon" in obs:
            sample_kwargs["execute_horizon"] = obs["execute_horizon"]
        # Reserved-key transport for sample_kwargs overrides (currently: noise).
        # Explicit `noise=` kwarg (in-process callers) takes precedence over obs-supplied noise.
        sample_kwargs_override = obs.get(_RESERVED_SAMPLE_KWARGS_KEY) or {}
        if not isinstance(sample_kwargs_override, dict):
            raise TypeError(
                f"obs[{_RESERVED_SAMPLE_KWARGS_KEY!r}] must be a dict, "
                f"got {type(sample_kwargs_override).__name__}"
            )
        unknown = set(sample_kwargs_override) - _ALLOWED_TRANSPORT_SAMPLE_KWARGS
        if unknown:
            raise ValueError(
                f"obs[{_RESERVED_SAMPLE_KWARGS_KEY!r}] contains unsupported keys: {sorted(unknown)}; "
                f"allowlist: {sorted(_ALLOWED_TRANSPORT_SAMPLE_KWARGS)}"
            )
        if "noise" in sample_kwargs_override and noise is None:
            noise = np.asarray(sample_kwargs_override["noise"])
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
