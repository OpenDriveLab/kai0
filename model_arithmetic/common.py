"""
Shared helpers for model arithmetic (used by both arithmetic.py and arithmetic_torch.py).
"""
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def mix_params(params_list, weights):
    """Weighted average of param dicts. Each param is dict[str, np.ndarray]. Returns dict[str, np.ndarray]."""
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    mixed = {}
    for key in tqdm(params_list[0].keys(), desc="Mixing parameters"):
        stacked = np.stack([np.asarray(p[key], dtype=np.float64) for p in params_list], axis=0)
        mixed[key] = np.average(stacked, axis=0, weights=weights).astype(np.float32)
    return mixed


def load_norm_stats(norm_stats_path: str) -> dict:
    """Load normalization statistics from JSON."""
    with open(norm_stats_path, "r") as f:
        data = json.load(f)
    if "norm_stats" not in data:
        raise ValueError(f"Invalid norm_stats format in {norm_stats_path}")
    return data["norm_stats"]


def mix_norm_stats(norm_stats_list: list, weights: list = None) -> dict:
    """Mix normalization statistics with optional weighting."""
    if len(norm_stats_list) == 1:
        return norm_stats_list[0]
    if weights is None:
        weights = [1.0 / len(norm_stats_list)] * len(norm_stats_list)
    else:
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
    result = {}
    for key in norm_stats_list[0].keys():
        values = [ns[key] for ns in norm_stats_list]
        if isinstance(values[0], dict):
            result[key] = {}
            for stat_key in values[0].keys():
                arrays = [np.array(v[stat_key]) for v in values]
                stacked = np.stack(arrays, axis=0)
                weighted_avg = np.average(stacked, axis=0, weights=weights)
                result[key][stat_key] = weighted_avg.tolist()
        else:
            result[key] = values[0]
    return result


def save_norm_stats(norm_stats: dict, output_path: str) -> None:
    """Save normalization statistics to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"norm_stats": norm_stats}, f, indent=2)


def compute_optimal_weights(losses):
    """Compute optimal weights based on inverse loss."""
    losses = np.array(losses)
    inv_losses = 1.0 / (losses + 1e-8)
    inv_losses = inv_losses ** 2
    weights = inv_losses / inv_losses.sum()
    return weights.tolist()
