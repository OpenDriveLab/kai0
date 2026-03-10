"""Tests for intervention forcing in discretize_advantage.py."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# discretize_advantage.py is not a package; add its directory to sys.path for import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from discretize_advantage import assign_task_index
from discretize_advantage import assign_task_index_staged
from discretize_advantage import calculate_rewards
from discretize_advantage import get_stage_index


def _make_parquet(tmp_path, name, frame):
    """Write a DataFrame to a parquet file and return the path."""
    path = str(tmp_path / name)
    frame.to_parquet(path, index=False)
    return path


def _make_base_df(n=100, *, include_intervention=True):
    """Create a synthetic DataFrame with low-advantage human frames.

    Human frames (intervention=1) get low absolute_advantage values so
    they would normally be labeled negative by percentile thresholding.
    """
    advantage = np.linspace(0, 1, n, dtype=np.float32)
    intervention = np.zeros(n, dtype=np.int64)
    if include_intervention:
        # Mark frames 0-29 as human (intervention=1), rest as policy
        intervention[:30] = 1
        # Give human frames low advantage (would be below threshold)
        advantage[:30] = np.linspace(0, 0.1, 30, dtype=np.float32)
    return pd.DataFrame({"absolute_advantage": advantage, "intervention": intervention})


def _make_staged_df(n=100, *, include_intervention=True):
    """Create a synthetic DataFrame with stage_progress_gt for staged labeling."""
    frame = _make_base_df(n, include_intervention=include_intervention)
    spg = np.zeros(n, dtype=np.float32)
    spg[n // 2 :] = 0.6  # Above 0.5 -> stage 1
    spg[: n // 2] = 0.2  # Below 0.5 -> stage 0
    frame["stage_progress_gt"] = spg
    return frame


# --- Non-staged path tests ---


def test_intervention_forcing_binary_nonstaged(tmp_path):
    """Human frames should be forced to task_index=1 in binary mode."""
    frame = _make_base_df(100)
    path = _make_parquet(tmp_path, "ep.parquet", frame)

    rewards = calculate_rewards(frame, chunk_size=50, advantage_source="absolute_advantage")
    threshold = float(np.percentile(rewards, 70))

    assign_task_index(path, threshold, chunk_size=50, discretion_type="binary",
                      advantage_source="absolute_advantage")

    result = pd.read_parquet(path)
    ti = result["task_index"].to_numpy()
    intv = result["intervention"].to_numpy()

    assert (ti[intv == 1] == 1).all(), f"Human frames not all positive: {ti[intv == 1]}"
    assert (ti[intv == 0] == 0).any(), "Expected some policy frames to be negative"


def test_intervention_forcing_nslices_nonstaged(tmp_path):
    """Human frames should be forced to highest bin in n_slices mode."""
    n_slices = 5
    frame = _make_base_df(100)
    path = _make_parquet(tmp_path, "ep.parquet", frame)

    rewards = calculate_rewards(frame, chunk_size=50, advantage_source="absolute_advantage")
    step_pct = 100 / n_slices
    boundaries = [float(np.percentile(rewards, step_pct * i)) for i in range(n_slices)]

    assign_task_index(
        path,
        threshold_percentile=0,
        chunk_size=50,
        discretion_type="n_slices",
        percentile_boundaries=boundaries,
        n_slices=n_slices,
        advantage_source="absolute_advantage",
    )

    result = pd.read_parquet(path)
    ti = result["task_index"].to_numpy()
    intv = result["intervention"].to_numpy()

    assert (ti[intv == 1] == n_slices - 1).all(), f"Human frames not at max bin: {ti[intv == 1]}"


# --- Staged path tests ---


def test_intervention_forcing_binary_staged(tmp_path):
    """Human frames should be forced to task_index=1 in staged binary mode."""
    frame = _make_staged_df(100)
    path = _make_parquet(tmp_path, "ep.parquet", frame)

    rewards = calculate_rewards(frame, chunk_size=50, advantage_source="absolute_advantage")
    stage_nums = 2

    stage_rewards: dict[int, list[float]] = {i: [] for i in range(stage_nums)}
    spg_vals = frame["stage_progress_gt"].to_numpy()
    for idx in range(len(rewards)):
        si = get_stage_index(float(spg_vals[idx]), stage_nums)
        stage_rewards[si].append(float(rewards[idx]))

    threshold_by_stage = {
        si: float(np.percentile(stage_rewards[si], 70)) if stage_rewards[si] else 0.0
        for si in range(stage_nums)
    }

    assign_task_index_staged(
        path,
        threshold_percentiles_by_stage=threshold_by_stage,
        percentile_boundaries_by_stage={},
        chunk_size=50,
        discretion_type="binary",
        advantage_source="absolute_advantage",
        stage_nums=stage_nums,
    )

    result = pd.read_parquet(path)
    ti = result["task_index"].to_numpy()
    intv = result["intervention"].to_numpy()

    assert (ti[intv == 1] == 1).all(), f"Human frames not all positive: {ti[intv == 1]}"


def test_intervention_forcing_nslices_staged(tmp_path):
    """Human frames should be forced to highest bin in staged n_slices mode."""
    n_slices = 5
    frame = _make_staged_df(100)
    path = _make_parquet(tmp_path, "ep.parquet", frame)

    rewards = calculate_rewards(frame, chunk_size=50, advantage_source="absolute_advantage")
    stage_nums = 2

    stage_rewards: dict[int, list[float]] = {i: [] for i in range(stage_nums)}
    spg_vals = frame["stage_progress_gt"].to_numpy()
    for idx in range(len(rewards)):
        si = get_stage_index(float(spg_vals[idx]), stage_nums)
        stage_rewards[si].append(float(rewards[idx]))

    boundaries_by_stage = {}
    for si in range(stage_nums):
        if stage_rewards[si]:
            step_pct = 100 / n_slices
            boundaries_by_stage[si] = [
                float(np.percentile(stage_rewards[si], step_pct * j)) for j in range(n_slices)
            ]
        else:
            boundaries_by_stage[si] = [0.0] * n_slices

    assign_task_index_staged(
        path,
        threshold_percentiles_by_stage={},
        percentile_boundaries_by_stage=boundaries_by_stage,
        chunk_size=50,
        discretion_type="n_slices",
        n_slices=n_slices,
        advantage_source="absolute_advantage",
        stage_nums=stage_nums,
    )

    result = pd.read_parquet(path)
    ti = result["task_index"].to_numpy()
    intv = result["intervention"].to_numpy()

    assert (ti[intv == 1] == n_slices - 1).all(), f"Human frames not at max bin: {ti[intv == 1]}"


# --- Backward compatibility ---


def test_no_intervention_column(tmp_path):
    """Both functions should work without error when intervention column is absent."""
    frame = _make_base_df(100, include_intervention=False)
    frame = frame.drop(columns=["intervention"])
    path_ns = _make_parquet(tmp_path, "ep_nonstaged.parquet", frame)

    rewards = calculate_rewards(frame, chunk_size=50, advantage_source="absolute_advantage")
    threshold = float(np.percentile(rewards, 70))

    # Non-staged -- should not raise
    assign_task_index(path_ns, threshold, chunk_size=50, discretion_type="binary",
                      advantage_source="absolute_advantage")
    result = pd.read_parquet(path_ns)
    assert "task_index" in result.columns

    # Staged -- should not raise
    frame_staged = frame.copy()
    frame_staged["stage_progress_gt"] = np.linspace(0, 0.9, len(frame), dtype=np.float32)
    path_st = _make_parquet(tmp_path, "ep_staged.parquet", frame_staged)

    assign_task_index_staged(
        path_st,
        threshold_percentiles_by_stage={0: threshold, 1: threshold},
        percentile_boundaries_by_stage={},
        chunk_size=50,
        discretion_type="binary",
        advantage_source="absolute_advantage",
        stage_nums=2,
    )
    result = pd.read_parquet(path_st)
    assert "task_index" in result.columns
