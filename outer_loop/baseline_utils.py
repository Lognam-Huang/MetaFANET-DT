# Projects/metaRL_merged/outer_loop/baseline_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import copy
import json

import numpy as np
import pandas as pd


# -------------------------
# I/O utilities
# -------------------------

def load_json(p: Path) -> Dict[str, Any]:
    """
    Load a JSON file into a dict.
    """
    p = Path(p).expanduser().resolve()
    with open(p, "r") as f:
        return json.load(f)


# -------------------------
# Path resolution utilities
# -------------------------

def resolve_path(path_str: str, project_root: Path) -> str:
    """
    Resolve a path string to an absolute path.

    If `path_str` is relative, it will be resolved against `project_root`.
    """
    project_root = Path(project_root).expanduser().resolve()
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return str(p)


def resolve_cfg_paths_for_sionna(
    cfg: Dict[str, Any],
    project_root: Path,
    *,
    inplace: bool = False,
) -> Dict[str, Any]:
    """
    Fix common path fields inside cfg so that Sionna scene loading works regardless of cwd.

    This function searches for common keys that may store a scene XML path and resolves it
    against `project_root` if it is a relative path.

    Parameters
    ----------
    cfg:
        The config dict loaded from JSON.
    project_root:
        Path to the project root, e.g., ~/Projects/metaRL_merged.
    inplace:
        If True, mutate cfg in-place. Otherwise, return a deep-copied cfg.

    Returns
    -------
    cfg_fixed:
        Config dict with resolved scene path(s).
    """
    cfg_fixed = cfg if inplace else copy.deepcopy(cfg)
    project_root = Path(project_root).expanduser().resolve()

    # Common candidate locations for scene path used by EnvironmentFramework/Sionna
    candidate_keys: Tuple[Tuple[str, ...], ...] = (
        ("scene", "scene_path"),
        ("scene", "path"),
        ("scene", "xml_path"),
        ("scene", "scene_file"),
        ("scene", "scene_xml"),
        ("scene_path",),
        ("scene_xml",),
    )

    def _set_nested(d: Dict[str, Any], keys: Tuple[str, ...]) -> bool:
        cur: Any = d
        for k in keys[:-1]:
            if isinstance(cur, dict) and k in cur and isinstance(cur[k], dict):
                cur = cur[k]
            else:
                return False
        last = keys[-1]
        if isinstance(cur, dict) and last in cur and isinstance(cur[last], str) and cur[last].strip():
            cur[last] = resolve_path(cur[last], project_root)
            return True
        return False

    for keys in candidate_keys:
        if _set_nested(cfg_fixed, keys):
            break

    return cfg_fixed


# -------------------------
# Baseline policies
# -------------------------

class FixedUAVBaseline:
    """
    Static baseline: always returns the same UAV positions.
    """

    def __init__(self, init_xyz: np.ndarray):
        self.init_xyz = np.asarray(init_xyz, dtype=np.float32)

    def get_uav_xyz(self, t: int) -> np.ndarray:
        return self.init_xyz


class RandomUAVBaseline:
    """
    Dynamic baseline: samples UAV positions uniformly within bounds each timestep.

    Notes
    -----
    - Uses a stateful RNG, so the sequence is reproducible for the same seed
      given the same call order over timesteps.
    """

    def __init__(self, bounds: np.ndarray, n_uav: int, seed: int = 0):
        self.bounds = np.asarray(bounds, dtype=np.float32)  # shape (3, 2)
        self.n_uav = int(n_uav)
        self.rng = np.random.default_rng(seed)

    def get_uav_xyz(self, t: int) -> np.ndarray:
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        xyz = self.rng.uniform(low=low, high=high, size=(self.n_uav, 3)).astype(np.float32)
        return xyz


# -------------------------
# Evaluation helper
# -------------------------

def eval_baseline_over_traj(
    cfg: Dict[str, Any],
    traj_by_t: Dict[int, np.ndarray],
    baseline: Any,
    t_start: int,
    t_end: int,
    *,
    max_depth: int = 2,
    num_samples: int = 200000,
    cell_size: Tuple[float, float] = (2.0, 2.0),
    uav_signal_power_w: float = 1.0,
    uav_bandwidth_mbps: float = 50.0,
    bs_signal_power_w: float = 1.0,
    bs_bandwidth_mbps: float = 50.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a baseline over a GU trajectory using Sionna-based SINR computation.

    This uses lazy imports to avoid module import errors when the project root is not
    yet on sys.path at import-time.
    """
    # Lazy import (prevents ModuleNotFoundError at baseline_utils import time)
    from eval_tools.model_a.model_a_functions import compute_sinr_db_gu_tx, compute_metrics_from_sinr

    rows = []
    for t in range(t_start, t_end + 1):
        gu_xyz = traj_by_t[int(t)]
        uav_xyz = baseline.get_uav_xyz(t)

        sinr_db = compute_sinr_db_gu_tx(
            cfg=cfg,
            uav_xyz=uav_xyz,
            gu_xyz=gu_xyz,
            max_depth=max_depth,
            num_samples=num_samples,
            cell_size=cell_size,
            uav_signal_power_w=uav_signal_power_w,
            uav_bandwidth_mbps=uav_bandwidth_mbps,
            bs_signal_power_w=bs_signal_power_w,
            bs_bandwidth_mbps=bs_bandwidth_mbps,
            verbose=verbose,
        )

        metrics = compute_metrics_from_sinr(cfg=cfg, sinr_db_gu_tx=sinr_db)

        row = {"t": int(t), "baseline": baseline.__class__.__name__}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                row[k] = float(v)
        rows.append(row)

    return pd.DataFrame(rows)
