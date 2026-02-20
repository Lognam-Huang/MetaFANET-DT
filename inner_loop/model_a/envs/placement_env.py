# model_a/envs/placement_env.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .EnvironmentFramework import Environment

try:
    import mitsuba as mi
except Exception:
    mi = None


# ----------------------------
# Building geometry helpers
# ----------------------------

@dataclass
class AABB:
    """Axis-aligned bounding box with (min_xyz, max_xyz)."""
    mn: np.ndarray  # (3,)
    mx: np.ndarray  # (3,)
    building_id: Optional[str] = None

    def contains(self, p: np.ndarray, margin: float = 0.0) -> bool:
        p = np.asarray(p, dtype=np.float32)
        m = float(margin)
        return np.all(p >= (self.mn - m)) and np.all(p <= (self.mx + m))


@dataclass
class PolygonBuilding:
    """
    Building proxy: AABB + 2D footprint polygon + z-range.
    footprint_xy: (M,2) float32 vertices in XY plane.
    """
    aabb: AABB
    footprint_xy: np.ndarray  # (M,2)
    zmin: float
    zmax: float


def _box_to_minmax(b: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    """
    Normalize a building record into xmin/xmax/ymin/ymax/zmin/zmax.

    Supports:
      - {xmin,xmax,ymin,ymax,zmin,zmax}
      - {center:[cx,cy,cz], size:[sx,sy,sz]}
      - {"min":[x,y,z], "max":[x,y,z]}  (optional legacy)
    """
    if all(k in b for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")):
        xmin, xmax = float(b["xmin"]), float(b["xmax"])
        ymin, ymax = float(b["ymin"]), float(b["ymax"])
        zmin, zmax = float(b["zmin"]), float(b["zmax"])
        return xmin, xmax, ymin, ymax, zmin, zmax

    if all(k in b for k in ("center", "size")):
        cx, cy, cz = map(float, b["center"])
        sx, sy, sz = map(float, b["size"])
        xmin, xmax = cx - sx / 2.0, cx + sx / 2.0
        ymin, ymax = cy - sy / 2.0, cy + sy / 2.0
        zmin, zmax = cz - sz / 2.0, cz + sz / 2.0
        return xmin, xmax, ymin, ymax, zmin, zmax

    if all(k in b for k in ("min", "max")):
        mn = list(map(float, b["min"]))
        mx = list(map(float, b["max"]))
        if len(mn) != 3 or len(mx) != 3:
            raise ValueError(f"Invalid min/max length: min={mn}, max={mx}")
        return mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]

    raise ValueError(f"Building record missing min/max info. keys={list(b.keys())}")


def _point_in_poly_2d(x: float, y: float, poly_xy: np.ndarray) -> bool:
    """
    Ray casting point-in-polygon test.
    poly_xy: (M,2)
    Returns True if inside polygon.
    """
    inside = False
    n = int(poly_xy.shape[0])
    if n < 3:
        return False

    x0, y0 = float(poly_xy[-1, 0]), float(poly_xy[-1, 1])
    for i in range(n):
        x1, y1 = float(poly_xy[i, 0]), float(poly_xy[i, 1])

        if ((y1 > y) != (y0 > y)):
            xinters = (x0 - x1) * (y - y1) / (y0 - y1 + 1e-12) + x1
            if xinters > x:
                inside = not inside

        x0, y0 = x1, y1

    return inside


def load_building_boxes(path: str, fmt: str = "auto") -> List[PolygonBuilding]:
    """
    Load buildings from JSON and return List[PolygonBuilding].

    Supports JSON root:
      - dict: {"buildings":[...]}  (new)
      - dict: {"boxes":[...]}      (legacy)
      - list: [ {...}, ... ]       (legacy)

    Each record may have:
      - xmin/xmax/ymin/ymax/zmin/zmax  (preferred)
      - center/size                    (legacy)
      - footprint: [[x,y],...]         (new, optional but recommended)

    If footprint missing/malformed, fallback to AABB rectangle footprint.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "buildings" in data:
            recs = data["buildings"]
        else:
            recs = data.get("boxes", [])
    elif isinstance(data, list):
        recs = data
    else:
        raise ValueError(f"Unsupported JSON root type: {type(data)} in {path}")

    buildings: List[PolygonBuilding] = []
    for i, b in enumerate(recs):
        if not isinstance(b, dict):
            raise ValueError(f"Record[{i}] must be a dict, got {type(b)}")

        xmin, xmax, ymin, ymax, zmin, zmax = _box_to_minmax(b)

        mn = np.asarray([xmin, ymin, zmin], dtype=np.float32)
        mx = np.asarray([xmax, ymax, zmax], dtype=np.float32)

        if np.any(mx < mn):
            raise ValueError(f"Record[{i}] has invalid bounds: mn={mn}, mx={mx}")

        bb = AABB(mn=mn, mx=mx, building_id=b.get("building_id", None))

        fp = b.get("footprint", None)
        if fp is None or (isinstance(fp, list) and len(fp) < 3):
            footprint_xy = np.asarray(
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                dtype=np.float32,
            )
        else:
            footprint_xy = np.asarray(fp, dtype=np.float32)
            if (
                footprint_xy.ndim != 2
                or footprint_xy.shape[1] != 2
                or footprint_xy.shape[0] < 3
                or not np.isfinite(footprint_xy).all()
            ):
                footprint_xy = np.asarray(
                    [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                    dtype=np.float32,
                )

        buildings.append(
            PolygonBuilding(
                aabb=bb,
                footprint_xy=footprint_xy,
                zmin=float(zmin),
                zmax=float(zmax),
            )
        )

    return buildings


def point_in_any_building(p: np.ndarray, buildings: List[PolygonBuilding], margin: float = 0.0) -> bool:
    """
    True if point is inside any building volume proxy.

    Decision:
      1) AABB broad-phase (fast) using margin
      2) z-range gate (with margin)
      3) XY point-in-polygon on footprint (no polygon offset; margin handled mainly by AABB)
    """
    if not buildings:
        return False

    p = np.asarray(p, dtype=np.float32)
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    m = float(margin)

    for b in buildings:
        if not b.aabb.contains(p, margin=m):
            continue
        if z < (b.zmin - m) or z > (b.zmax + m):
            continue
        if _point_in_poly_2d(x, y, b.footprint_xy):
            return True

    return False


# ----------------------------
# Config dataclasses
# ----------------------------

@dataclass
class RewardCfg:
    coverage_tau_db: float = 5.0
    sinr_target_db: float = 8.0 
    w_coverage: float = 1.0
    w_sinr_max: float = 0.2
    w_sinr_mean: float = 0.5
    w_sinr_min: float = 0.3
    w_load_var: float = 0.2
    enable_backhaul: bool = False
    backhaul_tau_db: float = 5.0
    w_backhaul: float = 1.0
    w_invalid_pos: float = 50.0  # penalty per invalid UAV (inside building after sanitize fails)


@dataclass
class RadiomapCfg:
    max_depth: int = 2
    num_samples: int = 20_000
    cell_size: Tuple[float, float] = (5.0, 5.0)


class SionnaPlacementEnv(gym.Env):
    """
    Single-agent, centralized placement environment.

    - Observation: concat([UAV xyz (n_uav*3), GU xyz (n_gu*3)]) float32
    - Action: absolute UAV xyz for all UAVs, shape (n_uav*3,), bounded by uav.bounds
    - Episode: 1-step placement (compute reward once then truncate)
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.project_root = Path(cfg.get("project_root", Path.cwd())).expanduser().resolve()
        self.seed_value = int(cfg.get("seed", 0))

        scene_path = Path(cfg["scene_xml"])
        if not scene_path.is_absolute():
            scene_path = (self.project_root / scene_path).resolve()
        self.scene_xml = str(scene_path)

        # UAV config
        uav_cfg = cfg["uav"]
        self.n_uav = int(uav_cfg["n_uav"])
        self.uav_bounds = np.asarray(uav_cfg["bounds"], dtype=np.float32)  # (3,2)
        self.uav_lo = self.uav_bounds[:, 0]
        self.uav_hi = self.uav_bounds[:, 1]
        self.building_margin = float(uav_cfg.get("building_margin", 0.0))

        # BS config
        self.bs_cfg = cfg.get("bs", {})

        # GU config
        gu_cfg = cfg["gu"]
        self.n_gu = int(gu_cfg["num_gus"])
        self.gu_xy_region = np.asarray(gu_cfg["xy_region"], dtype=np.float32)  # (2,2)
        self.gu_height = float(gu_cfg.get("height", 1.5))
        self.gu_com_type = str(gu_cfg.get("com_type", "rx"))
        self.max_resample_attempts_per_gu = int(gu_cfg.get("max_resample_attempts", 2000))

        # Buildings -> ALWAYS List[PolygonBuilding]
        bcfg = cfg.get("buildings", {})
        self.buildings: List[PolygonBuilding] = []
        if bcfg.get("boxes_json"):
            boxes_path = Path(bcfg["boxes_json"])
            if not boxes_path.is_absolute():
                boxes_path = (self.project_root / boxes_path).resolve()
            self.buildings = load_building_boxes(
                str(boxes_path),
                fmt=bcfg.get("format", "auto"),
            )

        # Reward / radiomap cfg
        self.reward_cfg = RewardCfg(**cfg.get("reward", {}))

        rm = cfg.get("radiomap", {})
        cell_size = rm.get("cell_size", [5.0, 5.0])
        self.radiomap_cfg = RadiomapCfg(
            max_depth=int(rm.get("max_depth", 2)),
            num_samples=int(rm.get("num_samples", 20_000)),
            cell_size=(float(cell_size[0]), float(cell_size[1])),
        )

        # Internal state
        self._t = 0
        self.env_raw: Optional[Environment] = None

        # Spaces
        obs_dim = self.n_uav * 3 + self.n_gu * 3
        act_dim = self.n_uav * 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        low = np.tile(self.uav_lo, self.n_uav).astype(np.float32)
        high = np.tile(self.uav_hi, self.n_uav).astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(act_dim,), dtype=np.float32)

    
    # ----------------------------
    # Sampling helpers (single RNG source)
    # ----------------------------

    def _rng_uniform(self, low: float, high: float) -> float:
        """
        Gymnasium provides self.np_random after reset(seed=...).
        Fallback to a local RNG if reset hasn't been called (should not happen in normal usage).
        """
        rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random.default_rng(self.seed_value)
        return float(rng.uniform(low, high))

    def _sample_point_outside_buildings(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        z_fixed: Optional[float] = None,
        max_tries: int = 2000,
        tag: str = "point",
    ) -> np.ndarray:
        """
        Sample a point uniformly in [lo, hi] that is outside buildings.
        If z_fixed is not None, z is fixed to that value (lo/hi for z ignored).
        """
        lo = np.asarray(lo, dtype=np.float32)
        hi = np.asarray(hi, dtype=np.float32)

        for t in range(int(max_tries)):
            x = self._rng_uniform(lo[0], hi[0])
            y = self._rng_uniform(lo[1], hi[1])
            z = float(z_fixed) if z_fixed is not None else self._rng_uniform(lo[2], hi[2])
            p = np.array([x, y, z], dtype=np.float32)

            if self.buildings and point_in_any_building(p, self.buildings, margin=self.building_margin):
                continue
            return p

        print(
            f"[InitFail] Cannot sample {tag} outside buildings after {max_tries} tries. "
            f"lo={lo.tolist()}, hi={hi.tolist()}, z_fixed={z_fixed}, "
            f"building_margin={self.building_margin}, n_buildings={len(self.buildings)}"
        )
        raise RuntimeError(
            f"Failed to sample {tag} outside buildings after {max_tries} tries. "
            f"Try: reduce building_margin / enlarge region / reduce num entities / verify building boxes."
        )

    # ----------------------------
    # Env building utilities
    # ----------------------------

    def _get_uav_init_xyz(self) -> np.ndarray:
        """
        UAV init:
          - fixed: uses cfg init_xyz; warns if inside building
          - random: rejection sampling to ensure outside buildings (consistent with GU sampling)
        """
        ucfg = self.cfg["uav"]
        mode = ucfg.get("init_mode", "fixed")

        if mode == "fixed":
            init = np.asarray(ucfg["init_xyz"], dtype=np.float32)
            assert init.shape == (self.n_uav, 3)

            if self.buildings:
                bad = []
                for i in range(self.n_uav):
                    if point_in_any_building(init[i], self.buildings, margin=self.building_margin):
                        bad.append(i)
                if bad:
                    print(f"[InitWarn] fixed UAV init positions inside buildings at indices={bad}. "
                          f"Consider adjusting init_xyz or building_margin.")
            return init

        if mode == "random":
            xyz: List[np.ndarray] = []
            for i in range(self.n_uav):
                p = self._sample_point_outside_buildings(
                    lo=self.uav_lo,
                    hi=self.uav_hi,
                    z_fixed=None,
                    max_tries=2000,
                    tag=f"UAV#{i}",
                )
                xyz.append(p)
            return np.asarray(xyz, dtype=np.float32)

        raise ValueError(f"Unknown uav.init_mode={mode}")

    def _add_base_stations(self):
        if not self.bs_cfg:
            return
        positions = self.bs_cfg.get("positions", [])
        if not positions:
            return

        device_type = self.bs_cfg.get("device_type", "tx")
        bandwidth = float(self.bs_cfg.get("bandwidth", 50.0))
        signal_power = float(self.bs_cfg.get("signal_power", 2.0))

        for pos in positions:
            self.env_raw.addBaseStation(
                device_type=device_type,
                pos=np.asarray(pos, dtype=np.float32),
                color=np.random.rand(3),
                bandwidth=bandwidth,
                signal_power=signal_power,
                throughput_capacity=int(self.bs_cfg.get("throughput_capacity", 5e8)),
            )

        if hasattr(self.env_raw, "setTransmitterArray"):
            self.env_raw.setTransmitterArray()
        if hasattr(self.env_raw, "setReceiverArray"):
            self.env_raw.setReceiverArray()

        # Optional: default lookAt() => straight down
        aim_mode = self.bs_cfg.get("aim_mode", "none")
        if aim_mode != "none":
            for b in getattr(self.env_raw, "base_stations", []):
                if hasattr(b, "lookAt"):
                    b.lookAt()

    def _spawn_gus_outside_buildings(self):
        xmin, xmax = float(self.gu_xy_region[0, 0]), float(self.gu_xy_region[0, 1])
        ymin, ymax = float(self.gu_xy_region[1, 0]), float(self.gu_xy_region[1, 1])

        lo = np.array([xmin, ymin, self.gu_height], dtype=np.float32)
        hi = np.array([xmax, ymax, self.gu_height], dtype=np.float32)

        for gi in range(self.n_gu):
            p = self._sample_point_outside_buildings(
                lo=lo,
                hi=hi,
                z_fixed=self.gu_height,
                max_tries=self.max_resample_attempts_per_gu,
                tag=f"GU#{gi}",
            )
            self.env_raw.addGU(
                pos=p,
                height=self.gu_height,
                com_type=self.gu_com_type,
                color=np.array([0.1, 0.9, 0.2], dtype=np.float32),
                delta_t=1,
            )

    # ----------------------------
    # Observation / reward
    # ----------------------------

    def _build_obs(self) -> np.ndarray:
        uav_xyz = self._get_uav_xyz().reshape(-1)
        gu_xyz = self._get_gu_xyz().reshape(-1)
        return np.concatenate([uav_xyz, gu_xyz], axis=0).astype(np.float32)

    def _compute_radiomap(self) -> Any:
        return self.env_raw.computeRadioMap(
            max_depth=int(self.radiomap_cfg.max_depth),
            num_samples=int(self.radiomap_cfg.num_samples),
            cell_size=tuple(self.radiomap_cfg.cell_size),
        )

    
    def _compute_reward(self, sinr_db_gu_tx: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Reward (GU-aggregated):
          - For each GU, pick best TX by SINR (dB): best_sinr[gu] = max_t sinr_db_gu_tx[gu, t]
          - Coverage: fraction of GUs with best_sinr >= coverage_tau_db (dB)
          - Load: variance of per-tx association counts
          - SINR score: compute stats over ALL GUs' best_sinr
              * raw: weighted max/mean/min in dB (for logging only)
              * norm: offset-from-target in dB, with clipping for stability
    
        Assumes sinr_db_gu_tx is in dB (may include very negative values like -300 dB due to eps).
        """
        sinr_db_gu_tx = np.asarray(sinr_db_gu_tx, dtype=np.float32)
        n_tx = int(sinr_db_gu_tx.shape[1]) if (sinr_db_gu_tx.ndim == 2) else 0
    
        # Safe defaults
        coverage_count = 0
        coverage_frac = 0.0
        per_tx_load = np.zeros((n_tx,), dtype=np.float32)
        per_uav_load = np.zeros((self.n_uav,), dtype=np.float32)
    
        sinr_score_raw = 0.0
        sinr_score_norm = 0.0
    
        # Handle degenerate cases
        if self.n_gu <= 0 or n_tx <= 0 or sinr_db_gu_tx.size == 0:
            load_var = 0.0
            reward = (
                self.reward_cfg.w_coverage * float(coverage_frac)
                + float(sinr_score_norm)
                - self.reward_cfg.w_load_var * float(load_var)
            )
            info = {
                "coverage_count": int(coverage_count),
                "coverage_frac": float(coverage_frac),
                "sinr_target_db": float(getattr(self.reward_cfg, "sinr_target_db", 8.0)),
                "sinr_score_raw": float(sinr_score_raw),
                "sinr_score_norm": float(sinr_score_norm),
                "load_var": float(load_var),
                "per_tx_load": per_tx_load,
                "per_uav_load": per_uav_load,
                "per_uav_load_max": 0.0,
                "per_uav_load_mean": 0.0,
                "per_uav_load_min": 0.0,
            }
            return float(reward), info
    
        # --- Best-TX association (per GU) ---
        best_tx = np.argmax(sinr_db_gu_tx, axis=1).astype(np.int64)  # (n_gu,)
        best_sinr = sinr_db_gu_tx[np.arange(self.n_gu), best_tx].astype(np.float32)  # (n_gu,)
    
        # --- Coverage (per GU) ---
        covered = (best_sinr >= float(self.reward_cfg.coverage_tau_db))
        coverage_count = int(np.sum(covered))
        coverage_frac = float(coverage_count) / float(max(self.n_gu, 1))
    
        # --- Load (per TX) ---
        per_tx_load = np.bincount(best_tx, minlength=n_tx).astype(np.float32)
        # "per_uav_load" assumes TX ordering: first n_uav are UAVs.
        per_uav_load = (
            per_tx_load[: self.n_uav].copy()
            if n_tx >= self.n_uav
            else np.zeros((self.n_uav,), dtype=np.float32)
        )
        load_var = float(np.var(per_tx_load)) if per_tx_load.size > 1 else 0.0
    
        # --- SINR score over ALL GUs (raw + normalized/clipped) ---
        # Raw dB stats (for logging/debug; not used directly in final reward unless you choose to)
        s_max = float(np.max(best_sinr))
        s_mean = float(np.mean(best_sinr))
        s_min = float(np.min(best_sinr))
        sinr_score_raw = (
            self.reward_cfg.w_sinr_max * s_max
            + self.reward_cfg.w_sinr_mean * s_mean
            + self.reward_cfg.w_sinr_min * s_min
        )
    
        # Target SINR in dB (interpretation: "0 offset" point)
        tgt = float(getattr(self.reward_cfg, "sinr_target_db", 8.0))
    
        # Clip best_sinr to ignore pathological eps-based floors (e.g., -300 dB)
        # and to limit extremely high outliers. These are still in dB.
        clip_low = -20.0
        clip_high = 40.0
        best_sinr_clip = np.clip(best_sinr, clip_low, clip_high)
    
        # Normalize in *dB offset* space (more meaningful than dB/tgt)
        # offsets: negative => below target, positive => above target
        s_max_off = float(np.max(best_sinr_clip) - tgt)
        s_mean_off = float(np.mean(best_sinr_clip) - tgt)
        s_min_off = float(np.min(best_sinr_clip) - tgt)
    
        # Offset clipping for reward stability
        # You can tune these later; start with symmetric bounds.
        norm_clip_low = -20.0
        norm_clip_high = 20.0
        s_max_n = float(np.clip(s_max_off, norm_clip_low, norm_clip_high))
        s_mean_n = float(np.clip(s_mean_off, norm_clip_low, norm_clip_high))
        s_min_n = float(np.clip(s_min_off, norm_clip_low, norm_clip_high))
    
        sinr_score_norm = (
            self.reward_cfg.w_sinr_max * s_max_n
            + self.reward_cfg.w_sinr_mean * s_mean_n
            + self.reward_cfg.w_sinr_min * s_min_n
        )
    
        # --- Final reward ---
        reward = (
            self.reward_cfg.w_coverage * float(coverage_frac)
            + float(sinr_score_norm)
            - self.reward_cfg.w_load_var * float(load_var)
        )
    
        info = {
            "coverage_count": int(coverage_count),
            "coverage_frac": float(coverage_frac),
    
            "sinr_target_db": float(tgt),
            "sinr_score_raw": float(sinr_score_raw),
            "sinr_score_norm": float(sinr_score_norm),
    
            # Extra debug fields (safe + helpful)
            "sinr_clip_low_db": float(clip_low),
            "sinr_clip_high_db": float(clip_high),
            "sinr_offset_clip_low_db": float(norm_clip_low),
            "sinr_offset_clip_high_db": float(norm_clip_high),
            "best_sinr_db_max": float(s_max),
            "best_sinr_db_mean": float(s_mean),
            "best_sinr_db_min": float(s_min),
    
            "load_var": float(load_var),
            "per_tx_load": per_tx_load,
            "per_uav_load": per_uav_load,
            "per_uav_load_max": float(np.max(per_uav_load)) if per_uav_load.size else 0.0,
            "per_uav_load_mean": float(np.mean(per_uav_load)) if per_uav_load.size else 0.0,
            "per_uav_load_min": float(np.min(per_uav_load)) if per_uav_load.size else 0.0,
        }
        return float(reward), info



    # ----------------------------
    # Helpers
    # ----------------------------

    def _sanitize_uav_positions(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sanitizes absolute UAV positions:
          1) clip to bounds
          2) if inside buildings: attempt to resample outside buildings using the *same* RNG source
          3) if fails: mark invalid[i]=True (still returns clipped xyz)
        """
        a = np.asarray(action, dtype=np.float32).reshape(self.n_uav, 3)
        xyz = np.clip(a, self.uav_lo, self.uav_hi)

        invalid = np.zeros((self.n_uav,), dtype=bool)
        if not self.buildings:
            return xyz, invalid

        for i in range(self.n_uav):
            if not point_in_any_building(xyz[i], self.buildings, margin=self.building_margin):
                continue

            # Try to repair by resampling (global). You can later change to "local jitter then global".
            ok = False
            for _ in range(200):
                cand = np.array([
                    self._rng_uniform(self.uav_lo[0], self.uav_hi[0]),
                    self._rng_uniform(self.uav_lo[1], self.uav_hi[1]),
                    self._rng_uniform(self.uav_lo[2], self.uav_hi[2]),
                ], dtype=np.float32)
                if not point_in_any_building(cand, self.buildings, margin=self.building_margin):
                    xyz[i] = cand
                    ok = True
                    break

            if not ok:
                invalid[i] = True
                print(
                    f"[SanitizeFail] UAV#{i} action/clipped position inside buildings and repair failed. "
                    f"clipped={xyz[i].tolist()}, bounds_lo={self.uav_lo.tolist()}, bounds_hi={self.uav_hi.tolist()}, "
                    f"building_margin={self.building_margin}, n_buildings={len(self.buildings)}"
                )

        return xyz, invalid

    def _get_uav_xyz(self) -> np.ndarray:
        return np.asarray([u.pos for u in self.env_raw.uavs], dtype=np.float32)

    def _get_gu_xyz(self) -> np.ndarray:
        """
        Always return GU positions as (n_gu, 3) float32.
    
        Notes:
          - Your GroundUser stores XY-only in gu.positions and getTimestampPosition()/getCurrentPosition()
            returns shape (2,). So we must append z explicitly.
          - During CSV-driven evaluation, _set_gu_xyz_from_csv() caches z into gu._csv_z (recommended).
          - Fallback z priority: gu._csv_z -> gu.height -> self.gu_height -> cfg default 1.5
        """
        assert self.env_raw is not None, "Call reset() before _get_gu_xyz()."
    
        gus = getattr(self.env_raw, "gus", [])
        n = len(gus)
        out = np.zeros((n, 3), dtype=np.float32)
    
        # robust default z
        default_z = float(getattr(self, "gu_height", 1.5))
    
        t = int(getattr(self.env_raw, "cur_step", 0))
    
        for i, g in enumerate(gus):
            # ---- XY ----
            xy = None
            if hasattr(g, "getTimestampPosition"):
                xy = g.getTimestampPosition(t)
            elif hasattr(g, "getCurrentPosition"):
                xy = g.getCurrentPosition()
            elif hasattr(g, "positions"):
                p = np.asarray(g.positions, dtype=np.float32)
                xy = p[0] if p.ndim == 2 and p.shape[0] > 0 else p.reshape(-1)[:2]
            elif hasattr(g, "pos"):
                xy = getattr(g, "pos")
    
            xy = np.asarray(xy, dtype=np.float32).reshape(-1)
            if xy.size < 2:
                raise ValueError(f"GU[{i}] position has invalid shape: {xy.shape}")
    
            x = float(xy[0])
            y = float(xy[1])
    
            # ---- Z ----
            z = float(getattr(g, "_csv_z", getattr(g, "height", default_z)))
    
            out[i, 0] = x
            out[i, 1] = y
            out[i, 2] = z
    
        return out



    def set_gu_xyz(self, gu_xyz: np.ndarray) -> None:
        """Public setter: call this BEFORE building obs/predict/step in evaluation."""
        self._set_gu_xyz_from_csv(gu_xyz)

    def _set_gu_xyz_from_csv(self, gu_xyz: np.ndarray) -> None:
        """
        Force-update GU positions at current time step using external (CSV-driven) positions.
    
        Accepts:
          gu_xyz: (n_gu,2) or (n_gu,3)
    
        Guarantees:
          - GU XY is updated (gu.positions stays XY-only as required by GroundUser)
          - env will be able to build obs with GU as 3D later (see _get_gu_xyz fix below)
          - If mitsuba (mi) is not available, silently skip gu.device.position update
        """
        assert self.env_raw is not None, "Call reset() before set_gu_xyz()."
    
        gu_xyz = np.asarray(gu_xyz, dtype=np.float32)
        if gu_xyz.ndim != 2 or gu_xyz.shape[0] != int(self.n_gu) or gu_xyz.shape[1] not in (2, 3):
            raise ValueError(f"gu_xyz must be (n_gu,2) or (n_gu,3), got {gu_xyz.shape}, n_gu={self.n_gu}")
    
        if not hasattr(self.env_raw, "gus"):
            raise AttributeError("env_raw has no attribute 'gus'")
        if len(self.env_raw.gus) != int(self.n_gu):
            raise ValueError(f"env_raw.gus length mismatch: expected {self.n_gu}, got {len(self.env_raw.gus)}")
    
        # default z if not provided
        default_z = float(getattr(self, "gu_height", 1.5))
    
        for i in range(int(self.n_gu)):
            x = float(gu_xyz[i, 0])
            y = float(gu_xyz[i, 1])
            z = float(gu_xyz[i, 2]) if gu_xyz.shape[1] == 3 else default_z
    
            gu = self.env_raw.gus[i]
    
            # keep GU internal time_step stable
            if hasattr(gu, "time_step"):
                gu.time_step = 0
    
            # GroundUser.positions is XY-only: store as (1,2)
            if hasattr(gu, "positions"):
                try:
                    gu.positions = np.asarray([[x, y]], dtype=np.float32)
                except Exception:
                    pass
    
            # store z somewhere so _get_gu_xyz can reconstruct 3D even though positions is XY-only
            # (this avoids changing GroundUser implementation)
            try:
                gu._csv_z = z
            except Exception:
                pass
    
            # update device position used by Sionna/Mitsuba if available
            if hasattr(gu, "device") and hasattr(gu.device, "position"):
                # do NOT hard-require mitsuba in evaluation
                if "mi" in globals() and (globals().get("mi", None) is not None):
                    mi_local = globals()["mi"]
                    gu.device.position = mi_local.Point3f([x, y, float(getattr(gu, "height", z))])
                else:
                    # mitsuba not available: skip device update
                    pass

    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = int(seed)
    
        self._t = 0
        self._gu_override_active = False  # optional flag, not required
    
        self.env_raw = Environment(
            self.scene_xml,
            position_df_path=None,
            time_step=1,
            ped_height=self.gu_height,
            ped_rx=True,
        )
    
        # (Recommended) ensure arrays exist before first radiomap
        try:
            self.env_raw.setTransmitterArray(None)
            self.env_raw.setReceiverArray(None)
        except Exception:
            pass
    
        init_xyz = self._get_uav_init_xyz()
        for i in range(self.n_uav):
            self.env_raw.addUAV(
                mass=10,
                efficiency=0.8,
                pos=np.asarray(init_xyz[i], dtype=np.float32),
                vel=np.zeros(3, dtype=np.float32),
                color=np.array([1.0, 0.2, 0.2]),
                bandwidth=50,
                rotor_area=0.5,
                signal_power=3.0,
            )
    
        for i in range(len(self.env_raw.uavs)):
            if hasattr(self.env_raw.uavs[i], "lookAt"):
                self.env_raw.uavs[i].lookAt()
    
        self._add_base_stations()
        self._spawn_gus_outside_buildings()
    
        obs = self._build_obs()
        info: Dict[str, Any] = {}
        return obs, info
    
    
    def step(self, action: np.ndarray):
        assert self.env_raw is not None, "Call reset() before step()."
        self._t += 1
    
        # Action -> absolute UAV positions (n_uav,3)
        target_xyz, invalid_mask = self._sanitize_uav_positions(action)
    
        # Apply UAV positions
        cur_xyz = self._get_uav_xyz()
        for i in range(self.n_uav):
            if hasattr(self.env_raw, "moveAbsUAV"):
                self.env_raw.moveAbsUAV(i, target_xyz[i], target_xyz[i] - cur_xyz[i])
            else:
                self.env_raw.uavs[i].pos = target_xyz[i]
    
        # One-step: compute radiomap + SINR
        radio_map = self._compute_radiomap()
        if self.n_gu > 0:
            sinr_lin_gu_tx = np.asarray(self.env_raw.getUserSINRS(radio_map), dtype=np.float32)
        else:
            sinr_lin_gu_tx = np.zeros((0, getattr(self.env_raw, "n_tx", 0)), dtype=np.float32)
        
        # Convert linear SINR -> dB for reward (avoid log(0))
        eps = 1e-30
        sinr_db_gu_tx = 10.0 * np.log10(np.maximum(sinr_lin_gu_tx, eps))
        
        reward, rinfo = self._compute_reward(sinr_db_gu_tx)
    
        # Invalid UAV penalty
        invalid_uav_count = int(np.sum(invalid_mask))
        if invalid_uav_count > 0:
            reward -= float(self.reward_cfg.w_invalid_pos) * float(invalid_uav_count)
    
        obs = self._build_obs()
    
        # Episode length control (train: 1; eval: e.g., 50)
        env_cfg = self.cfg.get("env", {}) if isinstance(self.cfg, dict) else {}
        max_steps = int(env_cfg.get("max_steps", 1))
        terminated = False
        truncated = bool(self._t >= max_steps)
    
        # Debug return controls
        dbg = self.cfg.get("debug", {}) if isinstance(self.cfg, dict) else {}
        return_radiomap = bool(dbg.get("return_radiomap", False))
        return_sinr = bool(dbg.get("return_sinr", False))
    
        info = {**rinfo, "invalid_uav_count": invalid_uav_count}
    
        if return_sinr:
            info["sinr_db_gu_tx"] = sinr_db_gu_tx  # safe (numpy)
        if return_radiomap:
            info["radio_map"] = radio_map  # heavy; eval-only
    
        return obs, float(reward), bool(terminated), bool(truncated), info

