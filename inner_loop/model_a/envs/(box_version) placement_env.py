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


def _box_to_minmax(b: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    """
    Normalize a building record into xmin/xmax/ymin/ymax/zmin/zmax.

    Supports:
      - {xmin,xmax,ymin,ymax,zmin,zmax}
      - {center:[cx,cy,cz], size:[sx,sy,sz]}
      - {"min":[x,y,z], "max":[x,y,z]}  (optional legacy)
    """
    # Prefer explicit min/max fields
    if all(k in b for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")):
        xmin, xmax = float(b["xmin"]), float(b["xmax"])
        ymin, ymax = float(b["ymin"]), float(b["ymax"])
        zmin, zmax = float(b["zmin"]), float(b["zmax"])
        return xmin, xmax, ymin, ymax, zmin, zmax

    # center/size
    if all(k in b for k in ("center", "size")):
        cx, cy, cz = map(float, b["center"])
        sx, sy, sz = map(float, b["size"])
        xmin, xmax = cx - sx / 2.0, cx + sx / 2.0
        ymin, ymax = cy - sy / 2.0, cy + sy / 2.0
        zmin, zmax = cz - sz / 2.0, cz + sz / 2.0
        return xmin, xmax, ymin, ymax, zmin, zmax

    # legacy min/max arrays
    if all(k in b for k in ("min", "max")):
        mn = list(map(float, b["min"]))
        mx = list(map(float, b["max"]))
        if len(mn) != 3 or len(mx) != 3:
            raise ValueError(f"Invalid min/max length: min={mn}, max={mx}")
        return mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]

    raise ValueError(f"Building record missing min/max info. keys={list(b.keys())}")


def load_building_boxes(path: str, fmt: str = "boxes_minmax") -> List[AABB]:
    """
    Load building AABBs from JSON and return List[AABB].

    Supports JSON root:
      - list: [ {xmin,xmax,...}, ... ]
      - dict: {"boxes": [ ... ]}

    'fmt' is kept for compatibility but not strictly required because we auto-detect.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        boxes_raw = data.get("boxes", [])
    elif isinstance(data, list):
        boxes_raw = data
    else:
        raise ValueError(f"Unsupported JSON root type: {type(data)} in {path}")

    buildings: List[AABB] = []
    for i, b in enumerate(boxes_raw):
        if not isinstance(b, dict):
            raise ValueError(f"Box[{i}] must be a dict, got {type(b)}")

        xmin, xmax, ymin, ymax, zmin, zmax = _box_to_minmax(b)
        mn = np.asarray([xmin, ymin, zmin], dtype=np.float32)
        mx = np.asarray([xmax, ymax, zmax], dtype=np.float32)

        # basic sanity
        if np.any(mx < mn):
            raise ValueError(f"Box[{i}] has invalid bounds: mn={mn}, mx={mx}")

        buildings.append(AABB(mn=mn, mx=mx, building_id=b.get("building_id", None)))

    return buildings


def point_in_any_building(p: np.ndarray, buildings: List[AABB], margin: float = 0.0) -> bool:
    if not buildings:
        return False
    return any(bb.contains(p, margin=margin) for bb in buildings)


# ----------------------------
# Config dataclasses
# ----------------------------

@dataclass
class RewardCfg:
    coverage_tau_db: float = 5.0
    w_coverage: float = 1.0
    w_sinr_max: float = 0.2
    w_sinr_mean: float = 0.5
    w_sinr_min: float = 0.3
    w_load_var: float = 0.2
    enable_backhaul: bool = False
    backhaul_tau_db: float = 5.0
    w_backhaul: float = 1.0
    w_invalid_pos: float = 50.0  # penalty when UAV is invalid (inside building after sanitize)


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

        # project root first
        self.project_root = Path(cfg.get("project_root", Path.cwd())).expanduser().resolve()

        self.seed_value = int(cfg.get("seed", 0))

        # scene xml
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
        # interpret as "max attempts per GU" (much safer than a global cap)
        self.max_resample_attempts_per_gu = int(gu_cfg.get("max_resample_attempts", 2000))

        # Buildings -> ALWAYS List[AABB]
        bcfg = cfg.get("buildings", {})
        self.buildings: List[AABB] = []
        if bcfg.get("boxes_json"):
            boxes_path = Path(bcfg["boxes_json"])
            if not boxes_path.is_absolute():
                boxes_path = (self.project_root / boxes_path).resolve()
            self.buildings = load_building_boxes(
                str(boxes_path),
                fmt=bcfg.get("format", "boxes_minmax"),
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

        # Absolute positions bounded by uav bounds
        low = np.tile(self.uav_lo, self.n_uav).astype(np.float32)
        high = np.tile(self.uav_hi, self.n_uav).astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(act_dim,), dtype=np.float32)

    # ----------------------------
    # Gymnasium API
    # ----------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = int(seed)

        self._t = 0

        # Rebuild raw env for clean state
        self.env_raw = Environment(
            self.scene_xml,
            position_df_path=None,
            time_step=1,
            ped_height=self.gu_height,
            ped_rx=True,
        )

        # Spawn UAVs
        init_xyz = self._get_uav_init_xyz()
        for i in range(self.n_uav):
            self.env_raw.addUAV(
                mass=10, efficiency=0.8,
                pos=np.asarray(init_xyz[i], dtype=np.float32),
                vel=np.zeros(3, dtype=np.float32),
                color=np.array([1.0, 0.2, 0.2]),
                bandwidth=50, rotor_area=0.5, signal_power=3.0,
            )
        for i in range(len(self.env_raw.uavs)):
            if hasattr(self.env_raw.uavs[i], "lookAt"):
                self.env_raw.uavs[i].lookAt()

        # Base stations
        self._add_base_stations()

        # GUs outside buildings
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

        # One-step placement
        radio_map = self._compute_radiomap()
        if self.n_gu > 0:
            sinr_db_gu_tx = self.env_raw.getUserSINRS(radio_map)
        else:
            sinr_db_gu_tx = np.zeros((0, getattr(self.env_raw, "n_tx", 0)), dtype=np.float32)

        reward, rinfo = self._compute_reward(sinr_db_gu_tx)

        # Penalty for invalid UAVs
        if np.any(invalid_mask):
            reward -= float(self.reward_cfg.w_invalid_pos) * float(np.sum(invalid_mask))

        obs = self._build_obs()
        terminated = False
        truncated = True  # 1-step episode
        info = {**rinfo, "invalid_uav_count": int(np.sum(invalid_mask))}
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ----------------------------
    # Env building utilities
    # ----------------------------

    def _get_uav_init_xyz(self) -> np.ndarray:
        ucfg = self.cfg["uav"]
        mode = ucfg.get("init_mode", "fixed")
        if mode == "fixed":
            init = np.asarray(ucfg["init_xyz"], dtype=np.float32)
            assert init.shape == (self.n_uav, 3)
            return init
        elif mode == "random":
            rng = np.random.default_rng(self.seed_value)
            xyz = np.stack([
                rng.uniform(self.uav_lo[0], self.uav_hi[0], size=(self.n_uav,)),
                rng.uniform(self.uav_lo[1], self.uav_hi[1], size=(self.n_uav,)),
                rng.uniform(self.uav_lo[2], self.uav_hi[2], size=(self.n_uav,)),
            ], axis=1).astype(np.float32)
            return xyz
        else:
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

        aim_mode = self.bs_cfg.get("aim_mode", "none")
        if aim_mode != "none":
            for b in getattr(self.env_raw, "base_stations", []):
                if hasattr(b, "lookAt"):
                    b.lookAt()

    def _spawn_gus_outside_buildings(self):
        rng = np.random.default_rng(self.seed_value)

        xmin, xmax = float(self.gu_xy_region[0, 0]), float(self.gu_xy_region[0, 1])
        ymin, ymax = float(self.gu_xy_region[1, 0]), float(self.gu_xy_region[1, 1])

        gus_added = 0
        while gus_added < self.n_gu:
            ok = False
            # attempts PER GU, not global
            for _ in range(self.max_resample_attempts_per_gu):
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)
                p = np.array([x, y, self.gu_height], dtype=np.float32)

                if self.buildings and point_in_any_building(p, self.buildings, margin=self.building_margin):
                    continue

                self.env_raw.addGU(
                    pos=p,
                    height=self.gu_height,
                    com_type=self.gu_com_type,
                    color=np.array([0.1, 0.9, 0.2], dtype=np.float32),
                    delta_t=1,
                )
                ok = True
                break

            if not ok:
                raise RuntimeError(
                    f"Failed to sample GU#{gus_added+1}/{self.n_gu} outside buildings after "
                    f"{self.max_resample_attempts_per_gu} attempts. "
                    f"Try: shrink buildings / reduce margin / enlarge xy_region / reduce num_gus."
                )

            gus_added += 1

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
        if sinr_db_gu_tx.size == 0:
            coverage_count = 0
            per_uav_load = np.zeros(self.n_uav, dtype=np.float32)
            sinr_score = 0.0
        else:
            best_tx = np.argmax(sinr_db_gu_tx, axis=1)  # (n_gu,)
            best_sinr = sinr_db_gu_tx[np.arange(self.n_gu), best_tx]
            covered = (best_sinr >= self.reward_cfg.coverage_tau_db)
            coverage_count = int(np.sum(covered))

            served_by_uav = np.where(best_tx < self.n_uav, best_tx, -1)

            per_uav_load_list: List[int] = []
            sinr_score = 0.0
            for u in range(self.n_uav):
                idx = np.where(served_by_uav == u)[0]
                per_uav_load_list.append(int(len(idx)))
                if len(idx) > 0:
                    s = best_sinr[idx]
                    sinr_score += (
                        self.reward_cfg.w_sinr_max * float(np.max(s))
                        + self.reward_cfg.w_sinr_mean * float(np.mean(s))
                        + self.reward_cfg.w_sinr_min * float(np.min(s))
                    )
            per_uav_load = np.asarray(per_uav_load_list, dtype=np.float32)

        load_var = float(np.var(per_uav_load)) if len(per_uav_load) > 1 else 0.0

        reward = (
            self.reward_cfg.w_coverage * float(coverage_count)
            + float(sinr_score)
            - self.reward_cfg.w_load_var * float(load_var)
        )

        info = {
            "coverage_count": coverage_count,
            "load_var": load_var,
            "per_uav_load": per_uav_load,
        }
        return float(reward), info

    # ----------------------------
    # Helpers
    # ----------------------------

    def _sanitize_uav_positions(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = np.asarray(action, dtype=np.float32).reshape(self.n_uav, 3)
        xyz = np.clip(a, self.uav_lo, self.uav_hi)

        invalid = np.zeros((self.n_uav,), dtype=bool)
        if not self.buildings:
            return xyz, invalid

        rng = np.random.default_rng(self.seed_value + 999)

        for i in range(self.n_uav):
            if not point_in_any_building(xyz[i], self.buildings, margin=self.building_margin):
                continue

            ok = False
            for _ in range(200):
                cand = np.array([
                    rng.uniform(self.uav_lo[0], self.uav_hi[0]),
                    rng.uniform(self.uav_lo[1], self.uav_hi[1]),
                    rng.uniform(self.uav_lo[2], self.uav_hi[2]),
                ], dtype=np.float32)
                if not point_in_any_building(cand, self.buildings, margin=self.building_margin):
                    xyz[i] = cand
                    ok = True
                    break

            if not ok:
                invalid[i] = True

        return xyz, invalid

    def _get_uav_xyz(self) -> np.ndarray:
        return np.asarray([u.pos for u in self.env_raw.uavs], dtype=np.float32)

    def _get_gu_xyz(self) -> np.ndarray:
        gus = getattr(self.env_raw, "gus", [])
        out = []
        for g in gus:
            if hasattr(g, "getTimestampPosition"):
                out.append(g.getTimestampPosition(getattr(self.env_raw, "cur_step", 0)))
            else:
                out.append(getattr(g, "pos"))
        return np.asarray(out, dtype=np.float32)
