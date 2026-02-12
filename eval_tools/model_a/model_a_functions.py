# model_a_functions.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import mitsuba as mi  # used by your EnvironmentFramework
except Exception:
    mi = None


# ============================================================
# Paths / JSON
# ============================================================

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with open(path, "r") as f:
        return json.load(f)


def resolve_under_project_root(project_root: Path, p: Union[str, Path]) -> str:
    p = Path(p)
    if p.is_absolute():
        return str(p.resolve())
    return str((project_root / p).resolve())


def resolve_cfg_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve cfg paths to absolute for:
      - project_root
      - scene_xml
      - buildings.boxes_json
    """
    cfg2 = dict(cfg)
    project_root = Path(cfg.get("project_root", Path.cwd())).expanduser().resolve()
    cfg2["project_root"] = str(project_root)

    if "scene_xml" in cfg2:
        cfg2["scene_xml"] = resolve_under_project_root(project_root, cfg2["scene_xml"])

    bcfg = cfg2.get("buildings", {})
    if isinstance(bcfg, dict) and bcfg.get("boxes_json"):
        bcfg2 = dict(bcfg)
        bcfg2["boxes_json"] = resolve_under_project_root(project_root, bcfg2["boxes_json"])
        cfg2["buildings"] = bcfg2

    return cfg2


# ============================================================
# Building geometry helpers (FOOTPRINT POLYGON VERSION)
# (copied from training for consistency)
# ============================================================

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
    """Ray casting point-in-polygon test. poly_xy: (M,2)."""
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


def load_building_boxes(path: Union[str, Path]) -> List[PolygonBuilding]:
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
    path = Path(path).expanduser().resolve()
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        recs = data["buildings"] if "buildings" in data else data.get("boxes", [])
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
            footprint_xy = np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)
        else:
            footprint_xy = np.asarray(fp, dtype=np.float32)
            if (
                footprint_xy.ndim != 2
                or footprint_xy.shape[1] != 2
                or footprint_xy.shape[0] < 3
                or not np.isfinite(footprint_xy).all()
            ):
                footprint_xy = np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)

        buildings.append(PolygonBuilding(aabb=bb, footprint_xy=footprint_xy, zmin=float(zmin), zmax=float(zmax)))

    return buildings


def point_in_any_building(p: np.ndarray, buildings: List[PolygonBuilding], margin: float = 0.0) -> bool:
    """
    True if point is inside any building proxy.
    Decision:
      1) AABB broad-phase (fast) using margin
      2) z-range gate (with margin)
      3) XY point-in-polygon on footprint (margin handled mainly by AABB)
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


def get_buildings_from_cfg(cfg: Dict[str, Any]) -> List[PolygonBuilding]:
    bcfg = cfg.get("buildings", {})
    if isinstance(bcfg, dict) and bcfg.get("boxes_json"):
        p = Path(str(bcfg["boxes_json"])).expanduser().resolve()
        if p.exists():
            return load_building_boxes(p)
    return []


# ============================================================
# SB3 model loading
# ============================================================

def get_algo_class(algo_name: str):
    algo = algo_name.strip().upper()
    if algo == "SAC":
        from stable_baselines3 import SAC
        return SAC
    if algo == "PPO":
        from stable_baselines3 import PPO
        return PPO
    if algo == "A2C":
        from stable_baselines3 import A2C
        return A2C
    if algo == "TD3":
        from stable_baselines3 import TD3
        return TD3
    if algo == "DDPG":
        from stable_baselines3 import DDPG
        return DDPG
    if algo == "DQN":
        from stable_baselines3 import DQN
        return DQN
    raise ValueError(f"Unknown algo: {algo_name}")


def load_sb3_model(model_zip_path: Union[str, Path], algo_name: str):
    model_zip_path = Path(model_zip_path).expanduser().resolve()
    AlgoCls = get_algo_class(algo_name)
    return AlgoCls.load(str(model_zip_path))


# ============================================================
# Observation / action helpers (must match training!)
# ============================================================

def get_uav_init_xyz(cfg: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> np.ndarray:
    ucfg = cfg["uav"]
    n_uav = int(ucfg["n_uav"])
    mode = str(ucfg.get("init_mode", "fixed")).lower()

    if mode == "fixed":
        init = np.asarray(ucfg["init_xyz"], dtype=np.float32)
        if init.shape != (n_uav, 3):
            raise ValueError(f"uav.init_xyz shape mismatch: expected {(n_uav,3)}, got {init.shape}")
        return init

    if mode == "random":
        bounds = np.asarray(ucfg["bounds"], dtype=np.float32)  # (3,2)
        lo, hi = bounds[:, 0], bounds[:, 1]
        if rng is None:
            rng = np.random.default_rng(int(cfg.get("seed", 0)))
        xyz = np.stack(
            [
                rng.uniform(lo[0], hi[0], size=(n_uav,)),
                rng.uniform(lo[1], hi[1], size=(n_uav,)),
                rng.uniform(lo[2], hi[2], size=(n_uav,)),
            ],
            axis=1,
        ).astype(np.float32)
        return xyz

    raise ValueError(f"Unknown uav.init_mode: {mode}")


def build_obs(cfg: Dict[str, Any], uav_xyz: np.ndarray, gu_xyz: np.ndarray) -> np.ndarray:
    """
    Must match training obs exactly:
      obs = concat([uav_xyz_flat, gu_xyz_flat]) float32
    """
    n_uav = int(cfg["uav"]["n_uav"])
    n_gu = int(cfg["gu"]["num_gus"])

    uav_xyz = np.asarray(uav_xyz, dtype=np.float32)
    gu_xyz = np.asarray(gu_xyz, dtype=np.float32)

    if uav_xyz.shape != (n_uav, 3):
        raise ValueError(f"uav_xyz shape {uav_xyz.shape} != ({n_uav},3)")
    if gu_xyz.shape != (n_gu, 3):
        raise ValueError(f"gu_xyz shape {gu_xyz.shape} != ({n_gu},3) (check CSV / cfg)")

    return np.concatenate([uav_xyz.reshape(-1), gu_xyz.reshape(-1)], axis=0).astype(np.float32)


def clip_uav_to_bounds(cfg: Dict[str, Any], uav_xyz: np.ndarray) -> np.ndarray:
    bounds = np.asarray(cfg["uav"]["bounds"], dtype=np.float32)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return np.clip(uav_xyz, lo, hi).astype(np.float32)


def count_invalid_positions(
    cfg: Dict[str, Any],
    xyz: np.ndarray,
    buildings: List[PolygonBuilding],
    margin: float,
    bounds_key: str = "uav",
) -> int:
    """Count points that are out of bounds or inside buildings."""
    bounds = np.asarray(cfg[bounds_key]["bounds"], dtype=np.float32)
    lo, hi = bounds[:, 0], bounds[:, 1]

    xyz = np.asarray(xyz, dtype=np.float32)
    invalid = 0
    for p in xyz:
        out_of_bounds = np.any(p < lo) or np.any(p > hi)
        in_building = point_in_any_building(p, buildings, margin=margin) if buildings else False
        if out_of_bounds or in_building:
            invalid += 1
    return int(invalid)


# ============================================================
# GU sampling and trajectory CSV
# ============================================================

def sample_gus_outside_buildings(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    n_gu: Optional[int] = None,
    margin: Optional[float] = None,
    max_attempts: int = 200000,
) -> Tuple[np.ndarray, int]:
    gu_cfg = cfg["gu"]
    if n_gu is None:
        n_gu = int(gu_cfg["num_gus"])

    xy_region = np.asarray(gu_cfg["xy_region"], dtype=np.float32)
    xmin, xmax = float(xy_region[0, 0]), float(xy_region[0, 1])
    ymin, ymax = float(xy_region[1, 0]), float(xy_region[1, 1])
    z = float(gu_cfg.get("height", 1.5))

    buildings = get_buildings_from_cfg(cfg)
    if margin is None:
        margin = float(cfg["uav"].get("building_margin", 0.0))

    gus = np.zeros((n_gu, 3), dtype=np.float32)
    attempts = 0
    placed = 0

    while placed < n_gu:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Failed to sample {n_gu} GUs outside buildings after {attempts} attempts. "
                f"Try smaller margin / larger xy_region / check footprints."
            )
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        p = np.array([x, y, z], dtype=np.float32)

        if buildings and point_in_any_building(p, buildings, margin=margin):
            continue

        gus[placed] = p
        placed += 1

    return gus, int(attempts)


def generate_gu_trajectory_csv(
    cfg_path: Union[str, Path],
    out_csv_path: Union[str, Path],
    n_gu: int,
    T: int,
    max_step_dist: float,
    building_margin: float = 0.0,
    seed: int = 0,
    max_resample_per_step: int = 2000,
) -> Path:
    """
    CSV columns:
      t, gu_id, x, y, z
    """
    cfg = resolve_cfg_paths(load_json(cfg_path))
    rng = np.random.default_rng(seed)

    out_csv_path = Path(out_csv_path).expanduser().resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    buildings = get_buildings_from_cfg(cfg)

    print("boxes_json =", cfg.get("buildings", {}).get("boxes_json", None))
    print("n_buildings =", len(buildings))

    gu_xyz, _ = sample_gus_outside_buildings(cfg, rng, n_gu=n_gu, margin=building_margin)
    gu_cfg = cfg["gu"]
    xy_region = np.asarray(gu_cfg["xy_region"], dtype=np.float32)
    xmin, xmax = float(xy_region[0, 0]), float(xy_region[0, 1])
    ymin, ymax = float(xy_region[1, 0]), float(xy_region[1, 1])
    z_fixed = float(gu_cfg.get("height", 1.5))

    with open(out_csv_path, "w") as f:
        f.write("t,gu_id,x,y,z\n")

        def write_frame(t: int, frame_xyz: np.ndarray):
            for i in range(n_gu):
                x, y, z = frame_xyz[i].tolist()
                f.write(f"{t},{i},{x:.6f},{y:.6f},{z:.6f}\n")

        write_frame(0, gu_xyz)

        for t in range(1, T):
            new_xyz = gu_xyz.copy()
            for i in range(n_gu):
                ok = False
                for _ in range(max_resample_per_step):
                    dx, dy = rng.uniform(-1.0, 1.0, size=(2,))
                    norm = float(np.hypot(dx, dy))
                    if norm < 1e-8:
                        continue
                    dx, dy = dx / norm, dy / norm
                    step = rng.uniform(0.0, max_step_dist)

                    cand = new_xyz[i].copy()
                    cand[0] = float(np.clip(cand[0] + dx * step, xmin, xmax))
                    cand[1] = float(np.clip(cand[1] + dy * step, ymin, ymax))
                    cand[2] = z_fixed

                    if buildings and point_in_any_building(cand, buildings, margin=building_margin):
                        continue

                    ok = True
                    new_xyz[i] = cand
                    break

                if not ok:
                    new_xyz[i] = gu_xyz[i]  # fallback: stay

            gu_xyz = new_xyz
            write_frame(t, gu_xyz)

    return out_csv_path


def load_gu_trajectory_csv(csv_path: Union[str, Path]) -> np.ndarray:
    """
    Load (t,gu_id,x,y,z) -> traj shape (T,n_gu,3)
    """
    import pandas as pd

    csv_path = Path(csv_path).expanduser().resolve()
    df = pd.read_csv(csv_path)

    required = {"t", "gu_id", "x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {sorted(required)}, got {df.columns.tolist()}")

    T = int(df["t"].max()) + 1
    n_gu = int(df["gu_id"].max()) + 1
    traj = np.zeros((T, n_gu, 3), dtype=np.float32)

    # vectorized fill (faster than iterrows)
    t_arr = df["t"].to_numpy(dtype=np.int64)
    i_arr = df["gu_id"].to_numpy(dtype=np.int64)
    traj[t_arr, i_arr, 0] = df["x"].to_numpy(dtype=np.float32)
    traj[t_arr, i_arr, 1] = df["y"].to_numpy(dtype=np.float32)
    traj[t_arr, i_arr, 2] = df["z"].to_numpy(dtype=np.float32)

    return traj


# ============================================================
# Metrics from SINR (independent, no Sionna dependency)
# ============================================================

def compute_metrics_from_sinr(cfg: Dict[str, Any], sinr_db_gu_tx: np.ndarray) -> Dict[str, Any]:
    """
    Metrics you listed:
      - coverage_tau_db, coverage_count, coverage_ratio
      - best_sinr_mean/min/max
      - per_uav_load (based on best-tx assignment), per_uav_load_{min/mean/max}
      - load_var (variance of per-tx load)
    Note:
      TX = UAV + BS (n_tx = n_uav + n_bs)  (consistent with your codebase)
    """
    sinr_db_gu_tx = np.asarray(sinr_db_gu_tx, dtype=np.float32)
    n_gu = int(cfg["gu"]["num_gus"])
    n_uav = int(cfg["uav"]["n_uav"])

    if sinr_db_gu_tx.ndim != 2 or sinr_db_gu_tx.shape[0] != n_gu:
        raise ValueError(f"sinr_db_gu_tx shape must be ({n_gu}, n_tx), got {sinr_db_gu_tx.shape}")

    n_tx = int(sinr_db_gu_tx.shape[1])
    tau = float(cfg.get("reward", {}).get("coverage_tau_db", 5.0))

    best_tx = np.argmax(sinr_db_gu_tx, axis=1)  # (n_gu,)
    best_sinr = sinr_db_gu_tx[np.arange(n_gu), best_tx]

    covered = (best_sinr >= tau)
    coverage_count = int(np.sum(covered))
    coverage_ratio = float(coverage_count) / float(max(n_gu, 1))

    # per-tx loads
    per_tx_load = np.zeros((n_tx,), dtype=np.float32)
    for tx in range(n_tx):
        per_tx_load[tx] = float(np.sum(best_tx == tx))

    # per-uav load is first n_uav
    per_uav_load = per_tx_load[:n_uav].copy() if n_tx >= n_uav else np.zeros((n_uav,), dtype=np.float32)
    load_var = float(np.var(per_tx_load)) if n_tx > 1 else 0.0

    return {
        "coverage_tau_db": float(tau),
        "coverage_count": int(coverage_count),
        "coverage_ratio": float(coverage_ratio),

        "best_sinr_mean": float(np.mean(best_sinr)) if best_sinr.size else 0.0,
        "best_sinr_min": float(np.min(best_sinr)) if best_sinr.size else 0.0,
        "best_sinr_max": float(np.max(best_sinr)) if best_sinr.size else 0.0,

        "load_var": float(load_var),
        "per_uav_load": per_uav_load,

        "per_uav_load_min": float(np.min(per_uav_load)) if per_uav_load.size else 0.0,
        "per_uav_load_mean": float(np.mean(per_uav_load)) if per_uav_load.size else 0.0,
        "per_uav_load_max": float(np.max(per_uav_load)) if per_uav_load.size else 0.0,
    }


# ============================================================
# SINR hook (project-dependent).
# ============================================================

def _get_env_scene_path(cfg: Dict[str, Any]) -> str:
    if "scene_xml" not in cfg or not cfg["scene_xml"]:
        raise ValueError("cfg 缺少 scene_xml 路径（cfg['scene_xml']）")
    return str(Path(cfg["scene_xml"]).expanduser().resolve())


def _as_xyz(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} 必须是 (N,3)，但得到 {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} 含有 NaN/Inf")
    return a


def compute_sinr_db_gu_tx(
    cfg: Dict[str, Any],
    uav_xyz: np.ndarray,
    gu_xyz: np.ndarray,
    *,
    # 这些参数会显著影响 radiomap 耗时；先给保守默认，必要时你自己调小/调大
    max_depth: int = 2,
    num_samples: int = 200000,          # 你原来默认 1,000,000 很慢；先用 2e5 方便跑通
    cell_size: Tuple[float, float] = (2.0, 2.0),  # cell 越大越快但越粗
    # TX 配置
    uav_signal_power_w: float = 1.0,    # 先给常数（W），如果你有更合理值可从 cfg 读
    uav_bandwidth_mbps: float = 50.0,   # 默认跟你 addUAV() 一致
    bs_signal_power_w: float = 1.0,
    bs_bandwidth_mbps: float = 50.0,
    # 运行模式
    verbose: bool = False,
) -> np.ndarray:
    """
    用 inner_loop/model_a/envs/EnvironmentFramework.py 里的 class Environment + Sionna radiomap 来计算 SINR(dB)。

    返回：
      sinr_db: np.ndarray, shape = (n_gu, n_tx)
      其中 n_tx = n_uav + n_bs(tx)（如果 cfg 里提供 bs positions，我们默认也加进去当 TX）
    """
    # --------- 输入检查 ---------
    uav_xyz = _as_xyz(uav_xyz, "uav_xyz")
    gu_xyz = _as_xyz(gu_xyz, "gu_xyz")

    scene_path = _get_env_scene_path(cfg)

    # --------- import 正确类（注意：不是 EnvironmentFramework） ---------
    try:
        from inner_loop.model_a.envs.EnvironmentFramework import Environment  # type: ignore
    except Exception as e:
        raise ImportError(
            "无法 import Environment。请确认 PYTHONPATH/工作目录使得 inner_loop 可被找到。\n"
            f"原始错误: {type(e).__name__}: {e}"
        )

    # --------- 构建环境：不走 position_df_path（避免 person_id 那套 CSV schema） ---------
    # ped_rx=True -> GUs 是 RX；UAV/BS 是 TX（我们会 addUAV/addBaseStation）
    ped_h = float(cfg.get("gu", {}).get("height", 1.5))
    env = Environment(
        scene_path=scene_path,
        position_df_path=None,   # 关键：绕开 createGroundUsers 的 SUMO CSV schema
        time_step=1,
        ped_height=ped_h,
        ped_rx=True,
    )

    # --- 必须：Sionna radiomap 要求 scene.tx_array / rx_array 已设置 ---
    try:
        env.setTransmitterArray(None)  # 设置默认 isotropic/3GPP pattern 的阵列
        env.setReceiverArray(None)
    except Exception as e:
        raise RuntimeError(f"设置 tx/rx array 失败：{type(e).__name__}: {e}")


    # --------- 添加 GUs (RX) ---------
    # 你 addGU 默认 com_type="rx" 正合适
    for p in gu_xyz:
        env.addGU(pos=np.asarray(p, dtype=np.float32), height=ped_h, com_type="rx")

    # --------- 添加 UAVs (TX) ---------
    # 注意：addUAV 内部会创建 Transmitter(name=f"uav{id}") 并 self.n_tx++，
    # 且 id = self.n_tx（当 ped_rx=True 时）。我们按顺序 add 就可以。
    for p in uav_xyz:
        env.addUAV(
            pos=np.asarray(p, dtype=np.float32),
            vel=np.zeros(3, dtype=np.float32),
            bandwidth=float(uav_bandwidth_mbps),
            signal_power=float(uav_signal_power_w),
        )

    # --------- 添加 BS (TX, 可选) ---------
    bs_pos = np.asarray(cfg.get("bs", {}).get("positions", []), dtype=np.float32)
    if bs_pos.size > 0:
        bs_pos = _as_xyz(bs_pos, "bs.positions")
        for p in bs_pos:
            env.addBaseStation(
                device_type="tx",
                pos=np.asarray(p, dtype=np.float32),
                bandwidth=float(bs_bandwidth_mbps),
                signal_power=float(bs_signal_power_w),
            )

    if verbose:
        print(f"[SINR] scene={scene_path}")
        print(f"[SINR] n_rx(GU)={env.n_rx}, n_tx(total)={env.n_tx}, n_bs={env.n_bs}")
        print(f"[SINR] radiomap max_depth={max_depth}, samples={num_samples}, cell_size={cell_size}")

    if len(env.scene.transmitters) == 0:
        raise RuntimeError("Scene has no transmitters：你没有成功 addUAV/addBaseStation 或者它们没有被注册为 Transmitter")
    if len(env.scene.receivers) == 0:
        raise RuntimeError("Scene has no receivers：你没有成功 addGU 或者 GU 没有被注册为 Receiver")


    # --------- 计算 radiomap + SINR ---------
    # 注意：computeRadioMap 可能非常慢；先用较小 num_samples/cell_size 跑通流程。
    radio_map = env.computeRadioMap(max_depth=int(max_depth), num_samples=int(num_samples), cell_size=cell_size)

    # getUserSINRS 返回 shape=(n_rx, n_tx)（按你的实现）
    sinr = env.getUserSINRS(radio_map)

    sinr = np.asarray(sinr, dtype=np.float32)
    if sinr.ndim != 2:
        raise RuntimeError(f"getUserSINRS 返回维度异常：{sinr.shape}")

    # 你这里返回的 radio_map.sinr 通常已经是线性域还是 dB？
    # Sionna 的 radiomap 指标 "sinr" 常见是线性比值；但你在 preview 用 rm_metric="sinr"。
    # 为了避免误判，这里做一个稳健处理：
    # - 如果数值大多在 (0, 100) 这类范围，更像线性；我们转 dB
    # - 如果数值出现大量负数，更像已经是 dB；我们直接返回
    if np.nanmedian(sinr) >= 0 and np.nanmedian(sinr) < 200 and np.nanmin(sinr) >= 0:
        # 认为是线性 SINR
        sinr_db = 10.0 * np.log10(np.clip(sinr, 1e-12, None))
    else:
        # 认为已经是 dB
        sinr_db = sinr

    return sinr_db.astype(np.float32)


# ============================================================
# Policy step (continuous rollout)
# ============================================================

def predict_uav_positions_step(
    cfg: Dict[str, Any],
    model,
    gu_xyz: np.ndarray,
    uav_xyz_start: np.ndarray,
    deterministic: bool = True,
    action_mode: str = "absolute",           # "absolute" or "delta"
    max_uav_step_dist: Optional[float] = None,
    smoothing_alpha: float = 0.0,            # 0 -> no smoothing, 0.2 -> slight inertia
) -> Dict[str, Any]:
    """
    One step:
      obs = concat([uav_xyz_start_flat, gu_xyz_flat])
      action = model.predict(obs)
      uav_xyz_next =
        - absolute: action reshape (n_uav,3)
        - delta: uav_xyz_start + action reshape
      optional: clamp step distance
      optional: smoothing (inertia)
      clip to bounds
      invalid count (bounds + building)
    """
    n_uav = int(cfg["uav"]["n_uav"])
    buildings = get_buildings_from_cfg(cfg)
    margin = float(cfg["uav"].get("building_margin", 0.0))

    obs = build_obs(cfg, uav_xyz=uav_xyz_start, gu_xyz=gu_xyz)
    action, _ = model.predict(obs, deterministic=deterministic)
    action = np.asarray(action, dtype=np.float32).reshape(n_uav, 3)

    if action_mode.lower() == "absolute":
        uav_next = action
    elif action_mode.lower() == "delta":
        uav_next = uav_xyz_start + action
    else:
        raise ValueError(f"Unknown action_mode={action_mode}, expected 'absolute' or 'delta'.")

    # step clamp (optional)
    if max_uav_step_dist is not None and max_uav_step_dist > 0:
        delta = uav_next - uav_xyz_start
        d = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-8
        scale = np.minimum(1.0, float(max_uav_step_dist) / d)
        uav_next = uav_xyz_start + delta * scale

    # smoothing (optional)
    if smoothing_alpha > 0:
        a = float(np.clip(smoothing_alpha, 0.0, 1.0))
        uav_next = (1.0 - a) * uav_next + a * uav_xyz_start

    uav_next = clip_uav_to_bounds(cfg, uav_next)
    invalid = count_invalid_positions(cfg, uav_next, buildings, margin=margin, bounds_key="uav")

    return {
        "obs": obs,
        "uav_xyz_next": uav_next,
        "invalid_uav_count": int(invalid),
    }


# ============================================================
# Continuous evaluation over GU trajectory CSV
# ============================================================

def evaluate_over_trajectory_csv(
    cfg_path: Union[str, Path],
    model_zip_path: Union[str, Path],
    gu_traj_csv: Union[str, Path],
    t_start: int = 0,
    t_end: Optional[int] = None,
    deterministic: bool = True,
    action_mode: str = "absolute",
    max_uav_step_dist: Optional[float] = None,
    smoothing_alpha: float = 0.0,
    save_cache: bool = False,
) -> Tuple["pd.DataFrame", Optional[Dict[str, Any]]]:
    """
    Continuous rollout:
      - t=t_start: uav starts at cfg.uav.init_xyz (or random init)
      - t>t_start: uav starts at previous step uav position (continuous)

    Outputs:
      df: one row per t with metrics + flattened UAV positions
      cache (optional): stores per-t arrays for visualization/debug
    """
    import pandas as pd

    cfg = resolve_cfg_paths(load_json(cfg_path))
    algo = str(cfg.get("sb3", {}).get("algo", "SAC"))
    model = load_sb3_model(model_zip_path, algo)

    traj = load_gu_trajectory_csv(gu_traj_csv)  # (T,n_gu,3)
    T_total = int(traj.shape[0])
    if t_end is None:
        t_end = T_total

    t_start = int(max(0, t_start))
    t_end = int(min(T_total, t_end))
    if t_start >= t_end:
        raise ValueError(f"Invalid time range: t_start={t_start}, t_end={t_end}, T={T_total}")

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    uav_prev = get_uav_init_xyz(cfg, rng=rng)  # t_start init

    rows: List[Dict[str, Any]] = []
    cache: Optional[Dict[str, Any]] = {"uav_xyz": {}, "gu_xyz": {}, "sinr_db_gu_tx": {}} if save_cache else None

    for t in range(t_start, t_end):
        gu_xyz = traj[t].astype(np.float32)

        step = predict_uav_positions_step(
            cfg=cfg,
            model=model,
            gu_xyz=gu_xyz,
            uav_xyz_start=uav_prev,
            deterministic=deterministic,
            action_mode=action_mode,
            max_uav_step_dist=max_uav_step_dist,
            smoothing_alpha=smoothing_alpha,
        )
        uav_next = step["uav_xyz_next"]

        sinr_db_gu_tx = compute_sinr_db_gu_tx(cfg, uav_xyz=uav_next, gu_xyz=gu_xyz)
        metrics = compute_metrics_from_sinr(cfg, sinr_db_gu_tx)

        # flatten UAVs into columns (better than json string)
        flat = uav_next.reshape(-1)
        row: Dict[str, Any] = {
            "t": int(t),
            "invalid_uav_count": int(step["invalid_uav_count"]),
            **metrics,
        }
        n_uav = int(cfg["uav"]["n_uav"])
        for i in range(n_uav):
            row[f"uav{i}_x"] = float(uav_next[i, 0])
            row[f"uav{i}_y"] = float(uav_next[i, 1])
            row[f"uav{i}_z"] = float(uav_next[i, 2])

        rows.append(row)

        if cache is not None:
            cache["uav_xyz"][int(t)] = uav_next
            cache["gu_xyz"][int(t)] = gu_xyz
            cache["sinr_db_gu_tx"][int(t)] = np.asarray(sinr_db_gu_tx, dtype=np.float32)

        # continuous state update
        uav_prev = uav_next

    df = pd.DataFrame(rows)
    return df, cache


def save_eval_csv(df, out_csv_path: Union[str, Path]) -> Path:
    out_csv_path = Path(out_csv_path).expanduser().resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return out_csv_path


# ============================================================
# Visualization helpers
# ============================================================

def _aabb_to_edges(b: AABB) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin, zmin = b.mn.tolist()
    xmax, ymax, zmax = b.mx.tolist()
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float32,
    )
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [corners[i,0], corners[j,0], np.nan]
        ys += [corners[i,1], corners[j,1], np.nan]
        zs += [corners[i,2], corners[j,2], np.nan]
    return np.array(xs), np.array(ys), np.array(zs)


def _collect_scene_points_for_bounds(
    cfg: Dict[str, Any],
    gu_xyz: Optional[np.ndarray],
    uav_xyz: Optional[np.ndarray],
    buildings: List[PolygonBuilding],
) -> np.ndarray:
    pts = []
    if gu_xyz is not None and len(gu_xyz) > 0:
        pts.append(np.asarray(gu_xyz, dtype=np.float32))
    if uav_xyz is not None and len(uav_xyz) > 0:
        pts.append(np.asarray(uav_xyz, dtype=np.float32))

    bs_cfg = cfg.get("bs", {})
    bs_pos = bs_cfg.get("positions", [])
    if bs_pos:
        pts.append(np.asarray(bs_pos, dtype=np.float32))

    if buildings:
        corners = []
        for pb in buildings:
            corners.append(pb.aabb.mn)
            corners.append(pb.aabb.mx)
        pts.append(np.asarray(corners, dtype=np.float32))

    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(pts)


def _make_cube_ranges_from_bounds(
    mn: np.ndarray,
    mx: np.ndarray,
    pad_ratio: float = 0.05,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Equal-length x/y/z ranges using the largest span among x/y/z.
    z range is forced to be non-negative: [0, z_high].
    Also ensures z_high covers actual scene max z.
    """
    mn = np.asarray(mn, dtype=np.float32)
    mx = np.asarray(mx, dtype=np.float32)

    spans = mx - mn
    max_span = float(np.max(spans))
    max_span = max(max_span, 1e-6)

    pad = float(pad_ratio) * max_span
    L = max_span + 2.0 * pad

    cx = float((mn[0] + mx[0]) / 2.0)
    cy = float((mn[1] + mx[1]) / 2.0)

    xr = [cx - L / 2.0, cx + L / 2.0]
    yr = [cy - L / 2.0, cy + L / 2.0]

    # z: no negative, and must include actual mx[2]
    z_high = max(L, float(mx[2] + pad), 1e-6)
    zr = [0.0, z_high]
    return xr, yr, zr


def visualize_scene_3d(
    cfg_path: Union[str, Path],
    gu_xyz: np.ndarray,
    uav_xyz: np.ndarray,
    title: str = "Model A Evaluation",
    show_buildings: bool = True,
    use_scene_cube: bool = True,
    print_scene_sources: bool = True,
):
    import plotly.graph_objects as go

    cfg = resolve_cfg_paths(load_json(cfg_path))

    if print_scene_sources:
        bcfg = cfg.get("buildings", {})
        print(f"[scene] cfg_path = {Path(cfg_path).expanduser().resolve()}")
        print(f"[scene] scene_xml = {cfg.get('scene_xml', None)}")
        if isinstance(bcfg, dict) and bcfg.get("boxes_json"):
            print(f"[scene] buildings.boxes_json = {Path(bcfg['boxes_json']).expanduser().resolve()}")
        else:
            print("[scene] buildings.boxes_json = <not provided>")

    fig = go.Figure()
    buildings = get_buildings_from_cfg(cfg) if show_buildings else []

    if show_buildings and buildings:
        for pb in buildings:
            x, y, z = _aabb_to_edges(pb.aabb)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", showlegend=False))

    bs_cfg = cfg.get("bs", {})
    bs_pos = bs_cfg.get("positions", [])
    if bs_pos:
        bs = np.asarray(bs_pos, dtype=np.float32)
        fig.add_trace(go.Scatter3d(
            x=bs[:, 0], y=bs[:, 1], z=bs[:, 2],
            mode="markers",
            name="BS",
            marker=dict(size=6, symbol="diamond"),
        ))

    gu = np.asarray(gu_xyz, dtype=np.float32)
    fig.add_trace(go.Scatter3d(
        x=gu[:, 0], y=gu[:, 1], z=gu[:, 2],
        mode="markers", name="GU", marker=dict(size=4),
    ))

    uav = np.asarray(uav_xyz, dtype=np.float32)
    fig.add_trace(go.Scatter3d(
        x=uav[:, 0], y=uav[:, 1], z=uav[:, 2],
        mode="markers", name="UAV", marker=dict(size=7, symbol="circle"),
    ))

    if use_scene_cube:
        pts = _collect_scene_points_for_bounds(cfg, gu_xyz=gu, uav_xyz=uav, buildings=buildings)
        if pts.shape[0] > 0:
            mn = np.min(pts, axis=0)
            mx = np.max(pts, axis=0)
            xr, yr, zr = _make_cube_ranges_from_bounds(mn, mx, pad_ratio=0.05)
            fig.update_layout(scene=dict(
                xaxis=dict(range=xr),
                yaxis=dict(range=yr),
                zaxis=dict(range=zr),
                xaxis_title="x", yaxis_title="y", zaxis_title="z",
            ))
        else:
            fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    else:
        fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))

    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), height=700)
    return fig


def visualize_scene_2d(
    cfg_path: Union[str, Path],
    gu_xyz: Optional[np.ndarray] = None,
    uav_xyz: Optional[np.ndarray] = None,
    bs_xyz: Optional[np.ndarray] = None,
    title: str = "Scene 2D",
    show_buildings: bool = True,
    alpha_building: float = 0.25,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    cfg = resolve_cfg_paths(load_json(cfg_path))

    print("[visualize_scene_2d] cfg_path:", str(Path(cfg_path).expanduser().resolve()))
    print("[visualize_scene_2d] scene_xml:", cfg.get("scene_xml"))
    bcfg = cfg.get("buildings", {})
    if isinstance(bcfg, dict):
        print("[visualize_scene_2d] buildings.boxes_json:", bcfg.get("boxes_json"))

    buildings = get_buildings_from_cfg(cfg)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)

    if show_buildings and buildings:
        for b in buildings:
            poly = np.asarray(b.footprint_xy, dtype=np.float32)
            if poly.ndim == 2 and poly.shape[0] >= 3:
                ax.add_patch(MplPolygon(poly, closed=True, fill=True, alpha=float(alpha_building)))

    if gu_xyz is not None:
        gu_xyz = np.asarray(gu_xyz, dtype=np.float32)
        ax.scatter(gu_xyz[:, 0], gu_xyz[:, 1], s=10, label="GU")

    if uav_xyz is not None:
        uav_xyz = np.asarray(uav_xyz, dtype=np.float32)
        ax.scatter(uav_xyz[:, 0], uav_xyz[:, 1], s=60, marker="x", label="UAV")

    if bs_xyz is None:
        bs_cfg = cfg.get("bs", {})
        if isinstance(bs_cfg, dict) and bs_cfg.get("positions"):
            bs_xyz = np.asarray(bs_cfg["positions"], dtype=np.float32)

    if bs_xyz is not None:
        bs_xyz = np.asarray(bs_xyz, dtype=np.float32)
        ax.scatter(bs_xyz[:, 0], bs_xyz[:, 1], s=80, marker="^", label="BS")

    if "gu" in cfg and "xy_region" in cfg["gu"]:
        xy = np.asarray(cfg["gu"]["xy_region"], dtype=np.float32)
        ax.set_xlim(float(xy[0, 0]), float(xy[0, 1]))
        ax.set_ylim(float(xy[1, 0]), float(xy[1, 1]))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.show()


# ============================================================
# 2D GIF for GU trajectory
# ============================================================

def make_gu_trajectory_gif_2d(
    cfg_path: Union[str, Path],
    gu_traj_csv: Union[str, Path],
    out_gif_path: Union[str, Path],
    t_start: int = 0,
    t_end: Optional[int] = None,
    fps: int = 8,
    point_size: int = 12,
    building_linewidth: float = 1.0,
    show_building_aabb: bool = False,
    margin: float = 0.0,
    dpi: int = 120,
) -> Path:
    """
    Render a 2D GIF showing GU movement over time with building footprints.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    import imageio.v2 as imageio
    from io import BytesIO

    cfg = resolve_cfg_paths(load_json(cfg_path))
    buildings = get_buildings_from_cfg(cfg)

    traj = load_gu_trajectory_csv(gu_traj_csv)
    T = int(traj.shape[0])
    if t_end is None:
        t_end = T
    t_start = max(0, int(t_start))
    t_end = min(T, int(t_end))
    if t_start >= t_end:
        raise ValueError(f"Invalid time range: t_start={t_start}, t_end={t_end}, T={T}")

    xy_region = np.asarray(cfg["gu"]["xy_region"], dtype=np.float32)
    xmin, xmax = float(xy_region[0, 0]), float(xy_region[0, 1])
    ymin, ymax = float(xy_region[1, 0]), float(xy_region[1, 1])

    out_gif_path = Path(out_gif_path).expanduser().resolve()
    out_gif_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []

    def add_buildings(ax):
        for b in buildings:
            poly = np.asarray(b.footprint_xy, dtype=np.float32)
            if poly.ndim != 2 or poly.shape[0] < 3:
                continue
            ax.add_patch(MplPolygon(poly, closed=True, fill=False, linewidth=float(building_linewidth)))

        if show_building_aabb:
            for b in buildings:
                mn = b.aabb.mn
                mx = b.aabb.mx
                rect = np.array(
                    [
                        [mn[0] - margin, mn[1] - margin],
                        [mx[0] + margin, mn[1] - margin],
                        [mx[0] + margin, mx[1] + margin],
                        [mn[0] - margin, mx[1] + margin],
                    ],
                    dtype=np.float32,
                )
                ax.add_patch(MplPolygon(rect, closed=True, fill=False, linewidth=0.6))

    for t in range(t_start, t_end):
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        ax.set_title(f"GU Trajectory (t={t})")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        add_buildings(ax)
        gu_xy = traj[t, :, :2]
        ax.scatter(gu_xy[:, 0], gu_xy[:, 1], s=int(point_size))

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

    imageio.mimsave(out_gif_path, frames, fps=int(fps))
    return out_gif_path

def evaluate_over_trajectory_csv_envstep(
    cfg_path: Union[str, Path],
    model_zip_path: Union[str, Path],
    gu_traj_csv: Union[str, Path],
    env_ctor,  # callable: env_ctor(cfg_dict) -> env instance (your Model-A env)
    t_start: int = 0,
    t_end: Optional[int] = None,
    deterministic: bool = True,
    action_mode: str = "absolute",   # keep for compatibility; env should already match training
    save_cache: bool = True,
) -> Tuple["pd.DataFrame", Optional[Dict[str, Any]]]:
    """
    CSV-driven continuous rollout:
      - reset once at t_start (UAV init)
      - for each t: set GU from CSV[t] -> build obs -> policy -> env.step()
      - collect sinr/radiomap from info (no offline recompute)
    """
    import pandas as pd

    cfg = resolve_cfg_paths(load_json(cfg_path))

    # # check gu number for evaluation
    # env = env_ctor(cfg_dict)
    
    # print("env class:", env.__class__.__name__)
    # print("env.n_uav:", getattr(env, "n_uav", None))
    # print("env.n_gu:", getattr(env, "n_gu", None))
    
    # # 常见：env_raw.gus / env_raw.n_rx / env_raw.n_tx
    # raw = getattr(env, "env_raw", None)
    # print("has env_raw:", raw is not None)
    # if raw is not None:
    #     print("env_raw.n_rx:", getattr(raw, "n_rx", None), "env_raw.n_tx:", getattr(raw, "n_tx", None))
    #     print("len(env_raw.gus):", len(getattr(raw, "gus", [])))
        
    algo = str(cfg.get("sb3", {}).get("algo", "SAC"))
    model = load_sb3_model(model_zip_path, algo)

    traj = load_gu_trajectory_csv(gu_traj_csv)  # (T,n_gu,3)
    T_total = int(traj.shape[0])
    if t_end is None:
        t_end = T_total

    t_start = int(max(0, t_start))
    t_end = int(min(T_total, t_end))
    if t_start >= t_end:
        raise ValueError(f"Invalid time range: t_start={t_start}, t_end={t_end}, T={T_total}")

    # Make env (single instance, continuous state)
    env = env_ctor(cfg)

    # force eval behavior: multi-step + return sinr/radiomap
    cfg.setdefault("env", {})
    cfg["env"]["max_steps"] = int(t_end - t_start)
    cfg.setdefault("debug", {})
    cfg["debug"]["return_sinr"] = True
    cfg["debug"]["return_radiomap"] = True

    # reset once
    obs, _ = env.reset(seed=int(cfg.get("seed", 0)))

    rows: List[Dict[str, Any]] = []
    cache: Optional[Dict[str, Any]] = None
    if save_cache:
        cache = {
            "uav_xyz": {},
            "gu_xyz": {},
            "sinr_db_gu_tx": {},
            "radio_map": {},   # heavy; use with care
            "info": {},
        }

    for t in range(t_start, t_end):
        gu_xyz = traj[t].astype(np.float32)

        # set GU from CSV (external truth)
        env.set_gu_xyz(gu_xyz)

        # build obs from current env state (UAV from last step, GU from CSV)
        obs = env._build_obs()

        # policy -> action
        action, _ = model.predict(obs, deterministic=deterministic)

        # env step -> returns sinr/radiomap via info
        obs2, reward, terminated, truncated, info = env.step(action)

        # metrics from TRUE sinr (from step info)
        sinr_db_gu_tx = info.get("sinr_db_gu_tx", None)
        if sinr_db_gu_tx is None:
            raise RuntimeError("env.step() did not return sinr_db_gu_tx in info; set cfg['debug']['return_sinr']=True")

        sinr_db_gu_tx = np.asarray(sinr_db_gu_tx, dtype=np.float32)
        metrics = compute_metrics_from_sinr(cfg, sinr_db_gu_tx)

        uav_xyz = env._get_uav_xyz().astype(np.float32)

        row: Dict[str, Any] = {
            "t": int(t),
            "reward": float(reward),
            "invalid_uav_count": int(info.get("invalid_uav_count", 0)),
            **metrics,
        }

        # flatten UAVs into columns
        n_uav = int(cfg["uav"]["n_uav"])
        for i in range(n_uav):
            row[f"uav{i}_x"] = float(uav_xyz[i, 0])
            row[f"uav{i}_y"] = float(uav_xyz[i, 1])
            row[f"uav{i}_z"] = float(uav_xyz[i, 2])

        # also include reward parts if present
        for k, v in info.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                row[f"rinfo/{k}"] = float(v)

        rows.append(row)

        if cache is not None:
            cache["uav_xyz"][int(t)] = uav_xyz
            cache["gu_xyz"][int(t)] = gu_xyz
            cache["sinr_db_gu_tx"][int(t)] = sinr_db_gu_tx
            if "radio_map" in info:
                cache["radio_map"][int(t)] = info["radio_map"]
            cache["info"][int(t)] = info

        # continue
        obs = obs2

    df = pd.DataFrame(rows)
    return df, cache