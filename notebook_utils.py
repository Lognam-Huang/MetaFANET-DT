import json
import plotly.graph_objects as go
import numpy as np
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class BuildingsData:
    """Normalized in-memory representation for visualization and checks."""
    meta: Dict[str, Any]
    buildings: List[Dict[str, Any]]


def load_buildings_json(json_path: str) -> BuildingsData:
    """
    Load buildings json and normalize to BuildingsData.

    Supports two formats:
      1) {"meta": {...}, "buildings": [ {...}, {...} ]}
      2) [ {...}, {...} ]  (list of building records)

    Returns:
      BuildingsData(meta, buildings)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "buildings" in data:
        meta = data.get("meta", {})
        buildings = data["buildings"]
        if not isinstance(buildings, list):
            raise ValueError(f"`buildings` must be a list in {json_path}")
        return BuildingsData(meta=meta, buildings=buildings)

    if isinstance(data, list):
        return BuildingsData(meta={}, buildings=data)

    raise ValueError(f"Unsupported JSON structure in {json_path}: {type(data)}")


def sanity_check_buildings(
    json_path: str,
    *,
    max_print: int = 10,
) -> Dict[str, Any]:
    """
    Print & return basic stats to quickly verify the JSON looks reasonable.
    """
    bd = load_buildings_json(json_path)
    bs = bd.buildings

    if len(bs) == 0:
        out = {"num_buildings": 0}
        print(out)
        return out

    xs, ys, zs, hs = [], [], [], []
    invalid_fp = 0
    missing_aabb = 0

    for b in bs:
        # AABB
        try:
            xs += [float(b["xmin"]), float(b["xmax"])]
            ys += [float(b["ymin"]), float(b["ymax"])]
            zmin = float(b.get("zmin", 0.0))
            zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))
            zs += [zmin, zmax]
            hs.append(zmax - zmin)
        except Exception:
            missing_aabb += 1

        fp = b.get("footprint", None)
        if not fp or not isinstance(fp, list) or len(fp) < 3:
            invalid_fp += 1

    xs_rng = (min(xs), max(xs)) if xs else (None, None)
    ys_rng = (min(ys), max(ys)) if ys else (None, None)
    zs_rng = (min(zs), max(zs)) if zs else (None, None)

    hs_arr = np.array(hs, dtype=float) if hs else np.array([], dtype=float)
    height_stats = {
        "min": float(hs_arr.min()) if hs_arr.size else None,
        "median": float(np.median(hs_arr)) if hs_arr.size else None,
        "max": float(hs_arr.max()) if hs_arr.size else None,
    }

    out = {
        "num_buildings": len(bs),
        "x_range": xs_rng,
        "y_range": ys_rng,
        "z_range": zs_rng,
        "height_stats": height_stats,
        "invalid_footprints_lt3": invalid_fp,
        "missing_or_bad_aabb": missing_aabb,
        "meta_keys": sorted(list(bd.meta.keys())) if bd.meta else [],
    }

    print(json.dumps(out, indent=2))
    if max_print > 0:
        print("\nSample building records (first few ids):")
        for b in bs[:max_print]:
            bid = b.get("building_id", "<no id>")
            fp_n = len(b.get("footprint", []) or [])
            print(f"  - {bid}, footprint_pts={fp_n}")

    return out


def plot_footprints_2d(
    json_path: str,
    *,
    max_buildings: Optional[int] = 300,
    show_aabb: bool = False,
    show_centers: bool = False,
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
):
    """
    2D top-view plot of building footprints. Optionally overlay AABB and/or centers.

    Usage in notebook:
        from scene_json_viz import plot_footprints_2d
        plot_footprints_2d(".../buildings.json", max_buildings=200, show_aabb=True)
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    bd = load_buildings_json(json_path)
    bs = bd.buildings
    if max_buildings is not None:
        bs = bs[: int(max_buildings)]

    segments = []
    center_pts = []
    rect_segments = []

    for b in bs:
        fp = b.get("footprint", None)
        if fp and isinstance(fp, list) and len(fp) >= 2:
            pts = np.asarray(fp, dtype=float)
            # close polygon
            pts2 = np.vstack([pts, pts[0]])
            seg = np.stack([pts2[:-1], pts2[1:]], axis=1)  # (M,2,2)
            segments.append(seg)

        if show_aabb:
            try:
                xmin, xmax = float(b["xmin"]), float(b["xmax"])
                ymin, ymax = float(b["ymin"]), float(b["ymax"])
                rect = np.array(
                    [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]],
                    dtype=float,
                )
                rect_seg = np.stack([rect[:-1], rect[1:]], axis=1)
                rect_segments.append(rect_seg)
            except Exception:
                pass

        if show_centers:
            c = b.get("center", None)
            if c and isinstance(c, list) and len(c) >= 2:
                center_pts.append([float(c[0]), float(c[1])])

    if not segments and not rect_segments:
        raise RuntimeError("No valid footprints/AABBs found to plot.")

    fig, ax = plt.subplots(figsize=figsize)

    if segments:
        seg_all = np.concatenate(segments, axis=0)
        ax.add_collection(LineCollection(seg_all, linewidths=0.8))

    if rect_segments:
        rect_all = np.concatenate(rect_segments, axis=0)
        ax.add_collection(LineCollection(rect_all, linewidths=0.6, linestyles="--"))

    if center_pts:
        cp = np.asarray(center_pts, dtype=float)
        ax.scatter(cp[:, 0], cp[:, 1], s=6)

    ax.set_aspect("equal", "box")
    ax.autoscale()

    if title is None:
        title = f"Footprints (top view) | {len(bs)} buildings"
        if show_aabb:
            title += " + AABB"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()


def show_buildings_3d_extruded(
    json_path: str,
    *,
    max_buildings: int = 200,
    min_height: float = 0.5,
    use_zmin: float = 0.0,
    prefer_roof_zmax: bool = False,
    skip_on_extrude_error: bool = True,
):
    """
    3D interactive preview by extruding each footprint polygon into a prism.

    Notes:
    - Uses trimesh + shapely internally via trimesh.creation.extrude_polygon
    - Some polygons may fail to extrude (self-intersection, duplicates). By default we skip them.

    Usage:
        from scene_json_viz import show_buildings_3d_extruded
        show_buildings_3d_extruded(".../buildings.json", max_buildings=150)
    """
    import trimesh

    # trimesh expects shapely Polygon; access helper in a robust way:
    try:
        from shapely.geometry import Polygon
    except Exception as e:
        raise RuntimeError(
            "shapely is required for extrusion. Install shapely in your env."
        ) from e

    bd = load_buildings_json(json_path)
    bs = bd.buildings

    scene = trimesh.Scene()
    added = 0
    skipped = 0

    for b in bs:
        if added >= max_buildings:
            break

        fp = b.get("footprint", None)
        if not fp or not isinstance(fp, list) or len(fp) < 3:
            skipped += 1
            continue

        poly_xy = np.asarray(fp, dtype=float)

        # choose height
        zmin = float(b.get("zmin", use_zmin))
        if prefer_roof_zmax and ("roof_zmax" in b):
            zmax = float(b.get("roof_zmax", zmin))
        else:
            zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))

        # if you want strict "extrude to ground", override zmin:
        zmin = float(use_zmin)
        height = max(0.0, zmax - zmin)

        if height < float(min_height):
            skipped += 1
            continue

        try:
            poly = Polygon(poly_xy)
            if not poly.is_valid or poly.area <= 0:
                skipped += 1
                continue

            m = trimesh.creation.extrude_polygon(poly, height=height)
            m.apply_translation([0.0, 0.0, zmin])

            name = b.get("building_id", f"b{added:03d}")
            scene.add_geometry(m, node_name=name)
            added += 1
        except Exception:
            if skip_on_extrude_error:
                skipped += 1
                continue
            raise

    print(f"[3D] added={added}, skipped={skipped}, source={json_path}")
    return scene.show()

def debug_extrude_candidates(
    json_path: str,
    *,
    max_check: int = 200,
    min_height: float = 1.0,
    use_zmin: float = 0.0,
    prefer_roof_zmax: bool = False,
):
    """
    Diagnose why buildings are skipped in extrusion pipeline.
    Prints counts by reason.
    """
    import numpy as np
    from shapely.geometry import Polygon

    bd = load_buildings_json(json_path)
    bs = bd.buildings[:max_check]

    reasons = {
        "no_footprint_or_lt3": 0,
        "nonfinite_xy": 0,
        "polygon_invalid_or_area0": 0,
        "height_too_small": 0,
        "extrude_exception": 0,
        "ok": 0,
    }

    # quick sample to show a few problematic items
    samples = {k: [] for k in reasons.keys()}

    import trimesh

    for i, b in enumerate(bs):
        bid = b.get("building_id", f"idx_{i}")
        fp = b.get("footprint", None)

        if not fp or not isinstance(fp, list) or len(fp) < 3:
            reasons["no_footprint_or_lt3"] += 1
            if len(samples["no_footprint_or_lt3"]) < 5:
                samples["no_footprint_or_lt3"].append(bid)
            continue

        xy = np.asarray(fp, dtype=float)
        if not np.isfinite(xy).all():
            reasons["nonfinite_xy"] += 1
            if len(samples["nonfinite_xy"]) < 5:
                samples["nonfinite_xy"].append(bid)
            continue

        zmin = float(use_zmin)
        if prefer_roof_zmax and ("roof_zmax" in b):
            zmax = float(b.get("roof_zmax", zmin))
        else:
            zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))

        height = max(0.0, zmax - zmin)
        if height < float(min_height):
            reasons["height_too_small"] += 1
            if len(samples["height_too_small"]) < 5:
                samples["height_too_small"].append((bid, height))
            continue

        poly = Polygon(xy)
        if (not poly.is_valid) or (poly.area <= 0):
            reasons["polygon_invalid_or_area0"] += 1
            if len(samples["polygon_invalid_or_area0"]) < 5:
                samples["polygon_invalid_or_area0"].append(bid)
            continue

        # try extrude
        try:
            _ = trimesh.creation.extrude_polygon(poly, height=height)
            reasons["ok"] += 1
        except Exception:
            reasons["extrude_exception"] += 1
            if len(samples["extrude_exception"]) < 5:
                samples["extrude_exception"].append(bid)

    print("=== Extrude candidate diagnostics ===")
    print("checked:", len(bs))
    for k, v in reasons.items():
        print(f"{k}: {v}")

    print("\n--- samples ---")
    for k, arr in samples.items():
        if arr:
            print(k, ":", arr)

def debug_one_extrude(json_path: str, building_id: str, use_zmin=0.0, prefer_roof_zmax=False):
    import numpy as np
    import trimesh
    from shapely.geometry import Polygon

    bd = load_buildings_json(json_path)
    b = next(x for x in bd.buildings if x.get("building_id") == building_id)

    fp = np.asarray(b["footprint"], dtype=float)
    poly = Polygon(fp)

    zmin = float(use_zmin)
    zmax = float(b.get("roof_zmax" if prefer_roof_zmax else "zmax", zmin))
    height = zmax - zmin

    print("poly valid:", poly.is_valid, "area:", poly.area, "height:", height)

    m = trimesh.creation.extrude_polygon(poly, height=height)
    m.apply_translation([0, 0, zmin])
    scene = trimesh.Scene()
    scene.add_geometry(m)
    return scene.show()

# ---------------------------
# Scene statistics helpers
# ---------------------------

from pathlib import Path
import os
import xml.etree.ElementTree as ET
import pandas as pd


DEFAULT_HEIGHT_BINS = [0, 5, 10, 20, 30, 50, 80, 120, 200, np.inf]
DEFAULT_HEIGHT_BIN_LABELS = [
    "[0,5)", "[5,10)", "[10,20)", "[20,30)", "[30,50)", "[50,80)",
    "[80,120)", "[120,200)", "[200,inf)"
]


def summarize_buildings_json(
    json_path: str,
    *,
    height_bins: Optional[Sequence[float]] = None,
    height_bin_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Summarize one buildings JSON file:
      - building count
      - XY extent
      - height stats (min/median/p90/max)
      - footprint points mean
      - height-bin distribution

    Returns a flat dict that can be assembled into a pandas DataFrame.
    """
    if height_bins is None:
        height_bins = DEFAULT_HEIGHT_BINS
    if height_bin_labels is None:
        height_bin_labels = DEFAULT_HEIGHT_BIN_LABELS

    bd = load_buildings_json(str(json_path))
    bs = bd.buildings

    xs_min, xs_max, ys_min, ys_max = [], [], [], []
    zmins, zmaxs, heights = [], [], []
    footprint_pts = []
    bad_records = 0

    for b in bs:
        if not isinstance(b, dict):
            bad_records += 1
            continue

        try:
            xmin = float(b["xmin"]); xmax = float(b["xmax"])
            ymin = float(b["ymin"]); ymax = float(b["ymax"])
            zmin = float(b.get("zmin", 0.0))
            zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))
        except Exception:
            bad_records += 1
            continue

        if not (np.isfinite([xmin, xmax, ymin, ymax, zmin, zmax]).all()):
            bad_records += 1
            continue

        xs_min.append(xmin); xs_max.append(xmax)
        ys_min.append(ymin); ys_max.append(ymax)
        zmins.append(zmin); zmaxs.append(zmax)
        heights.append(max(0.0, zmax - zmin))

        fp = b.get("footprint", [])
        footprint_pts.append(len(fp) if isinstance(fp, list) else 0)

    n_total = len(bs)
    n_valid = len(heights)

    scene_name = Path(str(json_path)).stem.replace("_buildings", "")

    row: Dict[str, Any] = {
        "file": Path(str(json_path)).name,
        "scene": scene_name,
        "n_buildings_total": int(n_total),
        "n_buildings_valid": int(n_valid),
        "bad_records": int(n_total - n_valid),
    }

    if n_valid == 0:
        # fill NaNs / zeros
        row.update({
            "x_min": np.nan, "x_max": np.nan, "y_min": np.nan, "y_max": np.nan,
            "x_range": np.nan, "y_range": np.nan, "area_xy": np.nan,
            "zmin_min": np.nan, "zmax_max": np.nan,
            "height_min": np.nan, "height_p50": np.nan, "height_p90": np.nan, "height_max": np.nan,
            "footprint_pts_mean": np.nan,
        })
        for lab in height_bin_labels:
            row[f"bin_cnt_{lab}"] = 0
            row[f"bin_pct_{lab}"] = 0.0
        return row

    xs_min = np.asarray(xs_min); xs_max = np.asarray(xs_max)
    ys_min = np.asarray(ys_min); ys_max = np.asarray(ys_max)
    zmins = np.asarray(zmins); zmaxs = np.asarray(zmaxs)
    heights = np.asarray(heights, dtype=float)
    footprint_pts = np.asarray(footprint_pts, dtype=float)

    x_min_all = float(xs_min.min()); x_max_all = float(xs_max.max())
    y_min_all = float(ys_min.min()); y_max_all = float(ys_max.max())
    x_range = x_max_all - x_min_all
    y_range = y_max_all - y_min_all

    counts, _ = np.histogram(heights, bins=np.asarray(height_bins, dtype=float))
    pcts = counts / max(1, len(heights))

    row.update({
        "x_min": x_min_all, "x_max": x_max_all,
        "y_min": y_min_all, "y_max": y_max_all,
        "x_range": float(x_range),
        "y_range": float(y_range),
        "area_xy": float(x_range * y_range),
        "zmin_min": float(zmins.min()),
        "zmax_max": float(zmaxs.max()),
        "height_min": float(np.min(heights)),
        "height_p50": float(np.percentile(heights, 50)),
        "height_p90": float(np.percentile(heights, 90)),
        "height_max": float(np.max(heights)),
        "footprint_pts_mean": float(np.mean(footprint_pts)),
    })

    for lab, c, p in zip(height_bin_labels, counts.tolist(), pcts.tolist()):
        row[f"bin_cnt_{lab}"] = int(c)
        row[f"bin_pct_{lab}"] = float(p)

    return row


def summarize_scenes_in_dir(
    scen_dir: Union[str, Path],
    *,
    pattern: str = "*_buildings.json",
    sort_by: str = "n_buildings_valid",
    descending: bool = True,
) -> "pd.DataFrame":
    """
    Scan a directory for *_buildings.json and return a summary DataFrame.
    """
    scen_dir = Path(scen_dir).expanduser()
    files = sorted(scen_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} under {scen_dir}")

    rows: List[Dict[str, Any]] = []
    for p in files:
        try:
            rows.append(summarize_buildings_json(str(p)))
        except Exception as e:
            rows.append({"file": p.name, "scene": p.stem.replace("_buildings", ""), "error": repr(e)})

    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort_values([sort_by, "scene"], ascending=[not descending, True]).reset_index(drop=True)
    else:
        df = df.sort_values(["scene"], ascending=[True]).reset_index(drop=True)
    return df


# ---------------------------
# XML / mesh sanity helpers
# ---------------------------

def parse_scene_xml_summary(xml_file: str) -> ET.Element:
    """
    Parse an XML file and print a lightweight summary.
    Returns the XML root element.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    names = []
    for elem in root.iter():
        name = elem.attrib.get("name")
        if name is not None:
            names.append(name)

    print("--- XML summary ---")
    print("root tag:", root.tag)
    print("num named objects:", len(names))
    print("first 10 names:", names[:10])
    return root


def compute_mesh_dir_bounds(mesh_dir: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute global bounds for all mesh files in a directory.
    Returns: (min_xyz, max_xyz, mesh_file_count)
    """
    import trimesh

    total_min = np.array([float("inf")] * 3)
    total_max = np.array([float("-inf")] * 3)
    mesh_counts = 0

    for file in os.listdir(mesh_dir):
        if not file.lower().endswith((".obj", ".stl", ".ply")):
            continue

        path = os.path.join(mesh_dir, file)
        try:
            scene_or_mesh = trimesh.load(path)
            if isinstance(scene_or_mesh, trimesh.Scene):
                mesh = scene_or_mesh.dump(concatenate=True)
            else:
                mesh = scene_or_mesh

            total_min = np.minimum(total_min, mesh.bounds[0])
            total_max = np.maximum(total_max, mesh.bounds[1])
            mesh_counts += 1
        except Exception:
            pass

    dims = total_max - total_min
    print("--- Mesh dir stats ---")
    print("mesh files:", mesh_counts)
    print("span (X,Y,Z):", dims)
    print("center:", (total_max + total_min) / 2)
    return total_min, total_max, mesh_counts


def build_buildings_scene_extruded(
    json_path: str,
    *,
    max_buildings: int = 200,
    min_height: float = 0.5,
    use_zmin: float = 0.0,
    prefer_roof_zmax: bool = False,
    skip_on_extrude_error: bool = True,
):
    """
    Build a trimesh.Scene by extruding each footprint polygon into a prism.

    This is a non-interactive builder (returns a trimesh.Scene) so that callers
    can overlay additional markers (e.g., UAV/BS positions, bounds wireframes)
    before calling scene.show().

    Notes:
    - Requires shapely (Polygon) for trimesh.creation.extrude_polygon
    - Some polygons may fail to extrude (invalid geometry). They can be skipped.
    """
    import trimesh
    try:
        from shapely.geometry import Polygon
    except Exception as e:
        raise RuntimeError("shapely is required for extrusion. Install shapely in your env.") from e

    bd = load_buildings_json(json_path)
    bs = bd.buildings

    scene = trimesh.Scene()
    added = 0
    skipped = 0

    for b in bs:
        if added >= max_buildings:
            break

        fp = b.get("footprint", None)
        if not fp or not isinstance(fp, list) or len(fp) < 3:
            skipped += 1
            continue

        poly_xy = np.asarray(fp, dtype=float)
        if poly_xy.ndim != 2 or poly_xy.shape[1] != 2 or (not np.isfinite(poly_xy).all()):
            skipped += 1
            continue

        zmin = float(use_zmin)
        if prefer_roof_zmax and ("roof_zmax" in b):
            zmax = float(b.get("roof_zmax", zmin))
        else:
            zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))

        height = max(0.0, zmax - zmin)
        if height < float(min_height):
            skipped += 1
            continue

        try:
            poly = Polygon(poly_xy)
            if (not poly.is_valid) or (poly.area <= 0):
                skipped += 1
                continue

            m = trimesh.creation.extrude_polygon(poly, height=height)
            m.apply_translation([0.0, 0.0, zmin])

            name = b.get("building_id", f"b{added:05d}")
            scene.add_geometry(m, node_name=name)
            added += 1
        except Exception:
            if skip_on_extrude_error:
                skipped += 1
                continue
            raise

    print(f"[build_buildings_scene_extruded] added={added}, skipped={skipped}, source={json_path}")
    return scene

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go


def _resolve_cfg_path(cfg: Dict[str, Any], p: str, fallback_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve relative paths using cfg['project_root'] (or fallback_root) as base."""
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp.resolve()
    root = Path(cfg.get("project_root", fallback_root or ".")).expanduser()
    return (root / pp).resolve()


def _scene_buildings_extent(buildings: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Compute global x/y/z extent from building AABBs."""
    xs, ys, zs = [], [], []
    for b in buildings:
        try:
            xs += [float(b["xmin"]), float(b["xmax"])]
            ys += [float(b["ymin"]), float(b["ymax"])]
            zmin = float(b.get("zmin", 0.0))
            zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))
            zs += [zmin, zmax]
        except Exception:
            pass
    return {
        "x": (min(xs), max(xs)) if xs else (np.nan, np.nan),
        "y": (min(ys), max(ys)) if ys else (np.nan, np.nan),
        "z": (min(zs), max(zs)) if zs else (np.nan, np.nan),
    }


def _wireframe_box_trace(
    bounds: Sequence[Sequence[float]],
    *,
    name: str,
    line_width: float = 2.0,
    opacity: float = 1.0,
    showlegend: bool = True,
) -> "go.Scatter3d":
    """Plotly wireframe for bounds = [[xmin,xmax],[ymin,ymax],[zmin,zmax]]."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    c = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=float)

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]

    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [c[i,0], c[j,0], None]
        ys += [c[i,1], c[j,1], None]
        zs += [c[i,2], c[j,2], None]

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        name=name,
        opacity=float(opacity),
        line=dict(width=float(line_width)),
        showlegend=bool(showlegend),
    )


def _buildings_wireframes_traces(
    buildings: Sequence[Dict[str, Any]],
    *,
    max_buildings: int = 800,
    opacity: float = 0.25,
    line_width: float = 1.0,
    legend_name: str = "buildings",
) -> list:
    """Create wireframe traces for building AABBs, deterministic first-N."""
    traces = []
    kept = 0
    for b in buildings[: int(max_buildings)]:
        try:
            xmin = float(b["xmin"]); xmax = float(b["xmax"])
            ymin = float(b["ymin"]); ymax = float(b["ymax"])
            zmin = float(b.get("zmin", 0.0))
            zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))
        except Exception:
            continue

        if xmax <= xmin or ymax <= ymin or zmax <= zmin:
            continue

        tr = _wireframe_box_trace(
            [[xmin, xmax], [ymin, ymax], [zmin, zmax]],
            name=legend_name if kept == 0 else legend_name,
            line_width=line_width,
            opacity=opacity,
            showlegend=(kept == 0),
        )
        traces.append(tr)
        kept += 1
    return traces

def _append_box_edges(xs, ys, zs, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Append 12 wireframe edges of an AABB box into (xs,ys,zs) lists.
    Uses None to break segments for Plotly Scatter3d.
    """
    c = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=float)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]

    for i, j in edges:
        xs += [c[i, 0], c[j, 0], None]
        ys += [c[i, 1], c[j, 1], None]
        zs += [c[i, 2], c[j, 2], None]


def _buildings_wireframe_single_trace(
    buildings: Sequence[Dict[str, Any]],
    *,
    max_buildings: int = 800,
    opacity: float = 0.20,
    line_width: float = 1.0,
    name: str = "buildings",
) -> "go.Scatter3d":
    """
    ONE-trace buildings wireframes (fast & stable in JupyterLab).
    """
    xs, ys, zs = [], [], []
    kept = 0

    for b in buildings[: int(max_buildings)]:
        try:
            xmin = float(b["xmin"]); xmax = float(b["xmax"])
            ymin = float(b["ymin"]); ymax = float(b["ymax"])
            zmin = float(b.get("zmin", 0.0))
            zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))
        except Exception:
            continue

        if xmax <= xmin or ymax <= ymin or zmax <= zmin:
            continue

        _append_box_edges(xs, ys, zs, xmin, xmax, ymin, ymax, zmin, zmax)
        kept += 1

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        name=f"{name} (n={kept})",
        opacity=float(opacity),
        line=dict(width=float(line_width)),
        showlegend=True,
    )


def visualize_scene_selftest_from_cfg(
    cfg_path: Union[str, Path],
    *,
    max_buildings: int = 800,
    buildings_opacity: float = 0.20,
    buildings_line_width: float = 1.0,
    uav_bounds_line_width: float = 6.0,
    bs_marker_size: float = 7.0,
    show_gu_region: bool = False,
    highlight_building_ids: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> "go.Figure":
    """
    Self-test 3D visualization:
    - buildings: AABB wireframes (fast & stable)
    - BS positions: markers
    - UAV bounds: thick wireframe (visualize altitude band)
    - optional GU xy_region: thin wireframe near ground
    """
    cfg_path = Path(cfg_path).expanduser().resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    buildings_json = _resolve_cfg_path(cfg, cfg["buildings"]["boxes_json"], fallback_root=cfg_path.parent)
    bd = load_buildings_json(str(buildings_json))
    buildings = bd.buildings

    uav_bounds = cfg.get("uav", {}).get("bounds", None)
    bs_positions = cfg.get("bs", {}).get("positions", []) or []
    gu_xy_region = cfg.get("gu", {}).get("xy_region", None)

    # traces = []
    # traces += _buildings_wireframes_traces(
    #     buildings,
    #     max_buildings=max_buildings,
    #     opacity=buildings_opacity,
    #     line_width=buildings_line_width,
    #     legend_name="buildings",
    # )
    highlight_set = set(highlight_building_ids or [])

    traces = []

    # buildings (single-trace wireframe recommended)
    traces.append(
        _buildings_wireframe_single_trace(
            buildings,
            max_buildings=max_buildings,
            opacity=buildings_opacity,
            line_width=buildings_line_width,
            name="buildings",
        )
    )

    # solid highlights (top-k)
    if highlight_set:
        for b in buildings:
            bid = b.get("building_id", None)
            if bid not in highlight_set:
                continue
            try:
                xmin = float(b["xmin"]); xmax = float(b["xmax"])
                ymin = float(b["ymin"]); ymax = float(b["ymax"])
                zmin = float(b.get("zmin", 0.0))
                zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))
            except Exception:
                continue
            if xmax <= xmin or ymax <= ymin or zmax <= zmin:
                continue
            traces.append(
                _aabb_to_box_mesh3d(
                    xmin, xmax, ymin, ymax, zmin, zmax,
                    name=f"highlight:{bid}",
                )
            )

    # UAV bounds (altitude band)
    if uav_bounds is not None:
        traces.append(
            _wireframe_box_trace(
                uav_bounds,
                name="uav_bounds",
                line_width=uav_bounds_line_width,
                opacity=1.0,
                showlegend=True,
            )
        )

    # BS markers
    if len(bs_positions) > 0:
        bs = np.asarray(bs_positions, dtype=float)
        traces.append(
            go.Scatter3d(
                x=bs[:, 0], y=bs[:, 1], z=bs[:, 2],
                mode="markers",
                name="bs",
                marker=dict(size=float(bs_marker_size)),
                showlegend=True,
            )
        )

    # Optional GU region wireframe near ground
    if show_gu_region and gu_xy_region is not None:
        (gxmin, gxmax), (gymin, gymax) = gu_xy_region
        traces.append(
            _wireframe_box_trace(
                [[float(gxmin), float(gxmax)], [float(gymin), float(gymax)], [0.0, 2.0]],
                name="gu_xy_region",
                line_width=2.0,
                opacity=1.0,
                showlegend=True,
            )
        )

    if title is None:
        title = f"Self-test 3D | {cfg_path.name}"

    # Set layout with aspectmode="data" so scale is correct
    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )
    return fig

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _building_height_from_record(b: Dict[str, Any]) -> float:
    """Height = zmax - zmin, using zmax/roof_zmax fallback."""
    zmin = float(b.get("zmin", 0.0))
    zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))
    return max(0.0, zmax - zmin)


def find_topk_tallest_buildings(
    buildings_json: Union[str, "Path"],
    *,
    k: int = 5,
    require_aabb: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load buildings JSON and return the top-k tallest building records.

    Returns a list of building dicts, each augmented with:
      - "_height": float
    """
    bd = load_buildings_json(str(buildings_json))
    buildings = bd.buildings

    valid = []
    for b in buildings:
        if not isinstance(b, dict):
            continue
        try:
            if require_aabb:
                _ = float(b["xmin"]); _ = float(b["xmax"])
                _ = float(b["ymin"]); _ = float(b["ymax"])
            h = _building_height_from_record(b)
            bb = dict(b)
            bb["_height"] = float(h)
            valid.append(bb)
        except Exception:
            continue

    valid.sort(key=lambda x: float(x.get("_height", 0.0)), reverse=True)
    return valid[: int(k)]


def print_topk_tallest_buildings(
    buildings_json: Union[str, "Path"],
    *,
    k: int = 5,
    header: str = "--- Top-K tallest buildings (by height=zmax-zmin) ---",
) -> Tuple[List[str], List[float]]:
    """
    Print top-k tallest building info and return (building_ids, heights).
    """
    topk = find_topk_tallest_buildings(buildings_json, k=k, require_aabb=True)

    print(header)
    if not topk:
        print("(no valid buildings found)")
        return [], []

    top_ids: List[str] = []
    top_heights: List[float] = []

    for rank, b in enumerate(topk, start=1):
        bid = b.get("building_id", f"rank_{rank}")
        xmin, xmax = float(b["xmin"]), float(b["xmax"])
        ymin, ymax = float(b["ymin"]), float(b["ymax"])
        zmin = float(b.get("zmin", 0.0))
        zmax = float(b.get("zmax", b.get("roof_zmax", zmin)))
        h = float(b.get("_height", _building_height_from_record(b)))
        center = b.get("center", [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])

        print(f"[{rank}] id={bid}")
        print(f"     height={h:.2f}, zmin={zmin:.2f}, zmax={zmax:.2f}")
        print(f"     aabb=([{xmin:.1f},{xmax:.1f}], [{ymin:.1f},{ymax:.1f}])")
        print(f"     center={center}")

        top_ids.append(str(bid))
        top_heights.append(float(h))

    return top_ids, top_heights

def _aabb_to_box_mesh3d(xmin, xmax, ymin, ymax, zmin, zmax, *, name="highlight"):
    """
    Create a solid AABB box as a Plotly Mesh3d trace (faces visible).
    """
    v = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=float)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 5, 1], [0, 4, 5],  # front
        [3, 2, 6], [3, 6, 7],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 6, 2], [1, 5, 6],  # right
    ], dtype=int)

    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        name=name,
        opacity=0.60,
        flatshading=True,
        showscale=False,
        showlegend=True,
    )
