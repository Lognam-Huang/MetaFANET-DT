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

