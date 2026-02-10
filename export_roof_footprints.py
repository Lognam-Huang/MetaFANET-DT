#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract building roof footprints (2D polygon) + height from a roof PLY file.

- Input:  one PLY file (ideally roof-only geometry)
- Output: a JSON file: list of buildings, each has:
    - xmin/xmax/ymin/ymax/zmin/zmax
    - center, size
    - footprint: polygon vertices in XY plane (convex hull)
    - roof_zmax: max z of the roof component
    - building_id, num_parts

Notes:
- This script assumes vertices in the PLY are already in world coordinates.
- It uses connected components splitting to approximate per-building roofs.
- Footprint uses convex hull (robust, low-risk). Concave hull can be added later.
"""

import argparse
import json
import math
from typing import List, Tuple

import numpy as np
import trimesh

import os
import glob

def extract_buildings_from_roof_dir(
    meshes_dir: str,
    *,
    zmin_override: float = 0.0,
    min_comp_verts: int = 50,
    min_hull_points: int = 10,
    pattern: str = "*-roof.ply",
) -> Tuple[List[dict], dict]:
    """
    Extract buildings from all roof PLY files in a directory.

    Returns:
      (buildings, meta)
    where:
      buildings: list of building records in the same schema as single-file mode
      meta: dict containing aggregated stats and per-file stats
    """
    roof_paths = sorted(glob.glob(os.path.join(meshes_dir, pattern)))

    meta_files = []
    all_buildings: List[dict] = []

    total_components = 0
    total_kept = 0
    total_dropped = 0

    for roof_path in roof_paths:
        geom = trimesh.load(roof_path, force="mesh")

        if isinstance(geom, trimesh.Scene):
            geom = trimesh.util.concatenate(tuple(geom.dump()))

        if not isinstance(geom, trimesh.Trimesh):
            # skip unsupported entries but record them
            meta_files.append({
                "source_ply": roof_path,
                "status": "skipped_non_mesh",
                "num_components_total": 0,
                "num_components_kept": 0,
                "num_components_dropped": 0,
            })
            continue

        components = mesh_to_components(geom)

        kept = 0
        dropped = 0

        # prefix ensures unique building_id across multiple roof files
        prefix = os.path.splitext(os.path.basename(roof_path))[0]

        for idx, comp in enumerate(components):
            vcount = len(comp.vertices)
            if vcount < min_comp_verts:
                dropped += 1
                continue

            building_id = f"{prefix}__element_{idx:05d}"
            rec = component_to_building_record(
                comp,
                building_id=building_id,
                zmin_override=zmin_override,
                min_points_for_hull=min_hull_points
            )
            # optional: keep traceability
            rec["source_ply"] = roof_path
            all_buildings.append(rec)
            kept += 1

        total_components += len(components)
        total_kept += kept
        total_dropped += dropped

        meta_files.append({
            "source_ply": roof_path,
            "status": "ok",
            "num_components_total": len(components),
            "num_components_kept": kept,
            "num_components_dropped": dropped,
        })

    meta = {
        "source_dir": meshes_dir,
        "pattern": pattern,
        "num_roof_files": len(roof_paths),
        "files": meta_files,
        "num_components_total": total_components,
        "num_components_kept": total_kept,
        "num_components_dropped": total_dropped,
        "assumption": "Each connected component in each roof mesh approximates one roof part; building volume extrudes to zmin.",
        "footprint_method": "convex_hull_2d (monotonic chain)",
    }

    return all_buildings, meta


def convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
    """
    Monotonic chain convex hull.
    points_xy: (N, 2) float array
    returns: (M, 2) hull points in CCW order, without repeating first at end
    """
    # Remove NaNs / inf
    pts = points_xy[np.isfinite(points_xy).all(axis=1)]
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)

    # Deduplicate
    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts

    # Sort by x then y
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def mesh_to_components(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    """
    Split mesh into connected components.
    If split fails or returns empty, treat the whole mesh as one component.
    """
    try:
        comps = mesh.split(only_watertight=False)
        if comps and len(comps) > 0:
            return list(comps)
    except Exception:
        pass
    return [mesh]


def component_to_building_record(
    comp: trimesh.Trimesh,
    building_id: str,
    zmin_override: float = 0.0,
    min_points_for_hull: int = 10
) -> dict:
    v = np.asarray(comp.vertices, dtype=float)
    if v.size == 0:
        # Empty geometry safeguard
        return {
            "xmin": 0.0, "xmax": 0.0,
            "ymin": 0.0, "ymax": 0.0,
            "zmin": float(zmin_override), "zmax": float(zmin_override),
            "center": [0.0, 0.0, float(zmin_override)],
            "size": [0.0, 0.0, 0.0],
            "building_id": building_id,
            "num_parts": 1,
            "footprint": [],
            "roof_zmax": float(zmin_override),
        }

    mins = v.min(axis=0)
    maxs = v.max(axis=0)

    xmin, ymin, zmin = mins.tolist()
    xmax, ymax, zmax = maxs.tolist()

    # Override zmin to 0 if you assume building extrudes to ground
    zmin = float(zmin_override)

    center = [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0]
    size = [xmax - xmin, ymax - ymin, zmax - zmin]

    # Footprint polygon via convex hull on XY
    pts_xy = v[:, :2]
    if len(pts_xy) < min_points_for_hull:
        hull_xy = np.unique(pts_xy, axis=0)
    else:
        hull_xy = convex_hull_2d(pts_xy)

    footprint = hull_xy.tolist()

    return {
        "xmin": float(xmin),
        "xmax": float(xmax),
        "ymin": float(ymin),
        "ymax": float(ymax),
        "zmin": float(zmin),
        "zmax": float(zmax),
        "center": [float(c) for c in center],
        "size": [float(s) for s in size],
        "building_id": building_id,
        "num_parts": 1,
        "footprint": footprint,
        "roof_zmax": float(zmax),
    }

def main():
    parser = argparse.ArgumentParser(
        description="Extract roof footprints (polygon) + height from roof PLY file(s) and write JSON."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ply", help="Path to input roof .ply file")
    group.add_argument("--dir", help="Directory that contains roof ply files (will match *-roof.ply)")

    parser.add_argument("--out", required=True, help="Path to output .json file")
    parser.add_argument(
        "--zmin",
        type=float,
        default=0.0,
        help="Override building zmin (default: 0.0). Use 0 if you assume building extrudes to ground."
    )
    parser.add_argument(
        "--min_comp_verts",
        type=int,
        default=50,
        help="Drop tiny components with fewer than this many vertices (default: 50)"
    )
    parser.add_argument(
        "--min_hull_points",
        type=int,
        default=10,
        help="If component has fewer XY points than this, hull will just use unique points (default: 10)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*-roof.ply",
        help="Filename pattern for roof ply files when using --dir (default: '*-roof.ply')"
    )

    args = parser.parse_args()

    if args.ply:
        # ---- single file mode (your original logic) ----
        geom = trimesh.load(args.ply, force="mesh")

        if isinstance(geom, trimesh.Scene):
            geom = trimesh.util.concatenate(tuple(geom.dump()))

        if not isinstance(geom, trimesh.Trimesh):
            raise RuntimeError(f"Unsupported geometry type loaded from {args.ply}: {type(geom)}")

        components = mesh_to_components(geom)

        buildings = []
        kept = 0
        dropped = 0

        for idx, comp in enumerate(components):
            vcount = len(comp.vertices)
            if vcount < args.min_comp_verts:
                dropped += 1
                continue

            building_id = f"element_{idx:05d}"
            rec = component_to_building_record(
                comp,
                building_id=building_id,
                zmin_override=args.zmin,
                min_points_for_hull=args.min_hull_points
            )
            buildings.append(rec)
            kept += 1

        meta = {
            "source_ply": args.ply,
            "num_components_total": len(components),
            "num_components_kept": kept,
            "num_components_dropped": dropped,
            "assumption": "Each connected component in roof mesh approximates one building roof; building volume extrudes to zmin.",
            "footprint_method": "convex_hull_2d (monotonic chain)",
        }

        output = {"meta": meta, "buildings": buildings}

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[ok] wrote {args.out}")
        print(f"     total components={len(components)}, kept={kept}, dropped={dropped}")

    else:
        # ---- directory mode (new) ----
        buildings, meta = extract_buildings_from_roof_dir(
            args.dir,
            zmin_override=args.zmin,
            min_comp_verts=args.min_comp_verts,
            min_hull_points=args.min_hull_points,
            pattern=args.pattern
        )

        output = {"meta": meta, "buildings": buildings}

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[ok] wrote {args.out}")
        print(f"     roof files={meta['num_roof_files']}, total components={meta['num_components_total']}, "
              f"kept={meta['num_components_kept']}, dropped={meta['num_components_dropped']}")


if __name__ == "__main__":
    main()
