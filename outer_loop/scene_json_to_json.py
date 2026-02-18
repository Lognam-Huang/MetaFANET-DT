import json
import numpy as np
from pathlib import Path

def _poly_area_xy(fp):
    xs = np.array([p[0] for p in fp], dtype=np.float64)
    ys = np.array([p[1] for p in fp], dtype=np.float64)
    if len(xs) < 3:
        return 0.0
    x2 = np.r_[xs, xs[0]]
    y2 = np.r_[ys, ys[0]]
    return 0.5 * abs(np.dot(x2[:-1], y2[1:]) - np.dot(y2[:-1], x2[1:]))


def generate_similar_buildings_json(
    proto_json_path: str,
    out_json_path: str,
    *,
    grid_n: int = 10,
    n_bins: int = 6,            # logspace bins count
    min_area_eps: float = 1e-3, # 过滤退化 footprint
    seed: int = 0,
    max_trials_per_building: int = 2000,
) -> None:
    """
    从原型 buildings.json 统计（并打印）场景特征，然后生成一个“统计相似”的新 buildings.json。
    统计与生成均只基于 buildings（不需要 scene bounds 文件）。

    生成对齐内容：
      - 总建筑数 N（过滤退化后）
      - 面积分桶（logspace）
      - grid_n × grid_n 网格内：每个 cell、每个 bin 的建筑数量配额（grid_bin_counts）

    注意：
      - 碰撞检测用 AABB（快速、保守），不是精确 polygon overlap。
      - 输出 json 的 building_id/source_ply 是占位符；后续你已有 json->ply+xml 生成器会写真实 mesh。
    """
    rng = np.random.default_rng(seed)
    proto_json_path = Path(proto_json_path)
    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(proto_json_path.read_text(encoding="utf-8"))
    buildings = cfg.get("buildings", [])
    assert len(buildings) > 0, "proto buildings 为空"

    # ---- extract valid buildings + stats ----
    areas = []
    centers = []
    bboxes = []        # (xmin,xmax,ymin,ymax)
    templates = []     # store (fp_rel, area, zmin, roof_zmax, zmax)

    xmins=[]; xmaxs=[]; ymins=[]; ymaxs=[]
    for b in buildings:
        fp = b.get("footprint")
        if (not fp) or len(fp) < 3:
            continue
        a = float(_poly_area_xy(fp))
        if a < min_area_eps:
            continue

        # center
        if "center" in b and b["center"] is not None:
            cx, cy = float(b["center"][0]), float(b["center"][1])
        else:
            pts = np.array(fp, dtype=np.float64)
            cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())

        # bbox
        xmin = float(b.get("xmin", min(p[0] for p in fp)))
        xmax = float(b.get("xmax", max(p[0] for p in fp)))
        ymin = float(b.get("ymin", min(p[1] for p in fp)))
        ymax = float(b.get("ymax", max(p[1] for p in fp)))

        # heights
        zmin = float(b.get("zmin", 0.0))
        roof_zmax = float(b.get("roof_zmax", b.get("zmax", 0.0)))
        zmax = float(b.get("zmax", roof_zmax))

        # normalize footprint around center
        fp_rel = [(float(p[0] - cx), float(p[1] - cy)) for p in fp]

        areas.append(a)
        centers.append((cx, cy))
        bboxes.append((xmin, xmax, ymin, ymax))
        templates.append((fp_rel, a, zmin, roof_zmax, zmax))

        xmins.append(xmin); xmaxs.append(xmax); ymins.append(ymin); ymaxs.append(ymax)

    areas = np.array(areas, dtype=np.float64)
    centers = np.array(centers, dtype=np.float64)

    assert len(templates) > 0, f"过滤后没有可用建筑（min_area_eps={min_area_eps} 可能太大）"

    # ---- scene bounds (from buildings bbox) ----
    xmin_s, xmax_s = float(min(xmins)), float(max(xmaxs))
    ymin_s, ymax_s = float(min(ymins)), float(max(ymaxs))
    W, H = (xmax_s - xmin_s), (ymax_s - ymin_s)
    scene_area = max(W * H, 1e-9)

    built_ratio = float(areas.sum() / scene_area)

    # ---- logspace bins ----
    a_min = max(float(areas.min()), min_area_eps)
    a_max = max(float(areas.max()), a_min * 1.001)
    edges_mid = np.logspace(np.log10(a_min), np.log10(a_max), num=n_bins+1)
    edges = np.r_[0.0, edges_mid[1:-1], np.inf]  # length = n_bins+1; intervals = n_bins
    hist_counts, _ = np.histogram(areas, bins=edges)
    hist_ratio = hist_counts / max(hist_counts.sum(), 1)

    # ---- assign bin id for each template ----
    # np.digitize with internal edges (excluding 0 and inf)
    inner_edges = edges[1:-1]
    bin_ids = np.digitize(areas, inner_edges, right=False)  # 0..n_bins-1

    # ---- grid occupancy + grid_bin_counts ----
    gx = np.clip(((centers[:,0] - xmin_s) / max(W, 1e-9) * grid_n).astype(int), 0, grid_n-1)
    gy = np.clip(((centers[:,1] - ymin_s) / max(H, 1e-9) * grid_n).astype(int), 0, grid_n-1)

    grid_counts = np.zeros((grid_n, grid_n), dtype=int)
    grid_bin_counts = np.zeros((grid_n, grid_n, n_bins), dtype=int)
    for ix, iy, bid in zip(gx, gy, bin_ids):
        grid_counts[iy, ix] += 1
        grid_bin_counts[iy, ix, bid] += 1

    # ---- print metrics (as you requested) ----
    print("Loaded:", str(proto_json_path))
    print("Top keys:", list(cfg.keys()))
    print("Num buildings total:", len(buildings))
    print("Valid buildings used:", len(templates))
    print("Area stats: min/median/max =", float(areas.min()), float(np.median(areas)), float(areas.max()))
    print("\nScene bounds (from buildings bbox):")
    print("xmin/xmax =", xmin_s, xmax_s)
    print("ymin/ymax =", ymin_s, ymax_s)
    print("W,H,Area  =", W, H, scene_area)
    print("Built-up ratio (sum footprint / bbox area):", built_ratio)

    print("\nArea bins (logspace, with 0..inf):")
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        print(f"  bin{i}: [{lo:.3g}, {hi:.3g})  count={hist_counts[i]}  ratio={hist_ratio[i]:.3f}")

    print("\nGrid occupancy summary:")
    print("grid_n:", grid_n)
    print("max buildings in one cell:", int(grid_counts.max()))
    non_empty = grid_counts[grid_counts > 0]
    print("mean buildings per non-empty cell:", float(non_empty.mean()) if non_empty.size else 0.0)
    print("non-empty cells:", int((grid_counts > 0).sum()), "/", grid_n * grid_n)

    densest = np.unravel_index(np.argmax(grid_counts), grid_counts.shape)
    print("\nDensest cell (y,x) =", densest, "count =", int(grid_counts[densest]))
    print("bin counts in densest cell:", grid_bin_counts[densest].tolist())

    # ---- build template pools per bin ----
    pools = [[] for _ in range(n_bins)]
    for t, bid in zip(templates, bin_ids):
        pools[int(bid)].append(t)

    # ---- helper: compute AABB from absolute footprint ----
    def fp_to_aabb(fp_abs):
        xs = [p[0] for p in fp_abs]
        ys = [p[1] for p in fp_abs]
        return (min(xs), max(xs), min(ys), max(ys))

    def aabb_overlap(a, b, margin=0.0):
        ax0, ax1, ay0, ay1 = a
        bx0, bx1, by0, by1 = b
        return not (ax1 + margin <= bx0 or bx1 + margin <= ax0 or ay1 + margin <= by0 or by1 + margin <= ay0)

    # ---- generation: match grid_bin_counts exactly (best-effort with retries) ----
    new_buildings = []
    placed_aabbs = []

    cell_w = W / grid_n
    cell_h = H / grid_n

    for iy in range(grid_n):
        for ix in range(grid_n):
            # cell bounds
            cx0 = xmin_s + ix * cell_w
            cx1 = cx0 + cell_w
            cy0 = ymin_s + iy * cell_h
            cy1 = cy0 + cell_h

            for bid in range(n_bins):
                need = int(grid_bin_counts[iy, ix, bid])
                if need <= 0:
                    continue
                if len(pools[bid]) == 0:
                    continue

                for _ in range(need):
                    ok = False
                    for _trial in range(max_trials_per_building):
                        fp_rel, a, zmin, roof_zmax, zmax = pools[bid][rng.integers(0, len(pools[bid]))]

                        # sample center inside this cell
                        cx = float(rng.uniform(cx0, cx1))
                        cy = float(rng.uniform(cy0, cy1))

                        fp_abs = [(cx + dx, cy + dy) for (dx, dy) in fp_rel]
                        bb = fp_to_aabb(fp_abs)

                        # keep within global bounds bbox (optional hard clamp)
                        if bb[0] < xmin_s or bb[1] > xmax_s or bb[2] < ymin_s or bb[3] > ymax_s:
                            continue

                        # AABB non-overlap
                        collision = False
                        for bb2 in placed_aabbs:
                            if aabb_overlap(bb, bb2, margin=0.0):
                                collision = True
                                break
                        if collision:
                            continue

                        # accept
                        placed_aabbs.append(bb)

                        # fill building record
                        xmin_b, xmax_b, ymin_b, ymax_b = bb
                        center3 = [cx, cy, (zmin + roof_zmax) * 0.5]
                        size3 = [xmax_b - xmin_b, ymax_b - ymin_b, roof_zmax - zmin]

                        new_buildings.append({
                            "xmin": xmin_b,
                            "xmax": xmax_b,
                            "ymin": ymin_b,
                            "ymax": ymax_b,
                            "zmin": zmin,
                            "zmax": zmax,
                            "center": center3,
                            "size": size3,
                            "building_id": f"gen_building_bin{bid}_{len(new_buildings):05d}",
                            "num_parts": 1,
                            "footprint": fp_abs,
                            "roof_zmax": roof_zmax,
                            "source_ply": "",  # 之后由 json->ply+xml 填充
                        })
                        ok = True
                        break

                    if not ok:
                        # best-effort：这个 cell/bin 的某栋放不进去就跳过（不会崩）
                        print(f"[WARN] placement failed at cell(y={iy},x={ix}) bin={bid}. Skipped 1 building.")
                        continue

    # ---- write output json ----
    out_cfg = {
        "meta": cfg.get("meta", {}),
        "buildings": new_buildings,
    }
    out_json_path.write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")
    print("\nWrote new buildings json:", str(out_json_path))
    print("New buildings count:", len(new_buildings))
