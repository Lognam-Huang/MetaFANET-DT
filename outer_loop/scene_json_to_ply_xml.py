import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]

# ----------------------------
# Mesh containers
# ----------------------------

@dataclass
class Mesh:
    verts: List[Vec3]                 # positions
    faces: List[Tuple[int, int, int]] # triangles

# ----------------------------
# Geometry utils
# ----------------------------

def _close_polygon_xy(fp: List[Vec2]) -> List[Vec2]:
    if len(fp) >= 3 and fp[0] != fp[-1]:
        return fp + [fp[0]]
    return fp

def _triangulate_fan(n: int) -> List[Tuple[int, int, int]]:
    # fan: (0, i, i+1)
    return [(0, i, i + 1) for i in range(1, n - 1)]

def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def _vcross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def _vnorm(a: Vec3) -> float:
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def _vnormalize(a: Vec3) -> Vec3:
    n = _vnorm(a)
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (a[0]/n, a[1]/n, a[2]/n)

def _compute_vertex_normals(mesh: Mesh) -> List[Vec3]:
    acc = [(0.0, 0.0, 0.0) for _ in mesh.verts]
    for (i, j, k) in mesh.faces:
        p0 = mesh.verts[i]
        p1 = mesh.verts[j]
        p2 = mesh.verts[k]
        n = _vcross(_vsub(p1, p0), _vsub(p2, p0))
        # area-weighted
        acc[i] = _vadd(acc[i], n)
        acc[j] = _vadd(acc[j], n)
        acc[k] = _vadd(acc[k], n)
    return [_vnormalize(v) for v in acc]

def _compute_uv_from_xy(verts: List[Vec3]) -> List[Vec2]:
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = max(xmax - xmin, 1e-9)
    dy = max(ymax - ymin, 1e-9)
    # simple bbox-normalized UV
    return [((v[0]-xmin)/dx, (v[1]-ymin)/dy) for v in verts]

# ----------------------------
# Build roof / wall meshes from one footprint
# ----------------------------

def build_roof_mesh_from_footprint(fp_xy: List[Vec2], z: float) -> Mesh:
    fp_core = fp_xy[:]  # not closed
    n = len(fp_core)
    verts = [(x, y, z) for (x, y) in fp_core]
    faces = _triangulate_fan(n)
    return Mesh(verts=verts, faces=faces)

def build_wall_mesh_from_footprint(fp_xy: List[Vec2], z0: float, z1: float) -> Mesh:
    # Create bottom and top rings
    n = len(fp_xy)
    bottom = [(x, y, z0) for (x, y) in fp_xy]
    top    = [(x, y, z1) for (x, y) in fp_xy]
    verts = bottom + top
    faces: List[Tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        bi, bj = i, j
        ti, tj = n + i, n + j
        # quad (bi, bj, tj, ti) -> two triangles
        faces.append((bi, bj, tj))
        faces.append((bi, tj, ti))
    return Mesh(verts=verts, faces=faces)

# ----------------------------
# Binary PLY writer (matches your header layout)
# ----------------------------

def write_binary_ply_with_layout(
    out_path: Path,
    mesh: Mesh,
    *,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """
    Layout per-vertex (float32 each):
      x y z nx ny nz s t Col_0 Col_1 Col_2
    Faces:
      uint8 n (should be 3) + int32 * n indices
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normals = _compute_vertex_normals(mesh)
    uvs = _compute_uv_from_xy(mesh.verts)

    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(mesh.verts)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float s",
        "property float t",
        "property float Col_0",
        "property float Col_1",
        "property float Col_2",
        f"element face {len(mesh.faces)}",
        "property list uchar int vertex_indices",
        "end_header",
        ""
    ]).encode("utf-8")

    with open(out_path, "wb") as f:
        f.write(header)

        # vertices
        for (p, n, uv) in zip(mesh.verts, normals, uvs):
            x, y, z = p
            nx, ny, nz = n
            s, t = uv
            c0, c1, c2 = color
            f.write(struct.pack(
                "<11f",
                float(x), float(y), float(z),
                float(nx), float(ny), float(nz),
                float(s), float(t),
                float(c0), float(c1), float(c2),
            ))

        # faces
        for (i, j, k) in mesh.faces:
            f.write(struct.pack("<B", 3))         # uchar count
            f.write(struct.pack("<3i", i, j, k))  # int32 indices

# ----------------------------
# Scene JSON -> combined roof/wall meshes
# ----------------------------

def scene_json_to_roof_wall_meshes(cfg: Dict[str, Any]) -> Tuple[Mesh, Mesh]:
    buildings = cfg.get("buildings", [])
    roof_verts: List[Vec3] = []
    roof_faces: List[Tuple[int, int, int]] = []
    wall_verts: List[Vec3] = []
    wall_faces: List[Tuple[int, int, int]] = []

    for b in buildings:
        fp = b.get("footprint")
        if not fp or len(fp) < 3:
            continue

        fp_xy = [(float(p[0]), float(p[1])) for p in fp]
        # footprint in your json is not closed, that's fine
        fp_xy = fp_xy[:]  # keep open
        if len(fp_xy) >= 3 and fp_xy[0] == fp_xy[-1]:
            fp_xy = fp_xy[:-1]

        z0 = float(b.get("zmin", 0.0))
        # prefer roof_zmax if exists, fallback to zmax
        z1 = float(b.get("roof_zmax", b.get("zmax", 0.0)))

        if z1 <= z0:
            # skip degenerate building
            continue

        # roof
        roof_mesh = build_roof_mesh_from_footprint(fp_xy, z=z1)
        roof_off = len(roof_verts)
        roof_verts.extend(roof_mesh.verts)
        roof_faces.extend([(a+roof_off, c+roof_off, d+roof_off) for (a, c, d) in roof_mesh.faces])

        # walls
        wall_mesh = build_wall_mesh_from_footprint(fp_xy, z0=z0, z1=z1)
        wall_off = len(wall_verts)
        wall_verts.extend(wall_mesh.verts)
        wall_faces.extend([(a+wall_off, c+wall_off, d+wall_off) for (a, c, d) in wall_mesh.faces])

    return Mesh(roof_verts, roof_faces), Mesh(wall_verts, wall_faces)

# ----------------------------
# XML writer
# ----------------------------

def write_final_scene_xml(
    out_path: Path,
    *,
    wall_ply_rel: str,
    roof_ply_rel: str,
    concrete_reflectance: str = "0.800000 0.800000 0.800000",
    max_depth: int = 12,
) -> None:
    """
    Writes Mitsuba-style scene xml matching your example.
    """
    xml = f"""<?xml version="1.0" ?>
<scene version="2.1.0">
  <integrator type="path" id="elm__0" name="elm__0">
    <integer name="max_depth" value="{max_depth}"/>
  </integrator>

  <bsdf type="twosided" id="mat-itu_concrete" name="mat-itu_concrete">
    <bsdf type="diffuse" name="bsdf">
      <rgb name="reflectance" value="{concrete_reflectance}"/>
    </bsdf>
  </bsdf>

  <bsdf type="twosided" id="mat-itu_wet_ground" name="mat-itu_wet_ground">
    <bsdf type="diffuse" name="bsdf">
      <rgb name="reflectance" value="{concrete_reflectance}"/>
    </bsdf>
  </bsdf>

  <shape type="ply" id="elm__6" name="elm__6">
    <string name="filename" value="{wall_ply_rel}"/>
    <ref id="mat-itu_concrete" name="bsdf"/>
  </shape>

  <shape type="ply" id="elm__7" name="elm__7">
    <string name="filename" value="{roof_ply_rel}"/>
    <ref id="mat-itu_concrete" name="bsdf"/>
  </shape>
</scene>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")

# ----------------------------
# Main entry
# ----------------------------

def generate_scene_meshes_and_xml(
    scene_json_path: str,
    out_scene_dir: str,
    *,
    scene_prefix: str = "None_buildings",
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, str]:
    """
    Output:
      out_scene_dir/
        final-scene.xml
        meshes/
          <scene_prefix>-roof.ply
          <scene_prefix>-wall.ply
    """
    scene_json_path = str(scene_json_path)
    out_scene_dir = Path(out_scene_dir)
    meshes_dir = out_scene_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(scene_json_path).read_text(encoding="utf-8"))

    roof_mesh, wall_mesh = scene_json_to_roof_wall_meshes(cfg)

    roof_name = f"{scene_prefix}-roof.ply"
    wall_name = f"{scene_prefix}-wall.ply"

    roof_ply_path = meshes_dir / roof_name
    wall_ply_path = meshes_dir / wall_name

    # Write PLYs
    write_binary_ply_with_layout(roof_ply_path, roof_mesh, color=color)
    write_binary_ply_with_layout(wall_ply_path, wall_mesh, color=color)

    # Write XML (relative paths like your example)
    xml_path = out_scene_dir / "final-scene.xml"
    write_final_scene_xml(
        xml_path,
        wall_ply_rel=f"meshes/{wall_name}",
        roof_ply_rel=f"meshes/{roof_name}",
    )

    return {
        "xml": str(xml_path),
        "roof_ply": str(roof_ply_path),
        "wall_ply": str(wall_ply_path),
        "meshes_dir": str(meshes_dir),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_json", required=True, help="Path to scene json")
    ap.add_argument("--out_dir", required=True, help="Output scene directory")
    ap.add_argument("--prefix", default="None_buildings", help="Prefix for generated ply files")
    args = ap.parse_args()

    out = generate_scene_meshes_and_xml(
        scene_json_path=args.scene_json,
        out_scene_dir=args.out_dir,
        scene_prefix=args.prefix,
    )
    print(json.dumps(out, indent=2))
