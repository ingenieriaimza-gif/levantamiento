#!/usr/bin/env python3
"""
Fusión RGB-D de paquetes grabados por la app 3D (formato "levanta-rgbd").

Lee una o varias partes (.zip o carpetas), fusiona todos los cuadros en un volumen TSDF
y exporta:
  nube_fusionada.e57   nube con color, metros, Z arriba  -> ReCap / Navisworks
  nube_fusionada.ply   igual, en PLY binario
  malla_color.ply      malla con color por vértice, metros, Z arriba
  malla_solidworks.stl malla en milímetros, Z arriba (reducida para SolidWorks)
  reporte.txt          resumen y parámetros

Uso:
  python3 -I procesar_rgbd.py parte1.zip parte2.zip --salida out/
  python3 -I procesar_rgbd.py paquete/ --voxel 0.004 --dmax 2.5 --conf 1
  python3 -I procesar_rgbd.py paquete.zip --escala 3.452 3.447   (real, medida en la nube)
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile

import cv2
import numpy as np
import open3d as o3d

# ARKit (x der, y arriba, z atrás)  ->  OpenCV/Open3D (x der, y abajo, z adelante)
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
# Mundo ARKit (Y arriba) -> mundo Z arriba (Navisworks / SolidWorks con Z arriba)
Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0],
                         [0, 0, -1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=float)


# Open3D (VoxelBlockGrid) toma el píxel de profundidad truncando la coordenada proyectada en vez de
# redondearla; sumar medio píxel al centro óptico equivale a usar el píxel más cercano.
PIXEL_OFFSET = float(os.environ.get("RGBD_PIXEL_OFFSET", "0.5"))


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- lectura

def extract_inputs(paths, workdir):
    """Devuelve una lista de carpetas de paquete (cada una con frames.jsonl)."""
    dirs = []
    for k, p in enumerate(paths):
        if os.path.isdir(p):
            dirs.extend(find_package_dirs(p))
        elif zipfile.is_zipfile(p):
            dest = os.path.join(workdir, f"in{k}")
            os.makedirs(dest)
            with zipfile.ZipFile(p) as z:
                for member in z.namelist():
                    target = os.path.realpath(os.path.join(dest, member))
                    if not target.startswith(os.path.realpath(dest) + os.sep):
                        raise ValueError(f"Ruta no permitida dentro del zip: {member}")
                z.extractall(dest)
            dirs.extend(find_package_dirs(dest))
        else:
            raise ValueError(f"No es carpeta ni zip: {p}")
    if not dirs:
        raise ValueError("No se encontró ningún frames.jsonl en las entradas.")
    return dirs


def find_package_dirs(root):
    out = []
    for cur, _sub, files in os.walk(root):
        if "frames.jsonl" in files and "__MACOSX" not in cur:
            out.append(cur)
    return sorted(out)


def load_frames(dirs):
    frames = []
    meta = None
    for d in dirs:
        mpath = os.path.join(d, "meta.json")
        if os.path.exists(mpath) and meta is None:
            with open(mpath) as f:
                meta = json.load(f)
        with open(os.path.join(d, "frames.jsonl")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fr = json.loads(line)
                name = f"{int(fr['i']):06d}"
                fr["jpg"] = os.path.join(d, "frames", name + ".jpg")
                fr["depth"] = os.path.join(d, "frames", name + ".depth")
                fr["conf"] = os.path.join(d, "frames", name + ".conf")
                if os.path.exists(fr["jpg"]) and os.path.exists(fr["depth"]):
                    frames.append(fr)
    # quitar duplicados (misma i en varias partes) y ordenar
    seen = {}
    for fr in frames:
        seen[int(fr["i"])] = fr
    frames = [seen[k] for k in sorted(seen)]
    if meta is None:
        raise ValueError("Falta meta.json en el paquete.")
    return meta, frames


def pose_matrix(fr):
    T = np.array(fr["T"], dtype=float).reshape(4, 4).T   # viene por columnas
    return T


def sharpness(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


# ---------------------------------------------------------------- fusión

def fuse(meta, frames, voxel, trunc, dmax, min_conf, drop_blur):
    dw, dh = int(meta["depthWidth"]), int(meta["depthHeight"])

    # Descartar fotos movidas (las más borrosas respecto al resto)
    used = frames
    dropped = 0
    if drop_blur and len(frames) > 10:
        sh = np.array([sharpness(fr["jpg"]) for fr in frames])
        thr = 0.45 * np.median(sh)
        used = [fr for fr, s in zip(frames, sh) if s >= thr]
        dropped = len(frames) - len(used)

    # Volumen TSDF en bloques dispersos (API tensorial de Open3D); crece según haga falta
    o3c = o3d.core
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel, block_resolution=8, block_count=50000,
        device=o3c.Device("CPU:0"))
    trunc_mult = trunc / voxel

    t0 = time.time()
    n_ok = 0
    for n, fr in enumerate(used):
        depth = np.fromfile(fr["depth"], dtype="<u2")
        if depth.size != dw * dh:
            continue
        depth = depth.reshape(dh, dw).astype(np.uint16)
        if os.path.exists(fr["conf"]) and min_conf > 0:
            conf = np.fromfile(fr["conf"], dtype=np.uint8)
            if conf.size == dw * dh:
                depth[conf.reshape(dh, dw) < min_conf] = 0

        color = cv2.imread(fr["jpg"], cv2.IMREAD_COLOR)
        if color is None:
            continue
        iw, ih = color.shape[1], color.shape[0]
        color = cv2.cvtColor(cv2.resize(color, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

        sx, sy = dw / float(fr.get("w", iw)), dh / float(fr.get("h", ih))
        # escalar el centro óptico respecto al centro del píxel
        K = np.array([[fr["fx"] * sx, 0, (fr["cx"] + 0.5) * sx - 0.5 + PIXEL_OFFSET],
                      [0, fr["fy"] * sy, (fr["cy"] + 0.5) * sy - 0.5 + PIXEL_OFFSET],
                      [0, 0, 1]], dtype=np.float64)
        cam_to_world = pose_matrix(fr) @ ARKIT_TO_CV
        Kt = o3c.Tensor(K, o3c.float64)
        Et = o3c.Tensor(np.linalg.inv(cam_to_world), o3c.float64)
        dimg = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth)))
        cimg = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(color)))
        coords = vbg.compute_unique_block_coordinates(dimg, Kt, Et, 1000.0, dmax,
                                                      trunc_voxel_multiplier=trunc_mult)
        vbg.integrate(coords, dimg, cimg, Kt, Et, 1000.0, dmax, trunc_voxel_multiplier=trunc_mult)
        n_ok += 1
        if (n + 1) % 50 == 0:
            log(f"  {n + 1}/{len(used)} cuadros fusionados ({time.time() - t0:.0f} s)")
    return vbg, n_ok, dropped


def extract(vbg, min_weight):
    """Malla y nube (legado) a partir del volumen. min_weight = lecturas mínimas por vóxel."""
    tmesh = vbg.extract_triangle_mesh(weight_threshold=min_weight)
    tpcd = vbg.extract_point_cloud(weight_threshold=min_weight)
    mesh = tmesh.to_legacy()
    pcd = tpcd.to_legacy()
    for g in (mesh, pcd):
        cols = np.asarray(g.vertex_colors if isinstance(g, o3d.geometry.TriangleMesh) else g.colors)
        if cols.size and cols.max() > 1.5:          # colores guardados en 0–255
            cols = cols / 255.0
            if isinstance(g, o3d.geometry.TriangleMesh):
                g.vertex_colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
            else:
                g.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
    return mesh, pcd


def clean_mesh(mesh, min_triangles):
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        return mesh
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    small = counts[labels] < min_triangles
    mesh.remove_triangles_by_mask(small)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------- exportación

def write_e57(path, pts, cols):
    import pye57
    e = pye57.E57(path, mode="w")
    data = {
        "cartesianX": pts[:, 0].astype(np.float64),
        "cartesianY": pts[:, 1].astype(np.float64),
        "cartesianZ": pts[:, 2].astype(np.float64),
        "colorRed": cols[:, 0].astype(np.uint8),
        "colorGreen": cols[:, 1].astype(np.uint8),
        "colorBlue": cols[:, 2].astype(np.uint8),
    }
    e.write_scan_raw(data, name="Levantamiento 3D")
    e.close()


def main():
    ap = argparse.ArgumentParser(description="Fusión RGB-D de la app 3D")
    ap.add_argument("entradas", nargs="+", help="partes .zip o carpetas del paquete")
    ap.add_argument("--salida", default="salida_rgbd")
    ap.add_argument("--voxel", type=float, default=0.004, help="tamaño de vóxel TSDF en m (def. 4 mm)")
    ap.add_argument("--trunc", type=float, default=None, help="truncamiento TSDF en m (def. 5×vóxel)")
    ap.add_argument("--dmax", type=float, default=3.0, help="profundidad máxima usada en m")
    ap.add_argument("--conf", type=int, default=2, choices=[0, 1, 2], help="confianza mínima 0/1/2 (def. 2, alta)")
    ap.add_argument("--lecturas", type=float, default=3.0,
                    help="lecturas mínimas por vóxel para conservarlo (def. 3)")
    ap.add_argument("--sin-filtro-borrosas", action="store_true")
    ap.add_argument("--escala", nargs=2, type=float, metavar=("REAL", "NUBE"),
                    help="calibración: distancia real y la misma distancia medida en la nube (m)")
    ap.add_argument("--max-triangulos-stl", type=int, default=1_500_000)
    args = ap.parse_args()

    trunc = args.trunc or 5 * args.voxel
    os.makedirs(args.salida, exist_ok=True)
    work = tempfile.mkdtemp(prefix="rgbd_")
    try:
        dirs = extract_inputs(args.entradas, work)
        meta, frames = load_frames(dirs)
        log(f"Paquete: {len(frames)} cuadros en {len(dirs)} parte(s), "
            f"profundidad {meta['depthWidth']}×{meta['depthHeight']}")

        vbg, used, dropped = fuse(meta, frames, args.voxel, trunc, args.dmax, args.conf,
                                  not args.sin_filtro_borrosas)
        log(f"Fusionados {used} cuadros ({dropped} descartados por borrosos)")

        mesh, pcd = extract(vbg, args.lecturas)
        mesh = clean_mesh(mesh, min_triangles=400)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.5)

        scale = 1.0
        if args.escala:
            real, measured = args.escala
            scale = real / measured
            log(f"Calibración de escala: ×{scale:.5f}")

        # A Z arriba + escala
        M = Y_UP_TO_Z_UP.copy()
        M[:3, :3] *= scale
        mesh.transform(M)
        pcd.transform(M)
        mesh.compute_vertex_normals()

        pts = np.asarray(pcd.points)
        cols = (np.clip(np.asarray(pcd.colors), 0, 1) * 255).round().astype(np.uint8)

        out = lambda n: os.path.join(args.salida, n)
        o3d.io.write_point_cloud(out("nube_fusionada.ply"), pcd, write_ascii=False)
        write_e57(out("nube_fusionada.e57"), pts, cols)
        o3d.io.write_triangle_mesh(out("malla_color.ply"), mesh, write_ascii=False)

        stl = o3d.geometry.TriangleMesh(mesh)
        if len(stl.triangles) > args.max_triangulos_stl:
            stl = stl.simplify_quadric_decimation(args.max_triangulos_stl)
        stl.scale(1000.0, center=(0, 0, 0))     # metros -> milímetros
        stl.compute_triangle_normals()
        stl.vertex_colors = o3d.utility.Vector3dVector()
        o3d.io.write_triangle_mesh(out("malla_solidworks.stl"), stl, write_ascii=False)

        lo, hi = pts.min(0), pts.max(0)
        rep = [
            "Levantamiento 3D — fusión RGB-D",
            f"Cuadros: {len(frames)} recibidos, {used} usados, {dropped} borrosos descartados",
            f"Vóxel TSDF: {args.voxel * 1000:.1f} mm · truncamiento {trunc * 1000:.0f} mm · "
            f"profundidad máx. {args.dmax} m · confianza mín. {args.conf} · lecturas mín. {args.lecturas:g}",
            f"Escala aplicada: ×{scale:.5f}",
            f"Nube: {len(pts):,} puntos",
            f"Malla: {len(mesh.triangles):,} triángulos (STL: {len(stl.triangles):,})",
            f"Extensión (X × Y × Z): {hi[0] - lo[0]:.3f} × {hi[1] - lo[1]:.3f} × {hi[2] - lo[2]:.3f} m",
            "Sistema: metros, Z hacia arriba (STL en milímetros). Origen: donde arrancó el escaneo.",
        ]
        with open(out("reporte.txt"), "w") as f:
            f.write("\n".join(rep) + "\n")
        log("\n".join(rep))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
