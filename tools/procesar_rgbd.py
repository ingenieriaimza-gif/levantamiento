#!/usr/bin/env python3
"""
Fusión RGB-D de paquetes grabados por la app 3D (formato "levanta-rgbd").

Lee una o varias partes (.zip o carpetas), fusiona todos los cuadros en un volumen TSDF
y exporta:
  nube_fusionada.e57   nube con color, metros, Z arriba  -> ReCap
  nube_fusionada.ply   igual, en PLY binario
  malla_color.ply      malla con color por vértice, metros, Z arriba
  malla_solidworks.stl malla en milímetros, Z arriba (reducida para SolidWorks)
  modelo_texturizado.glb  malla con las fotos (aspecto fotográfico), metros, Y arriba
  modelo_texturizado.usdz igual, para abrir directo en el iPhone (Vista rápida y AR)
  obj/                 la misma malla en OBJ + MTL + JPG, metros, Z arriba
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # módulos propios (texturizar)

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


def progreso(frac, etapa):
    """Línea de avance para el programa de la PC (barra, porcentaje y tiempo restante)."""
    print(f"@@PROGRESO {max(0.0, min(1.0, frac)):.4f}|{etapa}", flush=True)


# avance total de este script (el programa de la PC deja el 4 % final para el Gaussian Splatting)
# (pesos medidos en una corrida real: la malla y el modelo con fotos son lo más tardado)
ETAPAS = {"carga": (0.00, 0.01), "nitidez": (0.01, 0.03), "fusion": (0.03, 0.10), "malla": (0.10, 0.18),
          "detalle": (0.18, 0.22), "completo": (0.22, 0.32), "fotos": (0.32, 0.90), "exportar": (0.90, 1.00)}


def avance(etapa, f, texto):
    a, b = ETAPAS[etapa]
    progreso(a + (b - a) * f, texto)


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

RASANTE_TAN = float(os.environ.get("RGBD_RASANTE_TAN", "2.5"))   # ≈ 68°


def quitar_rasantes(depth_mm, fx, max_tan=2.5):
    """Quita puntos vistos casi de refilón (pendiente de la superficie respecto a la vista > max_tan).
    En profundidad de alta resolución esos puntos engordan los contornos de tubos y bordes."""
    z = depth_mm.astype(np.float32) / 1000.0
    valid = z > 0
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    slope = np.hypot(gx, gy) * fx / np.maximum(z, 1e-3)
    nb = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0     # gradiente fiable
    out = depth_mm.copy()
    out[nb & (slope > max_tan)] = 0
    return out


def fuse(meta, frames, voxel, trunc, dmax, min_conf, drop_blur, ia=None, ia_width=768, etapa="fusion",
         max_tan=None, zona=None, block_count=50000):
    """ia: RefinadorIA opcional; si se da, cada cuadro se refina a ia_width de ancho antes de fusionarse."""
    dw, dh = int(meta["depthWidth"]), int(meta["depthHeight"])
    ia_aporte = []

    # Descartar fotos movidas (las más borrosas respecto al resto)
    used = frames
    dropped = 0
    if drop_blur and len(frames) > 10:
        sh = []
        for i, fr in enumerate(frames):
            sh.append(sharpness(fr["jpg"]))
            if i % 20 == 0 and etapa == "fusion":
                avance("nitidez", i / len(frames), "Revisando fotos movidas")
        sh = np.array(sh)
        thr = 0.45 * np.median(sh)
        used = [fr for fr, s in zip(frames, sh) if s >= thr]
        dropped = len(frames) - len(used)

    # Volumen TSDF en bloques dispersos (API tensorial de Open3D); crece según haga falta
    o3c = o3d.core
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel, block_resolution=8, block_count=block_count,
        device=o3c.Device("CPU:0"))
    trunc_mult = trunc / voxel

    t0 = time.time()
    n_ok = 0
    for n, fr in enumerate(used):
        depth = np.fromfile(fr["depth"], dtype="<u2")
        if depth.size != dw * dh:
            continue
        depth = depth.reshape(dh, dw).astype(np.uint16)
        conf = None
        if os.path.exists(fr["conf"]):
            c = np.fromfile(fr["conf"], dtype=np.uint8)
            if c.size == dw * dh:
                conf = c.reshape(dh, dw)
                if min_conf > 0:
                    depth[conf < min_conf] = 0

        color_full = cv2.imread(fr["jpg"], cv2.IMREAD_COLOR)
        if color_full is None:
            continue
        iw, ih = color_full.shape[1], color_full.shape[0]
        ow, oh = dw, dh
        if ia is not None:
            # resolución de trabajo con IA: misma proporción que la foto
            ow = int(ia_width)
            oh = int(round(ow * ih / iw))
            rgb_full = cv2.cvtColor(color_full, cv2.COLOR_BGR2RGB)
            from ia_profundidad import refinar_cuadro
            depth, frac = refinar_cuadro(ia, rgb_full, depth, conf, (ow, oh))
            ia_aporte.append(frac)
            depth = quitar_rasantes(depth, fr["fx"] * ow / float(fr.get("w", iw)), max_tan=max_tan or RASANTE_TAN)
        color = cv2.cvtColor(cv2.resize(color_full, (ow, oh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if zona is not None:
            # solo lo que cae dentro de la caja de la pieza (+2 cm): el resto no se integra
            depth = recortar_a_zona(depth, fr, zona, ow, oh, iw, ih)
            if not depth.any():
                continue

        sx, sy = ow / float(fr.get("w", iw)), oh / float(fr.get("h", ih))
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
        if n % 5 == 0 and etapa in ("fusion", "completo"):
            avance(etapa, 0.85 * (n + 1) / len(used),
                   f"Uniendo cuadros {n + 1} de {len(used)}" if etapa == "fusion"
                   else f"Completando zonas difíciles ({n + 1} de {len(used)})")
        if (n + 1) % 50 == 0:
            log(f"  {n + 1}/{len(used)} cuadros fusionados ({time.time() - t0:.0f} s)")
    if ia_aporte:
        log(f"  IA: aportó detalle en {np.mean(ia_aporte) * 100:.0f} % de los píxeles (el resto, LiDAR)")
    return vbg, n_ok, dropped


def recortar_a_zona(depth_mm, fr, zona, ow, oh, iw, ih, margen=0.02):
    """Pone en 0 la profundidad de los píxeles cuyo punto 3D queda fuera de la caja de la pieza."""
    sx, sy = ow / float(fr.get("w", iw)), oh / float(fr.get("h", ih))
    fx, fy = fr["fx"] * sx, fr["fy"] * sy
    cx, cy = (fr["cx"] + 0.5) * sx - 0.5, (fr["cy"] + 0.5) * sy - 0.5
    v, u = np.nonzero(depth_mm)
    if len(u) == 0:
        return depth_mm
    d = depth_mm[v, u].astype(np.float64) / 1000.0
    pc = np.stack([(u - cx) / fx * d, -(v - cy) / fy * d, -d, np.ones_like(d)], 1)
    pw = pc @ pose_matrix(fr).T
    c, h = np.asarray(zona["centro"], float), np.asarray(zona["medio"], float) + margen
    out = (np.abs(pw[:, :3] - c) > h).any(1)
    depth_mm = depth_mm.copy()
    depth_mm[v[out], u[out]] = 0
    return depth_mm


def cargar_zonas(dirs):
    """Piezas enfocadas en el iPhone (zonas.json en el paquete)."""
    for d in dirs:
        p = os.path.join(d, "zonas.json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    z = [x for x in json.load(f).get("zonas", []) if "centro" in x and "medio" in x]
                if z:
                    return z
            except (OSError, ValueError):
                pass
    return []


def en_zonas(P, zonas, encoger=0.0):
    """Máscara de puntos (N×3, mundo de ARKit) dentro de alguna caja (encogida `encoger` m)."""
    m = np.zeros(len(P), bool)
    for z in zonas:
        c, h = np.asarray(z["centro"], float), np.asarray(z["medio"], float) - encoger
        m |= (np.abs(P - c) <= h).all(1)
    return m


def zonas_a(M, zonas, scale=1.0):
    """Cajas en otro sistema (Z arriba, escala): centro transformado; medios con los ejes cambiados."""
    out = []
    for z in zonas:
        c = (M @ np.r_[np.asarray(z["centro"], float), 1.0])[:3]
        hx, hy, hz = (np.asarray(z["medio"], float) * scale).tolist()
        out.append(dict(z, centro=c.tolist(), medio=[hx, hz, hy]))
    return out


def piezas_en_detalle(meta, frames, zonas, base, lecturas, voxel=0.002):
    """Reconstruye cada pieza enfocada a 2 mm con las tomas cercanas y la une a la malla general:
    dentro de la caja se cambia la malla de 4 mm por la de 2 mm."""
    trozos = []
    for k, z in enumerate(zonas):
        c, h = np.asarray(z["centro"], float), np.asarray(z["medio"], float)
        cerca = []
        for fr in frames:
            p = pose_matrix(fr)[:3, 3]
            if np.linalg.norm(np.maximum(np.abs(p - c) - h, 0)) <= 1.2:
                cerca.append(fr)
        nombre = z.get("nombre", f"Pieza {k + 1}")
        if len(cerca) < 3:
            log(f"  {nombre}: muy pocas tomas de cerca ({len(cerca)}); se queda con el detalle normal")
            continue
        avance("detalle", k / len(zonas), f"Detalle de {nombre} ({len(cerca)} tomas de cerca)")
        vbg, n_ok, _ = fuse(meta, cerca, voxel, 5.0 * voxel, 1.5, 2, True, etapa="detalle", zona=z,
                            block_count=40000)
        fino, _ = extract(vbg, lecturas)
        del vbg
        F = np.asarray(fino.triangles)
        if len(F) == 0:
            continue
        cen = np.asarray(fino.vertices)[F].mean(1)
        fino.remove_triangles_by_mask(~en_zonas(cen, [z]))
        fino.remove_unreferenced_vertices()
        fino = clean_mesh(fino, min_triangles=50) if len(fino.triangles) else fino
        if len(fino.triangles) == 0:
            continue
        fino = fino.filter_smooth_taubin(number_of_iterations=5)
        trozos.append(fino)
        log(f"  {nombre}: {len(fino.triangles):,} triángulos a 2 mm con {n_ok} tomas de cerca")
    if not trozos:
        return base, 0
    # quitar de la malla general lo que queda dentro de las cajas con detalle (encogidas 4 mm: traslape mínimo)
    F = np.asarray(base.triangles)
    cen = np.asarray(base.vertices)[F].mean(1)
    out = o3d.geometry.TriangleMesh(base)
    out.remove_triangles_by_mask(en_zonas(cen, zonas, encoger=0.004))
    out.remove_unreferenced_vertices()
    n_fino = 0
    for t in trozos:
        out += t
        n_fino += len(t.triangles)
    out.compute_vertex_normals()
    return out, n_fino


def separar_zonas(mesh, zonas):
    """(malla fuera de las piezas, malla dentro) según el centro de cada triángulo."""
    F = np.asarray(mesh.triangles)
    if not zonas or len(F) == 0:
        return mesh, None
    m = en_zonas(np.asarray(mesh.vertices)[F].mean(1), zonas)
    if not m.any():
        return mesh, None
    fuera, dentro = o3d.geometry.TriangleMesh(mesh), o3d.geometry.TriangleMesh(mesh)
    fuera.remove_triangles_by_mask(m)
    dentro.remove_triangles_by_mask(~m)
    for g in (fuera, dentro):
        g.remove_unreferenced_vertices()
    return fuera, dentro


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


def acabado(mesh):
    """Superficies más reales: suavizado Taubin (quita la ondulación del LiDAR sin encoger).
    En la escena de prueba no cambia la mediana (1.2 mm) y baja el error del 95 % de 6.3 a 5.9 mm.
    (El relleno de huecos de Open3D se probó, pero en Windows cierra el programa: no se usa.)"""
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.compute_vertex_normals()
    return mesh


def solo_lo_nuevo(completa, exacta, cerca=0.010, min_triangles=60):
    """De la malla completa (confianza media) deja solo lo que la malla exacta no tiene: los triángulos
    con algún vértice a más de `cerca` de la exacta. Así, donde hay datos de alta confianza el modelo
    con fotos usa la geometría exacta (para medir), y lo demás solo rellena."""
    if len(exacta.triangles) == 0:
        return completa
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(exacta))
    d = sc.compute_distance(o3d.core.Tensor(np.asarray(completa.vertices).astype(np.float32))).numpy()
    F = np.asarray(completa.triangles)
    keep = (d[F] > cerca).any(1)
    out = o3d.geometry.TriangleMesh(completa)
    out.remove_triangles_by_mask(~keep)
    out.remove_unreferenced_vertices()
    return clean_mesh(out, min_triangles=min_triangles) if len(out.triangles) else out


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
    ap.add_argument("--max-triangulos-stl", type=int, default=800_000)
    ap.add_argument("--sin-textura", action="store_true", help="no generar el modelo con fotos")
    ap.add_argument("--sin-completo", action="store_true",
                    help="modelo con fotos sobre la malla exacta (sin completar zonas de confianza media)")
    ap.add_argument("--ia", action="store_true", help="refinar la profundidad con IA (Prompt Depth Anything)")
    ap.add_argument("--ia-ancho", type=int, default=768, help="ancho de la profundidad refinada por IA")
    ap.add_argument("--triangulos-textura", type=int, default=700_000)
    ap.add_argument("--pagina-textura", type=int, default=4096, help="tamaño de cada página de textura")
    ap.add_argument("--escala-textura", type=float, default=1.0,
                    help="reduce las fotos en la textura (p. ej. 0.5 para un modelo ligero para el iPhone)")
    args = ap.parse_args()

    trunc = args.trunc or 5 * args.voxel
    os.makedirs(args.salida, exist_ok=True)
    work = tempfile.mkdtemp(prefix="rgbd_")
    try:
        avance("carga", 0, "Leyendo lo que mandó el iPhone")
        dirs = extract_inputs(args.entradas, work)
        meta, frames = load_frames(dirs)
        log(f"Paquete: {len(frames)} cuadros en {len(dirs)} parte(s), "
            f"profundidad {meta['depthWidth']}×{meta['depthHeight']}")

        ia = None
        if args.ia:
            from ia_profundidad import RefinadorIA, disponible
            if not disponible():
                log("La IA no está instalada en esta PC: se procesa solo con el LiDAR.")
            else:
                ia = RefinadorIA()
                log(f"IA de profundidad activa ({ia.device})")
        vbg, used, dropped = fuse(meta, frames, args.voxel, trunc, args.dmax, args.conf,
                                  not args.sin_filtro_borrosas, ia=ia, ia_width=args.ia_ancho)
        del ia
        log(f"Fusionados {used} cuadros ({dropped} descartados por borrosos)")

        avance("malla", 0, "Formando las superficies")
        mesh, pcd = extract(vbg, args.lecturas)
        del vbg                                   # liberar el volumen antes de texturizar
        import gc
        gc.collect()
        scale = 1.0
        if args.escala:
            real, measured = args.escala
            scale = real / measured
            log(f"Calibración de escala: ×{scale:.5f}")
        mesh = clean_mesh(mesh, min_triangles=400)
        mesh = acabado(mesh)
        zonas = cargar_zonas(dirs)
        n_detalle = 0
        if zonas:
            log(f"Piezas enfocadas en el iPhone: {len(zonas)}")
            mesh, n_detalle = piezas_en_detalle(meta, frames, zonas, mesh, args.lecturas)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.5)


        tex_info = None
        if not args.sin_textura:
            import texturizar
            from texturizar import texture_mesh
            texturizar.AVANCE = lambda f, t: avance("fotos", f, t)
            kf = [f for f in frames if f.get("kf", True)] or frames     # fotos clave (las más nítidas)
            # Modelo con fotos "completo": se vuelve a unir todo aceptando también lecturas de confianza
            # media, ángulos más inclinados y hasta 5 m. Cubre piso, objetos oscuros y lo lejano que la
            # malla exacta descarta (en un escaneo real: 67 % -> 89 % de lo que se ve en las fotos).
            # Donde la malla exacta tiene datos se conserva tal cual; lo nuevo solo rellena lo que falta.
            tex_mesh, tex_dmax = mesh, args.dmax
            if not args.sin_completo:
                try:
                    v2 = max(args.voxel, 0.006)
                    vbg2, _, _ = fuse(meta, frames, v2, 5.0 * v2, max(args.dmax, 5.0), 1,
                                      not args.sin_filtro_borrosas, etapa="completo", max_tan=5.0)
                    m2, _ = extract(vbg2, args.lecturas)
                    del vbg2
                    gc.collect()
                    avance("completo", 0.9, "Limpiando el modelo completo")
                    m2 = acabado(clean_mesh(m2, min_triangles=400))
                    extra = solo_lo_nuevo(m2, mesh)
                    del m2
                    if len(extra.triangles) > 0:
                        tex_mesh = o3d.geometry.TriangleMesh(mesh)
                        tex_mesh += extra
                        tex_mesh.compute_vertex_normals()
                        tex_dmax = max(args.dmax, 5.0)
                        log(f"Modelo completo para las fotos: malla exacta + {len(extra.triangles):,} "
                            f"triángulos de zonas difíciles")
                except Exception as e:      # noqa: BLE001 — si no alcanza la memoria, se usa la malla exacta
                    log(f"  (modelo completo omitido: {e})")
            avance("fotos", 0, "Preparando las fotos")
            target_tex = args.triangulos_textura
            if n_detalle:
                # el presupuesto de triángulos se reparte: las piezas conservan su detalle (hasta la mitad)
                fuera, dentro = separar_zonas(tex_mesh, zonas)
                if dentro is not None:
                    lim_d = target_tex // 2
                    if len(dentro.triangles) > lim_d:
                        dentro = dentro.simplify_quadric_decimation(lim_d)
                    lim_f = max(200_000, target_tex - len(dentro.triangles))
                    if len(fuera.triangles) > lim_f:
                        fuera = fuera.simplify_quadric_decimation(lim_f)
                    tex_mesh = fuera + dentro
                    tex_mesh.compute_vertex_normals()
                    target_tex = len(tex_mesh.triangles) + 1
                    del fuera, dentro
            holder, hibrida = [tex_mesh], tex_mesh is not mesh
            del tex_mesh
            gc.collect()
            tex_info = texture_mesh(holder.pop(), kf, args.salida, dmax=tex_dmax, consume=hibrida,
                                    target_triangles=target_tex,
                                    page_size=args.pagina_textura,
                                    texture_scale=args.escala_textura,
                                    to_z_up=Y_UP_TO_Z_UP, scale=scale)

        # A Z arriba + escala
        M = Y_UP_TO_Z_UP.copy()
        M[:3, :3] *= scale
        mesh.transform(M)
        pcd.transform(M)
        mesh.compute_vertex_normals()

        pts = np.asarray(pcd.points)
        cols = (np.clip(np.asarray(pcd.colors), 0, 1) * 255).round().astype(np.uint8)

        out = lambda n: os.path.join(args.salida, n)
        avance("exportar", 0.0, "Guardando la nube")
        log("Exportando nube (PLY, E57)…")
        o3d.io.write_point_cloud(out("nube_fusionada.ply"), pcd, write_ascii=False)
        write_e57(out("nube_fusionada.e57"), pts, cols)
        avance("exportar", 0.7, "Guardando la malla para Navisworks y SolidWorks")
        log("Exportando malla (PLY, STL)…")
        o3d.io.write_triangle_mesh(out("malla_color.ply"), mesh, write_ascii=False)
        gc.collect()

        # malla para SolidWorks/Navisworks: suavizado Taubin (quita la ondulación del LiDAR sin
        # encoger), reducción por vóxel (poca memoria) y otro suavizado ligero. En la escena de prueba
        # esto quedó MÁS exacto que la malla sin tratar (95 %: 4.3 mm vs 6.2 mm) con la mitad de triángulos.
        stl = o3d.geometry.TriangleMesh(mesh)          # ya viene suavizada (acabado)
        zonas_z = zonas_a(M, zonas, scale) if n_detalle else []
        stl, stl_det = separar_zonas(stl, zonas_z)       # las piezas en detalle no se reducen
        if len(stl.triangles) > args.max_triangulos_stl:
            ratio = math.sqrt(len(stl.triangles) / args.max_triangulos_stl)
            stl = stl.simplify_vertex_clustering(voxel_size=args.voxel * ratio)
            stl.remove_degenerate_triangles()
            stl.remove_unreferenced_vertices()
        stl = stl.filter_smooth_taubin(number_of_iterations=5)
        if stl_det is not None:
            stl += stl_det
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
            "Sistema: Z hacia arriba; STL y OBJ en milímetros (nube en metros). Origen: donde arrancó el escaneo.",
            "Para medir e interferencias exactas: malla_solidworks.stl (solo lecturas de alta confianza).",
            *([f"Piezas enfocadas: {len(zonas)}; reconstruidas a 2 mm ({n_detalle:,} triángulos de detalle)."]
              if zonas else []),
            "Con colores y texturas: Navisworks -> obj/modelo_navisworks.fbx (junto a sus textura_*.jpg);",
            "SolidWorks 2021+ -> modelo_solidworks.3mf (textura dentro del archivo).",
            "El modelo con fotos usa esa misma malla exacta y además rellena zonas de confianza media",
            "(piso, objetos oscuros, lo lejano); en esas zonas rellenadas la medida puede variar unos mm.",
        ]
        if tex_info:
            rep.append(f"Modelo con fotos: {tex_info['triangles']:,} triángulos, {tex_info['photos_used']} fotos, "
                       f"{tex_info['pages']} página(s) de textura, {tex_info['coverage'] * 100:.1f} % cubierto "
                       "(modelo_texturizado.glb/.usdz en metros, Y arriba; obj/ para Navisworks en milímetros, Z arriba)")
        with open(out("reporte.txt"), "w") as f:
            f.write("\n".join(rep) + "\n")
        log("\n".join(rep))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
