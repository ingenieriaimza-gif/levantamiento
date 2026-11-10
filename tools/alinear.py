"""
Corrección de la deriva del iPhone.

ARKit calcula dónde está el teléfono en cada cuadro, pero en escaneos largos se va corriendo unos
milímetros o centímetros. Si el mismo objeto se vuelve a ver después de que la posición se corrió,
la fusión lo pone en dos lugares: aparece una "segunda capa" 1–3 cm detrás (puertas, paredes y
estantes dobles que en los visores se ven como manchas rasgadas).

refinar_poses: alinea cada cuadro contra el modelo ya formado (ICP punto-a-plano, como hacen los
programas de escaneo profesionales) y corrige su posición. Después se vuelve a formar el modelo con
las posiciones corregidas; las fotos también quedan mejor colocadas.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open3d as o3d

# Límites de una corrección creíble (más que esto es un error de alineación, no deriva)
MAX_MOVER = 0.06              # m
MAX_GIRO = np.deg2rad(3.0)


def _nube_cuadro(fr, dw, dh, ARKIT_TO_CV, paso=2, dmax=3.0):
    d = np.fromfile(fr["depth"], "<u2")
    c = np.fromfile(fr["conf"], np.uint8) if fr.get("conf") else None
    if d.size != dw * dh:
        return None
    d = d.reshape(dh, dw)[::paso, ::paso].astype(np.float64) / 1000.0
    ok = (d > 0.2) & (d < dmax)
    if c is not None and c.size == dw * dh:
        ok &= c.reshape(dh, dw)[::paso, ::paso] >= 2
    vv, uu = np.nonzero(ok)
    if len(vv) < 500:
        return None
    z = d[vv, uu]
    sx, sy = dw / float(fr["w"]), dh / float(fr["h"])
    fx, fy = fr["fx"] * sx, fr["fy"] * sy
    cx, cy = (fr["cx"] + 0.5) * sx - 0.5, (fr["cy"] + 0.5) * sy - 0.5
    u, v = uu * paso, vv * paso
    pc = np.c_[(u - cx) / fx * z, (v - cy) / fy * z, z]            # cámara (convención CV)
    T = np.array(fr["T"], float).reshape(4, 4).T @ ARKIT_TO_CV       # cámara -> mundo
    return pc @ T[:3, :3].T + T[:3, 3]


def refinar_poses(meta, frames, mesh, ARKIT_TO_CV, avance=None, log=print):
    """Corrige frames[i]["T"] (en el lugar) alineando cada cuadro contra `mesh`. Devuelve cuántos se corrigieron."""
    t0 = time.time()
    dw, dh = int(meta["depthWidth"]), int(meta["depthHeight"])
    # modelo de referencia: puntos con normales cada ~6 mm
    ref = o3d.geometry.PointCloud(mesh.vertices)
    ref.normals = mesh.vertex_normals if mesh.has_vertex_normals() else o3d.utility.Vector3dVector()
    if not ref.has_normals():
        ref.estimate_normals()
    ref = ref.voxel_down_sample(0.006)
    arbol = o3d.geometry.KDTreeFlann(ref)          # noqa: F841 (se construye una vez dentro de cada ICP)
    P = np.asarray(ref.points)

    def una(k):
        fr = frames[k]
        pts = _nube_cuadro(fr, dw, dh, ARKIT_TO_CV)
        if pts is None:
            return k, None
        # referencia recortada a lo que ve este cuadro (+10 cm): el ICP es mucho más rápido
        lo, hi = pts.min(0) - 0.10, pts.max(0) + 0.10
        sel = np.nonzero(np.all((P >= lo) & (P <= hi), axis=1))[0]
        if len(sel) < 1000:
            return k, None
        tgt = ref.select_by_index(sel)
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(0.008)
        crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        W = np.eye(4)
        for dist in (0.04, 0.015):
            r = o3d.pipelines.registration.registration_icp(src, tgt, dist, W, est, crit)
            W = r.transformation
        if r.fitness < 0.5:
            return k, None
        ang = np.arccos(np.clip((np.trace(W[:3, :3]) - 1) / 2, -1, 1))
        if np.linalg.norm(W[:3, 3]) > MAX_MOVER or ang > MAX_GIRO:
            return k, None
        return k, W

    from paralelo import nucleos
    correcciones = {}
    with ThreadPoolExecutor(max_workers=nucleos()) as ex:
        for j, (k, W) in enumerate(ex.map(una, range(len(frames)))):
            if W is not None:
                correcciones[k] = W
            if avance and j % 10 == 0:
                avance(j / max(1, len(frames)), f"Corrigiendo la deriva del iPhone ({j + 1} de {len(frames)})")
    movs = []
    for k, W in correcciones.items():
        Tw = np.array(frames[k]["T"], float).reshape(4, 4).T          # cámara ARKit -> mundo
        frames[k]["T"] = list((W @ Tw).T.reshape(-1))
        movs.append(np.linalg.norm(W[:3, 3]) * 1000)
    if movs:
        log(f"  deriva corregida en {len(movs)} de {len(frames)} cuadros: mediana {np.median(movs):.1f} mm, "
            f"máx. {np.max(movs):.0f} mm ({time.time() - t0:.0f} s)")
    return len(movs)
