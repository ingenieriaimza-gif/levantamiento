#!/usr/bin/env python3
"""
Coloca cada foto de 12 MP exactamente sobre el modelo.

La posición que ARKit da a una foto de máxima resolución trae un error de milímetros (y la corrección de
deriva que se le copia del cuadro vecino, otro tanto). En una foto de 12 MP eso son varios píxeles: al
pegarla, las líneas se quiebran entre una foto y la siguiente y los bordes de los tubos quedan corridos.

afinar_fotos_hr: busca los mismos detalles en la foto de 12 MP y en el cuadro normal más cercano en el
tiempo (que ya está alineado al modelo con el LiDAR), toma del modelo la posición 3D de esos detalles y
recalcula con ellos la posición de la foto (PnP con RANSAC). Si no hay suficientes detalles en común
(superficies lisas, foto movida), la foto se queda como estaba.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import open3d as o3d

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
MIN_COINCIDENCIAS = 40
MAX_MOVER = 0.04                  # m: más que esto no es un ajuste fino, es un error
MAX_GIRO = np.deg2rad(2.0)


def _gris(path, ancho):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None, 1.0
    f = ancho / g.shape[1]
    if f < 0.999:
        g = cv2.resize(g, (ancho, int(round(g.shape[0] * f))), interpolation=cv2.INTER_AREA)
    else:
        f = 1.0
    return g, f


def afinar_fotos_hr(fotos, frames, mesh, log=print, avance=None, ancho=2016):
    """fotos, frames: dicts con jpg, fx, fy, cx, cy, w, h, T (cámara→mundo de ARKit por columnas) y t.
    mesh: modelo ya alineado (mundo de ARKit, metros). Modifica fotos[i]["T"]. Devuelve cuántas se afinaron."""
    if not fotos or not len(mesh.triangles):
        return 0
    t0 = time.time()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    ts = np.array([fr.get("t", np.nan) for fr in frames], float)
    idx_ok = np.nonzero(np.isfinite(ts))[0]
    if not len(idx_ok):
        return 0

    def uno(ft):
        if "t" not in ft:
            return None
        j = idx_ok[np.argmin(np.abs(ts[idx_ok] - float(ft["t"])))]
        if abs(ts[j] - float(ft["t"])) > 1.0:
            return None
        fr = frames[j]
        g_lr, f_lr = _gris(fr["jpg"], ancho)
        g_hr, f_hr = _gris(ft["jpg"], ancho)
        if g_lr is None or g_hr is None:
            return None
        sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
        k1, d1 = sift.detectAndCompute(g_lr, None)
        k2, d2 = sift.detectAndCompute(g_hr, None)
        if d1 is None or d2 is None or len(k1) < MIN_COINCIDENCIAS or len(k2) < MIN_COINCIDENCIAS:
            return None
        m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
        buenos = [a for a, b in (p for p in m if len(p) == 2) if a.distance < 0.75 * b.distance]
        if len(buenos) < MIN_COINCIDENCIAS:
            return None
        # píxeles en la foto completa (centro de píxel)
        p1 = (np.array([k1[a.queryIdx].pt for a in buenos]) + 0.5) / f_lr - 0.5
        p2 = (np.array([k2[a.trainIdx].pt for a in buenos]) + 0.5) / f_hr - 0.5
        # posición 3D de cada detalle: rayo del cuadro normal contra el modelo
        Tc = np.array(fr["T"], float).reshape(4, 4).T @ ARKIT_TO_CV
        d = np.stack([(p1[:, 0] - fr["cx"]) / fr["fx"], (p1[:, 1] - fr["cy"]) / fr["fy"], np.ones(len(p1))], 1)
        dw = d @ Tc[:3, :3].T
        rays = np.concatenate([np.broadcast_to(Tc[:3, 3], dw.shape), dw], 1).astype(np.float32)
        th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        ok = np.isfinite(th) & (th > 0.15) & (th < 6.0)
        if ok.sum() < MIN_COINCIDENCIAS:
            return None
        P = (rays[ok, :3] + rays[ok, 3:] * th[ok, None]).astype(np.float64)
        q = p2[ok].astype(np.float64)
        K = np.array([[ft["fx"], 0, ft["cx"]], [0, ft["fy"], ft["cy"]], [0, 0, 1]], float)
        T0 = np.array(ft["T"], float).reshape(4, 4).T @ ARKIT_TO_CV          # cámara (CV) -> mundo
        W0 = np.linalg.inv(T0)
        rvec0, _ = cv2.Rodrigues(W0[:3, :3])
        tvec0 = W0[:3, 3].reshape(3, 1).copy()
        err = max(2.0, 3.0 * float(ft["w"]) / 4032.0)
        try:
            okp, rvec, tvec, inl = cv2.solvePnPRansac(P, q, K, None, rvec0.copy(), tvec0.copy(), useExtrinsicGuess=True,
                                                      iterationsCount=300, reprojectionError=err, confidence=0.999,
                                                      flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            return None
        if not okp or inl is None or len(inl) < MIN_COINCIDENCIAS:
            return None
        inl = inl.ravel()
        # los detalles deben estar repartidos en la foto (si no, el giro queda mal determinado)
        if np.ptp(q[inl, 0]) < 0.35 * ft["w"] or np.ptp(q[inl, 1]) < 0.35 * ft["h"]:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(P[inl], q[inl], K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        Wn = np.eye(4)
        Wn[:3, :3], Wn[:3, 3] = R, tvec.ravel()
        Tn = np.linalg.inv(Wn)
        mover = np.linalg.norm(Tn[:3, 3] - T0[:3, 3])
        giro = np.arccos(np.clip((np.trace(Tn[:3, :3].T @ T0[:3, :3]) - 1) / 2, -1, 1))
        if mover > MAX_MOVER or giro > MAX_GIRO:
            return None
        return (Tn @ ARKIT_TO_CV), mover, giro, len(inl)

    from paralelo import nucleos
    n, mov, gir = 0, [], []
    with ThreadPoolExecutor(max_workers=max(1, min(8, nucleos() // 2))) as ex:
        for i, res in enumerate(ex.map(uno, fotos)):
            if avance is not None and i % 5 == 0:
                avance(i / len(fotos), f"Colocando las fotos de 12 MP sobre el modelo ({i + 1} de {len(fotos)})")
            if res is None:
                continue
            T, mover, giro, _ = res
            fotos[i]["T"] = list(T.T.reshape(-1))
            n += 1
            mov.append(mover)
            gir.append(giro)
    if n:
        log(f"  fotos de 12 MP colocadas sobre el modelo: {n} de {len(fotos)}; ajuste mediano "
            f"{np.median(mov) * 1000:.1f} mm y {np.degrees(np.median(gir)):.3f}° ({time.time() - t0:.0f} s)")
    else:
        log(f"  fotos de 12 MP: no hubo detalles suficientes para afinarlas; se usan como llegaron")
    return n
