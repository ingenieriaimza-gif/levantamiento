#!/usr/bin/env python3
"""
Diagnóstico de la IA de profundidad contra la geometría exacta del simulador.
Para varios cuadros compara, por zonas (planas / bordes), el error de:
  LiDAR (vecino más cercano), IA cruda, IA alineada al LiDAR, y la combinación con resguardo.
Uso: python diagnostico_ia.py paquete escena_real.ply [cuadros]
"""
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import procesar_rgbd as P  # noqa: E402
from ia_profundidad import RefinadorIA, combinar  # noqa: E402

pkg, gt_path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
meta, frames = P.load_frames([pkg])
dw, dh = meta["depthWidth"], meta["depthHeight"]
gt = o3d.io.read_triangle_mesh(gt_path)
sc = o3d.t.geometry.RaycastingScene()
sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
ia = RefinadorIA()
W = 768
rows = []
for fr in frames[:: max(1, len(frames) // n)][:n]:
    img = cv2.cvtColor(cv2.imread(fr["jpg"]), cv2.COLOR_BGR2RGB)
    H = int(round(W * img.shape[0] / img.shape[1]))
    d = np.fromfile(fr["depth"], "<u2").reshape(dh, dw)
    c = np.fromfile(fr["conf"], np.uint8).reshape(dh, dw)
    # profundidad exacta a la resolución de trabajo
    s = W / fr["w"]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    xf, yf = (u + .5) / s - .5, (v + .5) / s - .5
    dirs = np.stack([(xf - fr["cx"]) / fr["fx"], -(yf - fr["cy"]) / fr["fy"], -np.ones_like(xf)], -1).reshape(-1, 3)
    T = P.pose_matrix(fr)
    dw_ = dirs @ T[:3, :3].T
    rays = np.concatenate([np.broadcast_to(T[:3, 3], dw_.shape), dw_], 1).astype(np.float32)
    z = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(H, W)
    valid = np.isfinite(z) & (z < 3.0)
    zr = np.where(valid, z, 0).astype(np.float32)
    lo = cv2.erode(np.where(valid, zr, 99).astype(np.float32), np.ones((5, 5), np.uint8))
    hi = cv2.dilate(zr, np.ones((5, 5), np.uint8))
    edge = valid & ((hi - lo) > 0.03)
    flat = valid & ~edge

    raw = ia.predecir(img, d, (W, H))
    lid_up = cv2.resize((d / 1000.0).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    comb_mm, frac = combinar(raw, d, c)
    comb = comb_mm.astype(np.float32) / 1000.0
    # IA alineada (lo mismo que hace combinar internamente, para medir el ajuste)
    m = (cv2.resize(lid_up, (dw, dh), interpolation=cv2.INTER_NEAREST) > 0) & (c >= 2)
    ia_s = cv2.resize(raw, (dw, dh), interpolation=cv2.INTER_AREA)
    a, b = np.polyfit(ia_s[m], (d / 1000.0)[m], 1)
    al = a * raw + b

    def err(x, zone):
        k = zone & (x > 0)
        return (np.median(np.abs(x[k] - z[k])) * 1000 if k.any() else float("nan"), k.sum() / max(zone.sum(), 1))
    row = {"cuadro": int(fr["i"]), "aporte_ia": round(frac, 2), "a": round(float(a), 4), "b_mm": round(float(b) * 1000, 1)}
    for name, x in (("lidar", lid_up), ("ia_cruda", raw), ("ia_alineada", al), ("combinada", comb)):
        ef, cf = err(x, flat)
        ee, ce = err(x, edge)
        row[name] = f"plano {ef:5.1f} mm ({cf * 100:3.0f}%) | borde {ee:5.1f} mm ({ce * 100:3.0f}%)"
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
