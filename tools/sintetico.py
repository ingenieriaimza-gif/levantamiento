#!/usr/bin/env python3
"""
Genera un paquete RGB-D sintético con el mismo formato que la app (levanta-rgbd) a partir de
una escena conocida (un skid con tuberías), para validar la fusión contra la geometría real.

Simula: cámara 1920×1440 (se guarda reducida), LiDAR 256×192, ruido de profundidad que crece con
la distancia, confianza baja en ángulos rasantes y lejanos, y una pequeña vibración del seguimiento.

Uso: python3 -I sintetico.py --salida paquete_sint --cuadros 120
Escribe además escena_real.ply (malla de referencia, mundo ARKit Y arriba, metros).
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import open3d as o3d

FULL_W, FULL_H = 1920, 1440
FX, FY, CX, CY = 1452.3, 1452.3, 959.6, 719.2      # intrínsecos típicos de cámara ancha
DEPTH_W, DEPTH_H = 256, 192
IMG_W, IMG_H = 512, 384                            # foto guardada (la app usa 1440×1080)


def cyl(radius, length, axis, center, color):
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=96, split=4)
    # el cilindro de Open3D va a lo largo de Z
    if axis == "x":
        R = m.get_rotation_matrix_from_xyz((0, math.pi / 2, 0))
    elif axis == "y":
        R = m.get_rotation_matrix_from_xyz((math.pi / 2, 0, 0))
    else:
        R = np.eye(3)
    m.rotate(R, center=(0, 0, 0))
    m.translate(center)
    m.paint_uniform_color(color)
    return m


def box(size, center, color):
    m = o3d.geometry.TriangleMesh.create_box(*size)
    m.translate(np.array(center) - np.array(size) / 2)
    m.paint_uniform_color(color)
    return m


def build_scene():
    parts = []
    # piso y muro (mundo ARKit: Y arriba)
    parts.append(box((5.0, 0.02, 4.0), (0, -0.01, 0), (0.55, 0.55, 0.52)))
    parts.append(box((5.0, 3.0, 0.05), (0, 1.5, -1.6), (0.82, 0.82, 0.80)))
    # bastidor del skid (PTR 75 mm)
    for x in (-1.2, 1.2):
        for z in (-0.6, 0.6):
            parts.append(box((0.075, 0.9, 0.075), (x, 0.45, z), (0.35, 0.37, 0.40)))
    parts.append(box((2.475, 0.075, 0.075), (0, 0.9, -0.6), (0.35, 0.37, 0.40)))
    parts.append(box((2.475, 0.075, 0.075), (0, 0.9, 0.6), (0.35, 0.37, 0.40)))
    # tubería sanitaria 2" (OD 50.8) y 3" (OD 76.2) a lo largo de X
    parts.append(cyl(0.0254, 2.6, "x", (0, 1.05, 0.25), (0.75, 0.76, 0.78)))
    parts.append(cyl(0.0381, 2.6, "x", (0, 1.20, -0.10), (0.70, 0.72, 0.75)))
    # bajante vertical 1.5" (OD 38.1)
    parts.append(cyl(0.01905, 1.2, "y", (0.9, 0.6, 0.25), (0.75, 0.76, 0.78)))
    # tanque vertical Ø 400
    parts.append(cyl(0.20, 0.7, "y", (-0.7, 1.35, -0.25), (0.78, 0.79, 0.80)))
    # motor (caja) azul
    parts.append(box((0.45, 0.28, 0.30), (0.35, 1.06, -0.45), (0.15, 0.30, 0.65)))
    mesh = o3d.geometry.TriangleMesh()
    for p in parts:
        mesh += p
    mesh.compute_vertex_normals()
    return mesh


def look_at_arkit(eye, target):
    """Matriz cámara→mundo con la convención de ARKit (cámara mira hacia -Z, Y arriba)."""
    f = target - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, up, -f, eye
    return T


def pixel_rays(T, w, h):
    """Rayos por centro de píxel a la resolución w×h, usando los intrínsecos de 1920×1440."""
    s = w / FULL_W
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    xf = (u + 0.5) / s - 0.5          # coordenada en la imagen completa (píxel centrado)
    yf = (v + 0.5) / s - 0.5
    dx = (xf - CX) / FX
    dy = -(yf - CY) / FY
    dirs_cam = np.stack([dx, dy, -np.ones_like(dx)], -1).reshape(-1, 3)
    dirs = dirs_cam @ T[:3, :3].T
    origins = np.broadcast_to(T[:3, 3], dirs.shape)
    return np.concatenate([origins, dirs], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="paquete_sint")
    ap.add_argument("--cuadros", type=int, default=120)
    ap.add_argument("--semilla", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.semilla)

    mesh = build_scene()
    os.makedirs(os.path.join(args.salida, "frames"), exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(args.salida, "..", "escena_real.ply"), mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    tri_colors = np.asarray(mesh.vertex_colors)[np.asarray(mesh.triangles)[:, 0]]

    # recorrido: arco frente al skid, a dos alturas, mirando a distintos puntos
    lines = []
    n = args.cuadros
    for i in range(n):
        a = -1.1 + 2.2 * i / (n - 1)
        r = 1.35 + 0.15 * math.sin(i * 0.7)
        height = 1.15 + 0.35 * (i % 2)
        eye = np.array([1.6 * math.sin(a), height, 0.15 + r * math.cos(a)])
        target = np.array([0.9 * math.sin(a * 1.3), 0.95, -0.15]) + rng.normal(0, 0.05, 3)
        T_true = look_at_arkit(eye, target)

        # imagen de color
        rays = pixel_rays(T_true, IMG_W, IMG_H)
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        ids = hit["primitive_ids"].numpy()
        ok = ids != scene.INVALID_ID
        col = np.zeros((IMG_H * IMG_W, 3))
        col[ok] = tri_colors[ids[ok]]
        nrm = hit["primitive_normals"].numpy()
        shade = 0.55 + 0.45 * np.abs(nrm @ np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3]))
        col = col * shade[:, None]
        # textura ligera para que la foto tenga detalle
        tex = 0.9 + 0.1 * np.sin(np.arange(IMG_W * IMG_H) * 0.37)
        img = (np.clip(col * tex[:, None], 0, 1) * 255).astype(np.uint8).reshape(IMG_H, IMG_W, 3)
        cv2.imwrite(os.path.join(args.salida, "frames", f"{i:06d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])

        # profundidad LiDAR (z-depth porque la componente z del rayo en cámara es -1)
        rays_d = pixel_rays(T_true, DEPTH_W, DEPTH_H)
        hd = scene.cast_rays(o3d.core.Tensor(rays_d))
        z = hd["t_hit"].numpy()
        nd = hd["primitive_normals"].numpy()
        dirs = rays_d[:, 3:] / np.linalg.norm(rays_d[:, 3:], axis=1, keepdims=True)
        cosang = np.abs(np.sum(nd * dirs, 1))
        valid = np.isfinite(z)
        sigma = 0.0025 + 0.0025 * np.where(valid, z, 0)          # 2.5 mm + 2.5 mm/m
        zn = np.where(valid, z + rng.normal(0, 1, z.shape) * sigma, 0)
        conf = np.full(z.shape, 2, np.uint8)
        conf[cosang < 0.35] = 1
        conf[(zn > 3.0) | (cosang < 0.12)] = 0
        mm = np.clip(np.round(zn * 1000), 0, 65535).astype("<u2")
        mm[~valid] = 0
        mm.tofile(os.path.join(args.salida, "frames", f"{i:06d}.depth"))
        conf.astype(np.uint8).tofile(os.path.join(args.salida, "frames", f"{i:06d}.conf"))

        # pose "medida" con pequeña vibración de seguimiento (1 mm, 0.05°)
        T_meas = T_true.copy()
        T_meas[:3, 3] += rng.normal(0, 0.001, 3)
        ang = rng.normal(0, math.radians(0.05), 3)
        T_meas[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(ang) @ T_meas[:3, :3]

        s = IMG_W / FULL_W
        lines.append(json.dumps({
            "i": i, "t": i * 0.3,
            "fx": FX * s, "fy": FY * s,
            "cx": (CX + 0.5) * s - 0.5, "cy": (CY + 0.5) * s - 0.5,
            "w": IMG_W, "h": IMG_H,
            "T": T_meas.T.reshape(-1).tolist(),       # por columnas, como la app
        }))

    with open(os.path.join(args.salida, "frames.jsonl"), "w") as f:
        f.write("\n".join(lines) + "\n")
    meta = {"format": "levanta-rgbd", "version": 1, "frames": n,
            "imageWidth": IMG_W, "imageHeight": IMG_H,
            "depthWidth": DEPTH_W, "depthHeight": DEPTH_H, "depthUnits": "mm",
            "poseConvention": "ARKit camera-to-world, column-major 4x4", "device": "simulado"}
    with open(os.path.join(args.salida, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Paquete sintético: {n} cuadros en {args.salida}")


if __name__ == "__main__":
    main()
