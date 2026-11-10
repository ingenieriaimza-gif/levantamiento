#!/usr/bin/env python3
"""
Exporta un paquete RGB-D de la app 3D al formato COLMAP (texto) para entrenar Gaussian Splatting
en Brush, Nerfstudio/gsplat, Postshot u otros, usando las posiciones exactas del iPhone (sin tener
que estimarlas) y la nube del LiDAR como punto de partida.

Salida:
  <salida>/images/000000.jpg …          fotos (descarta las más borrosas)
  <salida>/sparse/0/cameras.txt         una cámara PINHOLE por foto (enfoque automático)
  <salida>/sparse/0/images.txt          orientación y posición de cada foto (mundo → cámara)
  <salida>/sparse/0/points3D.txt        puntos iniciales con color (de la fusión o del LiDAR)
  <salida>/transforms.json              lo mismo en formato Nerfstudio

Mundo: el de ARKit (metros, Y arriba). Convenciones COLMAP: cámara x derecha, y abajo, z adelante;
el centro del píxel superior izquierdo es (0.5, 0.5).

Uso:
  python3 -I exportar_splat.py parte1.zip parte2.zip … --salida splat_oficina
  python3 -I exportar_splat.py paquete/ --salida splat --nube salida/nube_fusionada.ply
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from procesar_rgbd import (ARKIT_TO_CV, Y_UP_TO_Z_UP, extract_inputs, load_frames, pose_matrix,
                           sharpness)


def rot_to_quat(R):
    """Matriz de rotación -> cuaternión (w, x, y, z), con w >= 0."""
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def lidar_points(meta, frames, max_points, dmax):
    """Nube inicial directa del LiDAR (alta confianza), si no se da una fusionada."""
    dw, dh = int(meta["depthWidth"]), int(meta["depthHeight"])
    pts, cols = [], []
    for fr in frames[::2]:
        d = np.fromfile(fr["depth"], "<u2").reshape(dh, dw) / 1000.0
        c = np.fromfile(fr["conf"], np.uint8).reshape(dh, dw) if os.path.exists(fr["conf"]) else np.full((dh, dw), 2)
        img = cv2.imread(fr["jpg"])
        sx, sy = dw / fr["w"], dh / fr["h"]
        fx, fy = fr["fx"] * sx, fr["fy"] * sy
        cx, cy = (fr["cx"] + 0.5) * sx - 0.5, (fr["cy"] + 0.5) * sy - 0.5
        v, u = np.nonzero((c >= 2) & (d > 0.15) & (d < dmax))
        z = d[v, u]
        P = np.stack([(u - cx) * z / fx, -(v - cy) * z / fy, -z, np.ones_like(z)], 1)
        pts.append((P @ pose_matrix(fr).T)[:, :3])
        iu = np.clip(((u + 0.5) / sx - 0.5).round().astype(int), 0, img.shape[1] - 1)
        iv = np.clip(((v + 0.5) / sy - 0.5).round().astype(int), 0, img.shape[0] - 1)
        cols.append(img[iv, iu][:, ::-1])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(cols) / 255.0)
    pcd = pcd.voxel_down_sample(0.01)
    return pcd


def main():
    ap = argparse.ArgumentParser(description="Paquete RGB-D -> COLMAP / Nerfstudio para Gaussian Splatting")
    ap.add_argument("entradas", nargs="+")
    ap.add_argument("--salida", default="splat")
    ap.add_argument("--nube", help="nube_fusionada.ply (Z arriba) de procesar_rgbd.py, para los puntos iniciales")
    ap.add_argument("--max-puntos", type=int, default=400_000)
    ap.add_argument("--dmax", type=float, default=3.5)
    ap.add_argument("--escala", type=float, default=1.0,
                    help="factor de calibración (el mismo del modelo): la vista queda en las mismas medidas")
    ap.add_argument("--umbral-borrosas", type=float, default=0.35,
                    help="descarta fotos con nitidez menor a esta fracción de la mediana")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="splat_")
    try:
        dirs = extract_inputs(args.entradas, work)
        meta, frames = load_frames(dirs)
        sh = np.array([sharpness(fr["jpg"]) for fr in frames])
        keep = sh >= args.umbral_borrosas * np.median(sh)
        frames = [fr for fr, k in zip(frames, keep) if k]
        print(f"{len(frames)} fotos ({int((~keep).sum())} borrosas descartadas)")

        out = args.salida
        img_dir = os.path.join(out, "images")
        sp_dir = os.path.join(out, "sparse", "0")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(sp_dir, exist_ok=True)

        esc = float(args.escala)

        def pose(fr):
            """Cámara -> mundo de ARKit, con la calibración de escala aplicada a la posición."""
            T = pose_matrix(fr).copy()
            T[:3, 3] *= esc
            return T

        cams, imgs, ns_frames = [], [], []
        for n, fr in enumerate(frames, start=1):
            name = f"{int(fr['i']):06d}.jpg"
            shutil.copy(fr["jpg"], os.path.join(img_dir, name))
            w, h = int(fr["w"]), int(fr["h"])
            # COLMAP: centro del píxel superior izquierdo en (0.5, 0.5)
            cams.append(f"{n} PINHOLE {w} {h} {fr['fx']:.6f} {fr['fy']:.6f} "
                        f"{fr['cx'] + 0.5:.6f} {fr['cy'] + 0.5:.6f}")
            c2w_cv = pose(fr) @ ARKIT_TO_CV
            w2c = np.linalg.inv(c2w_cv)
            q = rot_to_quat(w2c[:3, :3])
            t = w2c[:3, 3]
            imgs.append(f"{n} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                        f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {n} {name}")
            imgs.append("")                       # línea de puntos 2D (vacía)
            # Nerfstudio usa la convención OpenGL (= la de ARKit) para la matriz cámara->mundo
            ns_frames.append({"file_path": f"images/{name}", "transform_matrix": pose(fr).tolist(),
                              "fl_x": fr["fx"], "fl_y": fr["fy"], "cx": fr["cx"] + 0.5, "cy": fr["cy"] + 0.5,
                              "w": w, "h": h})

        with open(os.path.join(sp_dir, "cameras.txt"), "w") as f:
            f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n" + "\n".join(cams) + "\n")
        with open(os.path.join(sp_dir, "images.txt"), "w") as f:
            f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[]\n" + "\n".join(imgs) + "\n")

        # puntos iniciales
        if args.nube:
            pcd = o3d.io.read_point_cloud(args.nube)
            pcd.transform(np.linalg.inv(Y_UP_TO_Z_UP))          # de vuelta al mundo de ARKit (ya viene calibrada)
        else:
            pcd = lidar_points(meta, frames, args.max_puntos, args.dmax)
            pcd.scale(esc, center=np.zeros(3))
        P = np.asarray(pcd.points)
        Cc = (np.clip(np.asarray(pcd.colors), 0, 1) * 255).round().astype(int)
        if len(P) > args.max_puntos:
            sel = np.random.default_rng(0).choice(len(P), args.max_puntos, replace=False)
            P, Cc = P[sel], Cc[sel]
        with open(os.path.join(sp_dir, "points3D.txt"), "w") as f:
            f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
            for k, (p, c) in enumerate(zip(P, Cc), start=1):
                f.write(f"{k} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]} 0\n")
        pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        pcd_out.colors = o3d.utility.Vector3dVector(Cc / 255.0)
        o3d.io.write_point_cloud(os.path.join(out, "puntos_iniciales.ply"), pcd_out)

        with open(os.path.join(out, "transforms.json"), "w") as f:
            json.dump({"camera_model": "OPENCV", "frames": ns_frames,
                       "ply_file_path": "puntos_iniciales.ply"}, f, indent=1)
        print(f"Listo: {len(frames)} fotos, {len(P):,} puntos iniciales en {out}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
