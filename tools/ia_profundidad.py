#!/usr/bin/env python3
"""
Profundidad con IA: Prompt Depth Anything (Apache-2.0) toma la foto del iPhone y el mapa del LiDAR
(256×192) y devuelve profundidad métrica de mucha mayor resolución, con bordes definidos por la foto.

Resguardo para que sirva para medir: el LiDAR manda.
  - En cada cuadro la IA se compara completa contra el LiDAR: se ajusta su escala y desfase, y si no
    coincide de forma consistente se descarta ese cuadro (solo LiDAR).
  - Por punto, la IA solo se acepta si coincide con el LiDAR dentro de 1 cm o 1 %.
  - Donde el LiDAR no tiene dato, la IA se usa solo si está cerca de lecturas válidas del LiDAR
    (rellena huecos pequeños como bordes, superficies oscuras), no en zonas que el LiDAR nunca vio.

Requiere torch y transformers (se instalan aparte desde el programa: "Inteligencia artificial").
"""
import os
import sys

import cv2
import numpy as np

MODEL_ID = "depth-anything/prompt-depth-anything-vits-hf"

# carpeta opcional con los paquetes de IA instalados por el programa (no vienen en el instalador base)
_RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IA_DIR = os.path.join(_RAIZ, "ia")
IA_PAQUETES = os.path.join(IA_DIR, "paquetes")
IA_MODELOS = os.path.join(IA_DIR, "modelos")
if os.path.isdir(IA_PAQUETES) and IA_PAQUETES not in sys.path:
    sys.path.append(IA_PAQUETES)          # al final: las librerías base tienen prioridad


def disponible():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:       # noqa: BLE001
        return False


def _fill_holes(depth_mm):
    """Rellena huecos del LiDAR con el valor válido más cercano (el modelo espera un mapa denso)."""
    valid = depth_mm > 0
    if valid.all() or not valid.any():
        return depth_mm.astype(np.float32)
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return depth_mm[tuple(idx)].astype(np.float32)


class RefinadorIA:
    def __init__(self, device=None, model_id=MODEL_ID, cache_dir=None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        cache_dir = cache_dir or (IA_MODELOS if os.path.isdir(IA_MODELOS) else None)
        self.proc = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()
        self.half = device == "cuda"
        if self.half:
            self.model = self.model.half()

    def predecir(self, rgb, depth_mm, out_size):
        """rgb: H×W×3 uint8 (RGB); depth_mm: h×w uint16 del LiDAR; out_size: (ancho, alto). Devuelve metros."""
        from PIL import Image
        prompt = _fill_holes(depth_mm)
        inputs = self.proc(images=Image.fromarray(rgb), prompt_depth=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.half:
            inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model(**inputs)
        W, H = out_size
        res = self.proc.post_process_depth_estimation(out, target_sizes=[(H, W)])
        return res[0]["predicted_depth"].float().cpu().numpy().astype(np.float32)


def combinar(ia_m, lidar_mm, conf, tol_abs=0.01, tol_rel=0.01, min_conf=2, max_hueco_px=6,
             max_dispersion=0.015):
    """
    Combina la profundidad de la IA (alta resolución, metros) con la del LiDAR (baja resolución, mm).

    1. Compara toda la imagen de la IA contra el LiDAR confiable y ajusta escala y desfase
       (robusto, sin dejarse llevar por puntos raros). Si aun así no coinciden de forma
       consistente, la IA se descarta en este cuadro y se usa solo el LiDAR.
    2. Por punto, la IA solo se acepta si coincide con el LiDAR dentro de tol_abs o tol_rel.
    3. Rellena huecos pequeños del LiDAR (cerca de lecturas válidas) solo si el cuadro pasó el paso 1.

    Devuelve (profundidad_mm uint16 a la resolución de la IA, fracción de píxeles donde aportó la IA).
    """
    H, W = ia_m.shape
    lid = lidar_mm.astype(np.float32) / 1000.0
    ok = lidar_mm > 0
    if conf is not None:
        ok &= conf >= min_conf
    lid_up = cv2.resize(np.where(ok, lid, 0).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    ok_up = lid_up > 0
    base = np.clip(np.round(lid_up * 1000), 0, 65535).astype(np.uint16)

    # 1) alinear la IA al LiDAR (lid ≈ a·ia + b) con mínimos cuadrados reponderados
    ia_s = cv2.resize(ia_m, (lidar_mm.shape[1], lidar_mm.shape[0]), interpolation=cv2.INTER_AREA)
    m = ok & np.isfinite(ia_s) & (ia_s > 0.1)
    if m.sum() < 500:
        return base, 0.0
    x, y = ia_s[m].astype(np.float64), lid[m].astype(np.float64)
    a, b = 1.0, 0.0
    w = np.ones_like(x)
    for _ in range(6):
        A = np.stack([x * w, w], 1)
        a, b = np.linalg.lstsq(A, y * w, rcond=None)[0]
        r = y - (a * x + b)
        sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = 1.0 / np.maximum(1.0, np.abs(r) / (2.5 * sigma))       # Huber
    r = y - (a * x + b)
    dispersion = 1.4826 * np.median(np.abs(r - np.median(r)))
    if not (0.9 < a < 1.1) or dispersion > max_dispersion:
        return base, 0.0                                           # la IA no es confiable en este cuadro
    ia_al = (a * ia_m + b).astype(np.float32)

    # 2) aceptar por punto solo donde coincide con el LiDAR
    tol = np.maximum(tol_abs, tol_rel * lid_up)
    agree = ok_up & (np.abs(ia_al - lid_up) <= tol)

    # 3) huecos pequeños del LiDAR cerca de lecturas válidas
    near = cv2.dilate(ok.astype(np.uint8), np.ones((2 * max_hueco_px + 1,) * 2, np.uint8)) > 0
    near_up = cv2.resize(near.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    fill = (~ok_up) & near_up & np.isfinite(ia_al) & (ia_al > 0.15)

    # en bordes (saltos de profundidad del LiDAR) el LiDAR mezcla frente y fondo: si la IA no
    # coincide ahí, mejor dejar sin dato que usar el borde borroso del LiDAR
    lid_f = np.where(ok, lid, np.nan).astype(np.float32)
    lo = cv2.erode(np.nan_to_num(lid_f, nan=99.0), np.ones((3, 3), np.uint8))
    hi = cv2.dilate(np.nan_to_num(lid_f, nan=0.0), np.ones((3, 3), np.uint8))
    edge = ((hi - lo) > np.maximum(0.03, 0.03 * np.nan_to_num(lid_f, nan=1.0))) & ok
    edge_up = cv2.resize(edge.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    out = np.where(agree | fill, ia_al, np.where(edge_up, 0, lid_up))
    out = np.where(ok_up | fill, out, 0)
    mm = np.clip(np.round(out * 1000), 0, 65535).astype(np.uint16)
    return mm, float((agree | fill).mean())


def refinar_cuadro(refinador, rgb, depth_mm, conf, out_size):
    ia = refinador.predecir(rgb, depth_mm, out_size)
    return combinar(ia, depth_mm, conf)


if __name__ == "__main__":
    # prueba rápida: python ia_profundidad.py foto.jpg profundidad.depth confianza.conf 256 192
    import time
    img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    dw, dh = int(sys.argv[4]), int(sys.argv[5])
    d = np.fromfile(sys.argv[2], "<u2").reshape(dh, dw)
    c = np.fromfile(sys.argv[3], np.uint8).reshape(dh, dw)
    r = RefinadorIA()
    t = time.time()
    mm, frac = refinar_cuadro(r, img, d, c, (768, 576))
    print(f"dispositivo {r.device}, {time.time() - t:.2f} s, IA aportó en {frac * 100:.1f} % de los píxeles")
