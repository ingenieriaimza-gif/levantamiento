"""
El proceso se ajusta a la PC y al tamaño del espacio escaneado.

Antes todo iba con ajustes fijos pensados para una PC grande (3 millones de triángulos, vóxel de 4 mm sin
importar el tamaño del lugar, texturas sin tope). En una PC de 5 GB eso pedía más memoria de la que hay:
Windows mandaba la memoria al disco y un modelo tardaba horas o no terminaba nunca.

Aquí se decide, según la RAM instalada:
  · cuántos triángulos lleva el modelo con fotos y cuántas páginas de textura,
  · cuántos procesos trabajan a la vez (cada uno copia parte de la malla),
  · hasta qué detalle llega la superficie continua (Poisson),
y, según el tamaño del espacio, el tamaño de vóxel para que el volumen SIEMPRE quepa en memoria
(un cuarto se une a 4 mm; una nave grande, más grueso: lo que la PC puede manejar).

Todo se puede forzar para pruebas con LEVANTA_PERFIL=ligera|media|completa.
"""
import os

import numpy as np


def ram_total_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:       # noqa: BLE001
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _M(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                           [(n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                              "ullAvailPageFile", "ullTotalVirtual",
                                                              "ullAvailVirtual", "ullAvailExtendedVirtual")]
            m = _M()
            m.dwLength = ctypes.sizeof(_M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1024 ** 3
        except Exception:       # noqa: BLE001
            pass
    return 8.0


def ram_libre_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:       # noqa: BLE001
        return ram_total_gb() * 0.5


_PERFILES = {
    #            triángulos   páginas 4096²  procesos  Poisson  rondas de alineación
    "ligera":   dict(triangulos=800_000, paginas=10, procesos=2, poisson=9, prioridad_alta=False),
    "media":    dict(triangulos=1_500_000, paginas=20, procesos=4, poisson=10, prioridad_alta=True),
    "completa": dict(triangulos=3_000_000, paginas=48, procesos=64, poisson=11, prioridad_alta=True),
}
_cache = None


def perfil():
    """Ajustes para esta PC. Claves: nombre, ram_gb, triangulos, paginas, procesos, poisson, volumen_mb."""
    global _cache
    if _cache is not None:
        return _cache
    ram = ram_total_gb()
    nombre = os.environ.get("LEVANTA_PERFIL", "").strip().lower()
    if nombre not in _PERFILES:
        nombre = "ligera" if ram < 7 else "media" if ram < 14 else "completa"
    p = dict(_PERFILES[nombre], nombre=nombre, ram_gb=round(ram, 1))
    p["procesos"] = max(1, min(p["procesos"], os.cpu_count() or 1))
    # memoria que puede ocupar el volumen de fusión: una quinta parte de la RAM (el resto es Windows, la malla,
    # las fotos...). Con 5 GB ≈ 1 GB; con 16 GB ≈ 3.5 GB.
    p["volumen_mb"] = float(os.environ.get("LEVANTA_VOLUMEN_MB", 0)) or max(600.0, ram * 1024 * 0.22)
    _cache = p
    return p


# ---------------------------------------------------------------- tamaño del espacio y vóxel

CELDA = 0.05                 # m: rejilla con la que se mide cuánta superficie tiene el escaneo
# Medido en una fusión real (escaneo de un cuarto de 3.5 × 3.4 m, 135 cuadros, vóxel de 4 mm):
#   13,738 celdas -> 62,055 bloques reales; Open3D pide la memoria por bloque RESERVADO: 10,360 bytes cada uno
#   (8×8×8 vóxeles × 20 bytes + tabla), y la malla salió de 4.9 millones de triángulos.
# De ahí salen los tres factores de abajo.
BYTES_BLOQUE = 10_400
HOLGURA_BLOQUES = 1.2        # se reserva un 20 % más de lo estimado: si se queda corto, Open3D duplica la tabla


def celdas_ocupadas(meta, frames, dmax, min_conf, matriz_pose, paso=6, max_cuadros=1500):
    """Cuántas celdas de 5 cm toca la superficie escaneada (lectura rápida de la profundidad, 1 de cada
    `paso` píxeles). Sirve para saber el tamaño del espacio antes de reservar memoria."""
    dw, dh = int(meta["depthWidth"]), int(meta["depthHeight"])
    sel = frames if len(frames) <= max_cuadros else [frames[i] for i in
                                                     np.linspace(0, len(frames) - 1, max_cuadros).astype(int)]
    vistas = np.zeros(0, np.int64)
    pend = []
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for n, fr in enumerate(sel):
        try:
            d = np.fromfile(fr["depth"], "<u2")
        except OSError:
            continue
        if d.size != dw * dh:
            continue
        d = d.reshape(dh, dw)[::paso, ::paso].astype(np.float64) / 1000.0
        ok = (d > 0.15) & (d < dmax)
        if min_conf > 0 and os.path.exists(fr.get("conf", "")):
            c = np.fromfile(fr["conf"], np.uint8)
            if c.size == dw * dh:
                ok &= c.reshape(dh, dw)[::paso, ::paso] >= min_conf
        vv, uu = np.nonzero(ok)
        if not len(vv):
            continue
        z = d[vv, uu]
        sx, sy = dw / float(fr["w"]), dh / float(fr["h"])
        fx, fy = fr["fx"] * sx, fr["fy"] * sy
        cx, cy = (fr["cx"] + 0.5) * sx - 0.5, (fr["cy"] + 0.5) * sy - 0.5
        u, v = uu * paso, vv * paso
        pc = np.c_[(u - cx) / fx * z, -(v - cy) / fy * z, -z, np.ones_like(z)]        # cámara ARKit
        pw = (pc @ matriz_pose(fr).T)[:, :3]
        lo, hi = np.minimum(lo, pw.min(0)), np.maximum(hi, pw.max(0))
        q = np.floor(pw / CELDA).astype(np.int64) + (1 << 19)
        pend.append((q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2])
        if len(pend) >= 64 or n == len(sel) - 1:
            vistas = np.unique(np.concatenate([vistas] + pend))
            pend = []
    if pend:
        vistas = np.unique(np.concatenate([vistas] + pend))
    ext = (hi - lo) if np.isfinite(lo).all() else np.zeros(3)
    return int(len(vistas)), ext


def bloques_estimados(n_celdas, voxel, trunc_mult=5.0):
    """Bloques de 8³ vóxeles que ocupará el volumen para esa superficie con ese vóxel."""
    b = 8.0 * voxel
    capas = 1.0 + 2.0 * trunc_mult * voxel / b              # la franja del TSDF cruza varios bloques
    return n_celdas * (CELDA / b) ** 2 * capas * 0.85


def bloques_a_reservar(n_celdas, voxel, trunc_mult=5.0):
    return int(bloques_estimados(n_celdas, voxel, trunc_mult) * HOLGURA_BLOQUES) + 4000


def memoria_volumen_mb(n_celdas, voxel, trunc_mult=5.0):
    return bloques_a_reservar(n_celdas, voxel, trunc_mult) * BYTES_BLOQUE / 1e6


def triangulos_estimados(n_celdas, voxel):
    return 2.3 * n_celdas * (CELDA / voxel) ** 2


def elegir_voxel(n_celdas, voxel_pedido, triangulos_finales, trunc_mult=5.0, maximo=0.03):
    """El vóxel más fino (desde el pedido) con el que el volumen cabe en la memoria de esta PC y la malla
    no sale absurdamente más densa de lo que al final se entrega (más de 8 veces)."""
    p = perfil()
    v = float(voxel_pedido)
    while v < maximo:
        if memoria_volumen_mb(n_celdas, v, trunc_mult) <= p["volumen_mb"] and \
                triangulos_estimados(n_celdas, v) <= 8.0 * triangulos_finales:
            break
        v = round(v + 0.0005, 4)
    return v
