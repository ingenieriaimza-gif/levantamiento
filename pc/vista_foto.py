#!/usr/bin/env python3
"""
Vista foto-realista (Gaussian Splatting) entrenada en la tarjeta de video de la PC.

Usa Brush (Apache-2.0, https://github.com/ArthurBrussee/brush): un solo ejecutable que corre en
tarjetas NVIDIA, AMD e Intel. El programa lo descarga una vez (unos 30 MB) y lo guarda en
<instalación>/motor_foto. Entrena con las fotos y posiciones exactas del iPhone y arranca desde
la nube del LiDAR, así que converge rápido.

El resultado (resultado/vista_foto.ply) es solo para VER: se ve como las fotos desde cualquier
ángulo. Para medir e interferencias se sigue usando la malla.
"""
import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MOTOR_DIR = os.path.join(RAIZ, "motor_foto")
VERSION = "v0.3.0"
WINDOWS = sys.platform.startswith("win")
ARCHIVO = "brush-app-x86_64-pc-windows-msvc.zip" if WINDOWS else "brush-app-x86_64-unknown-linux-gnu.tar.xz"
URL = f"https://github.com/ArthurBrussee/brush/releases/download/{VERSION}/{ARCHIVO}"
NO_WINDOW = 0x08000000 if WINDOWS else 0

# Configuración fija (pensada para una RTX 4060 de 8 GB)
PASOS = 12000            # pasos de entrenamiento
MAX_RESOLUCION = 1600    # lado largo de las fotos al entrenar
MAX_SPLATS = 2_500_000   # tope de "manchas": cabe en 8 GB y se mueve fluido en el visor


def motor_exe():
    """Ruta del ejecutable de Brush si ya está descargado."""
    name = "brush_app.exe" if WINDOWS else "brush_app"
    for root, _dirs, files in os.walk(MOTOR_DIR):
        if name in files:
            return os.path.join(root, name)
    return None


def gpu():
    """Nombre de la tarjeta de video dedicada (NVIDIA), o None."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=10, creationflags=NO_WINDOW)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except Exception:       # noqa: BLE001
        return None


def descargar_motor(avance=None):
    """Descarga y descomprime Brush. avance(fracción) opcional. Devuelve la ruta del ejecutable."""
    exe = motor_exe()
    if exe:
        return exe
    os.makedirs(MOTOR_DIR, exist_ok=True)
    tmp = os.path.join(MOTOR_DIR, ARCHIVO + ".parte")
    req = urllib.request.Request(URL, headers={"User-Agent": "Levantamiento3D"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 18)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if avance and total:
                avance(done / total)
    if ARCHIVO.endswith(".zip"):
        with zipfile.ZipFile(tmp) as z:
            z.extractall(MOTOR_DIR)
    else:
        import tarfile
        with tarfile.open(tmp) as t:
            t.extractall(MOTOR_DIR)
    os.remove(tmp)
    exe = motor_exe()
    if not exe:
        raise RuntimeError("el paquete del motor no trae el ejecutable esperado")
    if not WINDOWS:
        os.chmod(exe, 0o755)
    return exe


def comando(exe, datos, salida_dir, nombre, pasos=PASOS, max_res=MAX_RESOLUCION, max_splats=MAX_SPLATS):
    return [exe, datos, "--total-steps", str(pasos), "--max-resolution", str(max_res),
            "--max-splats", str(max_splats), "--export-every", str(pasos),
            "--export-path", salida_dir, "--export-name", nombre]


_RE_PASO = re.compile(r"(?:Refine|Eval) iter (\d+)")


async def entrenar(sdir, on_avance, pasos=PASOS, max_res=MAX_RESOLUCION, on_linea=None):
    """
    Entrena la vista foto-realista del escaneo en `sdir` (usa sdir/splat, hecho por exportar_splat).
    on_avance(fracción 0–1, texto). Devuelve la ruta del .ply o lanza RuntimeError con el motivo.
    """
    datos = os.path.join(sdir, "splat")
    if not os.path.isdir(os.path.join(datos, "sparse", "0")):
        raise RuntimeError("faltan los datos de las fotos (carpeta splat)")
    loop = asyncio.get_running_loop()
    exe = motor_exe()
    if not exe:
        on_avance(0.0, "Descargando el motor foto-realista (una sola vez)")
        try:
            exe = await loop.run_in_executor(
                None, descargar_motor, lambda f: loop.call_soon_threadsafe(
                    on_avance, 0.02 * f, "Descargando el motor foto-realista (una sola vez)"))
        except Exception as e:      # noqa: BLE001
            raise RuntimeError(f"no se pudo descargar el motor foto-realista; conecta la PC a internet "
                               f"una vez y vuelve a intentar ({e})") from e

    res = os.path.join(sdir, "resultado")
    tmp = os.path.join(sdir, "splat_entrenando")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    env = dict(os.environ, RUST_LOG="info", NO_COLOR="1")
    cmd = comando(exe, datos, tmp, "vista_foto.ply", pasos, max_res)
    cmd += os.environ.get("LEVANTA_FOTO_EXTRA", "").split()        # solo para pruebas
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT, env=env,
                                                creationflags=NO_WINDOW)
    ultimas = []
    on_avance(0.02, "Cargando las fotos en la tarjeta de video")
    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        ultimas = (ultimas + [line])[-25:]
        if on_linea:
            on_linea(line)
        m = _RE_PASO.search(line)
        if m:
            it = int(m.group(1))
            on_avance(0.02 + 0.97 * min(1.0, it / pasos), f"Vista foto-realista: paso {it:,} de {pasos:,}")
    await proc.wait()
    ply = os.path.join(tmp, "vista_foto.ply")
    if proc.returncode != 0 or not os.path.exists(ply):
        shutil.rmtree(tmp, ignore_errors=True)
        detalle = " | ".join(ultimas[-4:])[-600:]
        raise RuntimeError(f"el entrenamiento terminó con error (código {proc.returncode}). {detalle}")
    os.makedirs(res, exist_ok=True)
    final = os.path.join(res, "vista_foto.ply")
    os.replace(ply, final)
    if not os.environ.get("LEVANTA_FOTO_EXTRA"):
        shutil.rmtree(tmp, ignore_errors=True)
    return final


if __name__ == "__main__":
    # prueba: python vista_foto.py <carpeta del escaneo> [pasos] [resolución]
    async def _main():
        t0 = time.time()
        p = await entrenar(sys.argv[1], lambda f, t: print(f"{f * 100:5.1f} %  {t}", flush=True),
                           int(sys.argv[2]) if len(sys.argv) > 2 else PASOS,
                           int(sys.argv[3]) if len(sys.argv) > 3 else MAX_RESOLUCION,
                           on_linea=lambda s: print("   ", s[:200], flush=True))
        print(f"Listo en {time.time() - t0:.0f} s: {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
    asyncio.run(_main())
