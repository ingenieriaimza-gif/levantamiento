#!/usr/bin/env python3
"""
Lanzador de Levantamiento 3D en Windows: arranca el receptor en segundo plano (sin consola)
y abre la ventana del programa. Si ya está corriendo, solo abre la ventana.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
PUERTO = 8765
URL = f"http://localhost:{PUERTO}"


def running():
    try:
        with urllib.request.urlopen(URL + "/estado", timeout=1.5) as r:
            return r.status == 200
    except Exception:       # noqa: BLE001
        return False


def open_window():
    """Ventana tipo aplicación con Edge (incluido en Windows); si no, el navegador normal."""
    candidates = [
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
        shutil.which("msedge") or "",
        os.path.join(os.environ.get("ProgramFiles", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for exe in candidates:
        if exe and os.path.exists(exe):
            subprocess.Popen([exe, f"--app={URL}", "--window-size=1400,900"], close_fds=True)
            return
    webbrowser.open(URL)


def apply_pending():
    """Aplica una actualización preparada en 'pendiente/' antes de cargar cualquier librería."""
    raiz = os.path.normpath(os.path.join(AQUI, ".."))
    pend = os.path.join(raiz, "pendiente")
    if not os.path.isdir(pend):
        return
    res = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "ok": False}
    try:
        try:
            with open(os.path.join(pend, "aplicar.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            meta = {}
        faltan = meta.get("faltan") or []
        if faltan:
            wheels = os.path.join(pend, "wheels")
            exe_dir = os.path.dirname(sys.executable)
            py = os.path.join(exe_dir, "python.exe")
            py = py if os.path.exists(py) else sys.executable
            site = os.path.join(exe_dir, "Lib", "site-packages")
            try:
                import pip  # noqa: F401
                cmd = [py, "-I", "-m", "pip"]
            except ImportError:
                pip_whl = sorted(glob.glob(os.path.join(wheels, "pip-*.whl")))
                if not pip_whl:
                    raise RuntimeError("el paquete no trae pip para instalar las librerías")
                cmd = [py, "-I", os.path.join(pip_whl[-1], "pip")]
            base = cmd + ["install", "--no-index", "--find-links", wheels, "--upgrade", "--no-warn-script-location"]
            if os.path.isdir(site):
                base += ["--target", site]
            flags = 0x08000000 if sys.platform.startswith("win") else 0
            # solo lo que falta; si sus dependencias ya están instaladas y no vienen en el paquete, sin dependencias
            for extra in ([], ["--no-deps"]):
                r = subprocess.run(base + extra + list(faltan), capture_output=True, text=True, creationflags=flags)
                print(r.stdout[-3000:], r.stderr[-3000:])
                if r.returncode == 0:
                    break
            else:
                raise RuntimeError("no se pudieron instalar las librerías (ver registro.txt)")
        backup = os.path.join(raiz, "respaldo_anterior")
        shutil.rmtree(backup, ignore_errors=True)
        os.makedirs(backup)
        for part in ("pc", "tools"):
            src = os.path.join(pend, part)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(raiz, part)
            if os.path.isdir(dst):
                shutil.copytree(dst, os.path.join(backup, part))
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        vsrc = os.path.join(pend, "version.json")
        if os.path.exists(vsrc):
            shutil.copyfile(vsrc, os.path.join(raiz, "version.json"))
        res.update(ok=True, version=meta.get("version"),
                   mensaje=f"Actualizado a la versión {meta.get('version', '')}".strip())
    except Exception as e:      # noqa: BLE001
        res["mensaje"] = f"No se aplicó la actualización: {e}. Sigues con la versión anterior."
    finally:
        shutil.rmtree(pend, ignore_errors=True)
        with open(os.path.join(raiz, "ultima_actualizacion.json"), "w", encoding="utf-8") as f:
            json.dump(res, f)
        print("Actualización:", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esperar", action="store_true", help="esperar a que se cierre la versión anterior")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir la ventana (reinicio tras actualizar)")
    ap.add_argument("--carpeta", default=os.path.join(os.path.expanduser("~"), "Levantamientos3D"))
    args = ap.parse_args()

    if args.esperar:
        for _ in range(40):
            if not running():
                break
            time.sleep(0.25)

    if running():
        open_window()
        return

    os.makedirs(args.carpeta, exist_ok=True)
    # con pythonw no hay consola: el registro va a un archivo
    log = open(os.path.join(args.carpeta, "registro.txt"), "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    print(f"\n--- inicio {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    apply_pending()

    if not args.sin_ventana:
        def wait_and_open():
            for _ in range(120):
                if running():
                    open_window()
                    return
                time.sleep(0.5)
        threading.Thread(target=wait_and_open, daemon=True).start()

    import servidor_levanta
    sys.argv = [sys.argv[0], "--puerto", str(PUERTO), "--carpeta", args.carpeta]
    servidor_levanta.main()


if __name__ == "__main__":
    main()
