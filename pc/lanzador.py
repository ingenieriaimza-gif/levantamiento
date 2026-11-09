#!/usr/bin/env python3
"""
Lanzador de Levantamiento 3D en Windows: arranca el receptor en segundo plano (sin consola)
y abre la ventana del programa. Si ya está corriendo, solo abre la ventana.
"""
import argparse
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
