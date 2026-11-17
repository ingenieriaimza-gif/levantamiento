"""
Ver por dentro y detener los procesos pesados de Levantamiento 3D (crear el modelo, vista foto, pruebas).
Sin librerías pesadas: lo usan el programa, el conector de Claude y ejecutar_registrado.py.
"""
import os
import subprocess
import sys

NO_WINDOW = 0x08000000 if sys.platform.startswith("win") else 0

# códigos de salida de Windows más comunes cuando un proceso muere sin decir nada
CODIGOS_WINDOWS = {
    3221225477: "acceso inválido a memoria (0xC0000005): falla interna de una librería (Open3D/numpy)",
    3221225725: "desbordamiento de pila (0xC00000FD)",
    3221226505: "falla grave detectada por el propio programa (0xC0000409)",
    3221225495: "sin memoria virtual suficiente (0xC0000017): la PC se quedó sin RAM",
    3221225786: "se cerró con Ctrl+C / se cerró la ventana (0xC000013A)",
    1: "error de Python (ver las últimas líneas del registro)",
}


def explicar_codigo(code):
    if code is None:
        return "sin código"
    c = code & 0xFFFFFFFF if code < 0 else code
    return f"código {code}" + (f" — {CODIGOS_WINDOWS[c]}" if c in CODIGOS_WINDOWS else "")


def matar_arbol(pid):
    """Termina un proceso y TODOS sus hijos (procesar_rgbd abre varios procesos para trabajar en paralelo)."""
    if not pid:
        return
    try:
        import psutil
        try:
            raiz = psutil.Process(int(pid))
            hijos = raiz.children(recursive=True)
        except psutil.Error:
            return
        for p in hijos + [raiz]:
            try:
                p.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(hijos + [raiz], timeout=5)
        return
    except ImportError:
        pass
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, creationflags=NO_WINDOW)
    else:
        try:
            os.kill(int(pid), 9)
        except OSError:
            pass


_PROCS = {}      # psutil.Process reutilizados: cpu_percent necesita dos lecturas del mismo objeto
_FALLOS = {}     # pid raíz -> (hora, fallos de página acumulados), para calcular fallos por segundo


def medir_arbol(pid):
    """CPU, memoria y número de procesos del trabajo en curso (para ver si sigue vivo o se atoró)."""
    try:
        import psutil
        raiz = psutil.Process(int(pid))
        procs = []
        for p in [raiz] + raiz.children(recursive=True):
            procs.append(_PROCS.setdefault((p.pid, p.create_time()), p))
        vivos = {(p.pid, p.create_time()) for p in procs}
        for k in list(_PROCS):
            if k not in vivos:
                del _PROCS[k]
        ram = cpu = privada = fallos = 0.0
        for p in procs:
            try:
                m = p.memory_info()
                ram += m.rss
                privada += getattr(m, "private", 0) or 0       # lo que pidió (en RAM o mandado al disco)
                fallos += getattr(m, "num_page_faults", 0) or 0
                cpu += p.cpu_percent(None)
            except psutil.Error:
                pass
        import time
        ahora = time.time()
        antes = _FALLOS.get(int(pid))
        _FALLOS[int(pid)] = (ahora, fallos)
        fps = None
        if antes and ahora > antes[0] and fallos >= antes[1]:
            fps = round((fallos - antes[1]) / (ahora - antes[0]))
        vm = psutil.virtual_memory()
        out = {"procesos": len(procs), "ram_mb": round(ram / 1e6), "cpu_pct": round(cpu),
               "ram_libre_mb": round(vm.available / 1e6), "ram_total_mb": round(vm.total / 1e6)}
        if privada:
            out["privada_mb"] = round(privada / 1e6)
            # más pedido que lo que cabe en RAM: Windows manda memoria al disco y todo se vuelve lento
            out["paginando"] = bool(privada > ram * 1.3 and vm.available < 0.1 * vm.total)
        if fps is not None:
            out["fallos_pag_s"] = fps
        return out
    except Exception:       # noqa: BLE001 — sin psutil o el proceso ya terminó
        return None
