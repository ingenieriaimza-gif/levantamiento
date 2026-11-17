#!/usr/bin/env python3
"""
Conector (MCP) de Levantamiento 3D para Claude.

Deja que Claude revise los escaneos REALES de esta PC: ver fotos del escaneo, ver el modelo terminado comparado
contra fotos reales, leer la verificación de exactitud y volver a crear el modelo con otros ajustes en una
carpeta de pruebas (sin tocar el resultado que usas en Navisworks).

Lo arranca la app de Claude para escritorio (queda registrado solo al abrir Levantamiento 3D). Habla MCP por
la entrada/salida estándar: un mensaje JSON-RPC por renglón. No necesita nada fuera de lo ya instalado.
"""
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(AQUI, "..", "tools"))
sys.path.insert(0, AQUI)          # con -I no se agrega solo (para importar diagnostico.py)
PUERTO = 8765
NOMBRE_OK = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


def carpeta_base():
    for i, a in enumerate(sys.argv):
        if a == "--carpeta" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    cfg = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "Levantamiento3D", "config.json")
    try:
        with open(cfg, encoding="utf-8") as f:
            path = json.load(f).get("carpeta")
        if path and os.path.isdir(path):
            return path
    except (OSError, ValueError):
        pass
    return os.path.join(os.path.expanduser("~"), "Levantamientos3D")


BASE = carpeta_base()


class Fallo(Exception):
    pass


def _escaneo(nombre):
    """Carpeta de un escaneo: solo subcarpetas directas de la carpeta de escaneos."""
    if not nombre or os.path.basename(nombre) != nombre or nombre in (".", ".."):
        raise Fallo("Nombre de escaneo no válido; usa uno de los que da 'estado'.")
    d = os.path.join(BASE, nombre)
    if not os.path.isdir(d):
        raise Fallo(f"No existe el escaneo '{nombre}'.")
    return d


def _leer(path, limite=20000):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()[-limite:]
    except OSError:
        return None


def _cola(path, n):
    t = _leer(path, 200000)
    if t is None:
        return ""
    lineas = [x for x in t.splitlines() if x.strip() and not x.startswith("@@PROGRESO")]
    return "\n".join(lineas[-n:])


def _programa():
    """Estado del programa Levantamiento 3D (si está abierto)."""
    try:
        with urllib.request.urlopen(f"http://localhost:{PUERTO}/estado", timeout=2) as r:
            return json.load(r)
    except Exception:       # noqa: BLE001
        return None


def _vivo(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259                              # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


TRABAJO = os.path.join(BASE, "trabajo_prueba.json")


def _trabajo():
    try:
        with open(TRABAJO, encoding="utf-8") as f:
            t = json.load(f)
    except (OSError, ValueError):
        return None
    t["en_curso"] = _vivo(t.get("pid"))
    return t


def _n(path, ext=None):
    try:
        return sum(1 for f in os.listdir(path) if ext is None or f.lower().endswith(ext))
    except OSError:
        return 0


def _resumen(nombre):
    d = os.path.join(BASE, nombre)
    hay_capturas = os.path.exists(os.path.join(d, "frames.jsonl"))
    r = {"carpeta": nombre, "capturas": hay_capturas,
         "cuadros": _n(os.path.join(d, "frames"), ".jpg") if hay_capturas else int((_leer(os.path.join(d, "cuadros.txt")) or "0").strip() or 0),
         "fotos_12mp": _n(os.path.join(d, "fotos_hr"), ".jpg"),
         "modelo": os.path.exists(os.path.join(d, "resultado", "obj", "modelo_navisworks.fbx")),
         "verificacion": os.path.exists(os.path.join(d, "verificacion.txt")),
         "comparacion": os.path.exists(os.path.join(d, "comparacion", "comparacion.json")),
         "pruebas": sorted(os.listdir(os.path.join(d, "pruebas"))) if os.path.isdir(os.path.join(d, "pruebas")) else []}
    try:
        r["modificado"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(d)))
    except OSError:
        pass
    return r


# ---------------------------------------------------------------- herramientas

def t_estado(a):
    esc = []
    if os.path.isdir(BASE):
        for n in sorted(os.listdir(BASE)):
            if os.path.isdir(os.path.join(BASE, n)) and (
                    os.path.exists(os.path.join(BASE, n, "frames.jsonl")) or os.path.isdir(os.path.join(BASE, n, "resultado"))):
                esc.append(_resumen(n))
    esc.sort(key=lambda e: e.get("modificado", ""), reverse=True)
    p = _programa()
    ver = _leer(os.path.join(AQUI, "..", "version.json"))
    return _texto({"carpeta_de_escaneos": BASE, "version": json.loads(ver) if ver else None,
                   "programa_abierto": p is not None,
                   "programa_creando_modelo": bool(p and p.get("procesando")),
                   "programa_mensaje": (p or {}).get("mensaje"),
                   "modo_prueba": os.path.exists(os.path.join(BASE, "modo_prueba.txt")),
                   "trabajo_de_prueba": _trabajo(), "escaneos": esc[: int(a.get("max", 30))]})


def t_modo_prueba(a):
    p = os.path.join(BASE, "modo_prueba.txt")
    if a.get("activar", True):
        os.makedirs(BASE, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("Modo de prueba: al terminar un modelo NO se borran las fotos del escaneo y se guarda la comparación\n"
                    "contra fotos reales. Lo activó Claude para revisar escaneos reales. Borra este archivo para apagarlo.\n")
        return _texto("Modo de prueba ACTIVADO: los próximos escaneos conservan sus fotos (ocupan más disco) "
                      "y guardan la comparación del modelo contra fotos reales.")
    try:
        os.remove(p)
    except OSError:
        pass
    return _texto("Modo de prueba apagado: al terminar cada modelo se vuelven a borrar las fotos del escaneo.")


def t_escaneo(a):
    d = _escaneo(a.get("carpeta"))
    base = d
    if a.get("prueba"):
        if not NOMBRE_OK.match(a["prueba"]):
            raise Fallo("Nombre de prueba no válido.")
        base = os.path.join(d, "pruebas", a["prueba"])
    out = dict(_resumen(os.path.basename(d)))
    out["verificacion_txt"] = _leer(os.path.join(base, "verificacion.txt"))
    c = _leer(os.path.join(base, "comparacion", "comparacion.json"))
    out["comparacion"] = json.loads(c) if c else None
    out["medidas_iphone"] = json.loads(_leer(os.path.join(d, "medidas_iphone.json")) or "null")
    out["error"] = _leer(os.path.join(base, "resultado", "error.txt"))
    if a.get("prueba"):
        out["registro"] = _cola(os.path.join(base, "registro.txt"), int(a.get("lineas", 60)))
    res = os.path.join(base, "resultado", "obj")
    if os.path.isdir(res):
        out["archivos_resultado"] = {f: os.path.getsize(os.path.join(res, f)) for f in sorted(os.listdir(res))}
    return _texto(out)


def _imagen(path, max_lado, recorte=None):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise Fallo(f"No se pudo leer {os.path.basename(path)}")
    h, w = img.shape[:2]
    if recorte:
        x0, y0, x1, y1 = [min(1.0, max(0.0, float(v))) for v in recorte]
        img = img[int(y0 * h):max(int(y0 * h) + 8, int(y1 * h)), int(x0 * w):max(int(x0 * w) + 8, int(x1 * w))]
        h, w = img.shape[:2]
    f = max_lado / max(h, w)
    if f < 1:
        img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"type": "image", "data": base64.b64encode(enc.tobytes()).decode("ascii"), "mimeType": "image/jpeg"}


def t_imagenes(a):
    d = _escaneo(a.get("carpeta"))
    tipo = a.get("tipo", "comparacion")
    if tipo == "comparacion":
        base = d
        if a.get("prueba"):
            if not NOMBRE_OK.match(a["prueba"]):
                raise Fallo("Nombre de prueba no válido.")
            base = os.path.join(d, "pruebas", a["prueba"])
        carpeta = os.path.join(base, "comparacion")
    elif tipo == "fotos":
        carpeta = os.path.join(d, "frames")
    elif tipo == "fotos_12mp":
        carpeta = os.path.join(d, "fotos_hr")
    elif tipo == "texturas":
        base = os.path.join(d, "pruebas", a["prueba"]) if a.get("prueba") and NOMBRE_OK.match(a["prueba"]) else d
        carpeta = os.path.join(base, "resultado", "obj")
    else:
        raise Fallo("tipo debe ser comparacion, fotos, fotos_12mp o texturas")
    if not os.path.isdir(carpeta):
        raise Fallo(f"Este escaneo no tiene '{tipo}' (¿ya se borraron las fotos o falta activar el modo de prueba?).")
    archivos = sorted(f for f in os.listdir(carpeta) if f.lower().endswith(".jpg"))
    if not archivos:
        raise Fallo(f"No hay imágenes de '{tipo}'.")
    idx = a.get("indices")
    if not idx:
        import numpy as np
        idx = sorted(set(np.linspace(0, len(archivos) - 1, min(3, len(archivos))).round().astype(int).tolist()))
    idx = [int(i) for i in idx][:4]
    max_lado = int(min(2000, max(200, a.get("max_lado", 1400))))
    out = [{"type": "text", "text": f"{len(archivos)} imágenes de '{tipo}'. Se muestran: "
            + ", ".join(f"[{i}] {archivos[i]}" for i in idx if 0 <= i < len(archivos))
            + (" · en comparacion: izquierda = modelo, derecha = foto real" if tipo == "comparacion" else "")}]
    for i in idx:
        if 0 <= i < len(archivos):
            out.append(_imagen(os.path.join(carpeta, archivos[i]), max_lado, a.get("recorte")))
    return out


def t_procesar_prueba(a):
    d = _escaneo(a.get("carpeta"))
    nombre = a.get("nombre") or time.strftime("p%m%d_%H%M")
    if not NOMBRE_OK.match(nombre):
        raise Fallo("El nombre de la prueba solo puede llevar letras, números, guion y guion bajo.")
    if not os.path.exists(os.path.join(d, "frames.jsonl")):
        raise Fallo("Este escaneo ya no tiene sus fotos (se borraron al terminar el modelo). Activa el modo de prueba "
                    "y haz un escaneo nuevo.")
    t = _trabajo()
    if t and t["en_curso"]:
        raise Fallo(f"Ya hay una prueba en curso ({t['carpeta']} / {t['nombre']}); espera a que termine.")
    p = _programa()
    if p and p.get("procesando"):
        raise Fallo("El programa está creando un modelo ahora; espera a que termine para no competir por la PC.")
    sal = os.path.join(d, "pruebas", nombre)
    os.makedirs(sal, exist_ok=True)
    py = sys.executable
    if os.path.basename(py).lower() == "pythonw.exe":
        py = os.path.join(os.path.dirname(py), "python.exe")
    extra = [str(x) for x in a.get("args", [])]
    prohibidos = {"--salida", "--verificacion", "--comparar"}
    if any(x in prohibidos for x in extra):
        raise Fallo("Las carpetas de salida las pone el conector; no pases --salida, --verificacion ni --comparar.")
    cmd = [py, "-I", "-X", "utf8", os.path.join(TOOLS, "procesar_rgbd.py"), d, "--salida", os.path.join(sal, "resultado"),
           "--verificacion", os.path.join(sal, "verificacion.txt"), "--comparar", os.path.join(sal, "comparacion")]
    if not any(x.startswith("--voxel") for x in extra):
        cmd += ["--voxel", "0.004"]
    # triángulos y texturas: los decide procesar_rgbd según la RAM de la PC (tools/perfil.py), salvo que se pidan
    for k, v in (("--dmax", "3.5"), ("--conf", "2"), ("--lecturas", "2")):
        if k not in extra:
            cmd += [k, v]
    if "--avance" not in extra:
        cmd += ["--avance", os.path.join(sal, "avance")]       # si se cancela, la misma prueba continúa
    cmd += extra
    env = dict(os.environ)
    nuc = str(os.cpu_count() or 4)
    env.update(OMP_NUM_THREADS=nuc, MKL_NUM_THREADS=nuc, OPENBLAS_NUM_THREADS=nuc, NUMEXPR_NUM_THREADS=nuc)
    for k, v in (a.get("env") or {}).items():
        if not (k.startswith("LEVANTA_") or k.startswith("RGBD_")):
            raise Fallo("Solo se aceptan ajustes que empiecen con LEVANTA_ o RGBD_.")
        env[k] = str(v)
    reg = os.path.join(sal, "registro.txt")
    with open(reg, "w", encoding="utf-8") as log:
        log.write("Comando: " + " ".join(cmd[4:]) + "\nAjustes: " + json.dumps(a.get("env") or {}) + "\n")
    flags = 0x08000000 if os.name == "nt" else 0          # sin ventana
    # ejecutar_registrado deja al final del registro el código de salida y qué significa (si se cae, se sabe por qué)
    envoltura = [py, "-I", "-X", "utf8", os.path.join(AQUI, "ejecutar_registrado.py"), reg, "--"]
    proc = subprocess.Popen(envoltura + cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL, env=env, creationflags=flags)
    with open(TRABAJO, "w", encoding="utf-8") as f:
        json.dump({"pid": proc.pid, "carpeta": os.path.basename(d), "nombre": nombre,
                   "inicio": time.strftime("%Y-%m-%d %H:%M:%S"), "t0": time.time()}, f)
    return _texto(f"Prueba '{nombre}' iniciada sobre {os.path.basename(d)}. Tarda varios minutos: consulta 'trabajo' para ver "
                  f"el avance y, al terminar, 'escaneo' e 'imagenes' con prueba='{nombre}'. El resultado de Navisworks "
                  "del escaneo no se toca.")


def t_trabajo(a):
    t = _trabajo()
    if not t:
        return _texto("No se ha lanzado ninguna prueba.")
    sal = os.path.join(BASE, t["carpeta"], "pruebas", t["nombre"])
    reg = os.path.join(sal, "registro.txt")
    crudo = _leer(reg, 4000) or ""
    prog = [x for x in crudo.splitlines() if x.startswith("@@PROGRESO")]
    t["minutos"] = round((time.time() - t.get("t0", time.time())) / 60, 1) if t["en_curso"] else None
    t["ultima_etapa"] = prog[-1].split("|", 1)[-1] if prog else None
    t["termino_bien"] = (not t["en_curso"]) and os.path.exists(os.path.join(sal, "resultado", "obj", "modelo_navisworks.fbx"))
    fin = [x for x in crudo.splitlines() if x.startswith("@@FIN ")]
    if fin:
        t["como_termino"] = fin[-1][6:]
    elif not t["en_curso"]:
        t["como_termino"] = ("se cortó sin dejar código de salida (prueba lanzada antes de esta versión, "
                             "o la PC se reinició / se cerró el proceso a la fuerza)")
    if t["en_curso"]:
        from diagnostico import medir_arbol
        t["diagnostico"] = medir_arbol(t["pid"])
        try:
            t["sin_mensajes_s"] = round(time.time() - os.path.getmtime(reg))
        except OSError:
            pass
    t["registro"] = _cola(reg, int(a.get("lineas", 25)))
    return _texto(t)


def _api(ruta, datos=None):
    """Llama al programa abierto (solo responde a esta misma PC)."""
    req = urllib.request.Request(f"http://localhost:{PUERTO}{ruta}",
                                 data=None if datos is None else json.dumps(datos).encode(),
                                 headers={"Content-Type": "application/json", "X-Levanta": "1"},
                                 method="GET" if datos is None else "POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def t_proceso(a):
    try:
        r = _api("/api/proceso")
    except Exception as e:      # noqa: BLE001
        raise Fallo(f"El programa Levantamiento 3D no responde (¿está abierto? ¿es versión 51 o más?): {e}")
    n = int(a.get("lineas", 60))
    r["salida"] = r.get("salida", [])[-n:]
    return _texto(r)


def t_cancelar(a):
    que = a.get("que", "programa")
    if que == "programa":
        try:
            r = _api("/api/cancelar", {})
        except Exception as e:      # noqa: BLE001
            raise Fallo(f"El programa no responde: {e}")
        return _texto(r.get("mensaje") or r)
    t = _trabajo()
    if not t or not t["en_curso"]:
        return _texto("No hay ninguna prueba corriendo.")
    from diagnostico import matar_arbol
    matar_arbol(t["pid"])
    reg = os.path.join(BASE, t["carpeta"], "pruebas", t["nombre"], "registro.txt")
    try:
        with open(reg, "a", encoding="utf-8") as f:
            f.write("\n@@FIN codigo=cancelada | cancelada por Claude desde el conector\n")
    except OSError:
        pass
    return _texto(f"Prueba '{t['nombre']}' cancelada (se detuvieron todos sus procesos). El escaneo no se tocó.")


def t_registro(a):
    return _texto(_cola(os.path.join(BASE, "registro.txt"), int(a.get("lineas", 80))) or "(registro vacío)")


def t_borrar_prueba(a):
    d = _escaneo(a.get("carpeta"))
    if not NOMBRE_OK.match(a.get("prueba", "")):
        raise Fallo("Nombre de prueba no válido.")
    t = _trabajo()
    if t and t["en_curso"] and t["nombre"] == a["prueba"]:
        raise Fallo("Esa prueba sigue corriendo.")
    import shutil
    p = os.path.join(d, "pruebas", a["prueba"])
    if not os.path.isdir(p):
        raise Fallo("No existe esa prueba.")
    shutil.rmtree(p, ignore_errors=True)
    return _texto(f"Prueba '{a['prueba']}' borrada (solo la carpeta de pruebas; el escaneo y su resultado siguen igual).")


def _texto(x):
    return [{"type": "text", "text": x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)}]


S = lambda **p: {"type": "string", **p}
HERRAMIENTAS = [
    ("estado", "Estado de Levantamiento 3D en esta PC: versión, carpeta, si el modo de prueba está activo, la prueba "
     "en curso y la lista de escaneos (con o sin fotos conservadas, modelo, verificación, pruebas hechas).",
     {"max": {"type": "integer", "description": "máximo de escaneos (def. 30)"}}, [], t_estado),
    ("modo_prueba", "Activa o apaga el modo de prueba. Activo: al terminar un modelo NO se borran las fotos del escaneo "
     "(para poder revisarlo y repetirlo) y se guarda la comparación del modelo contra fotos reales. Ocupa más disco.",
     {"activar": {"type": "boolean"}}, ["activar"], t_modo_prueba),
    ("escaneo", "Detalle de un escaneo (o de una de sus pruebas): verificación de exactitud, comparación del modelo "
     "contra fotos reales (números), medidas del iPhone, archivos del resultado y registro de la prueba.",
     {"carpeta": S(), "prueba": S(description="nombre de una prueba de ese escaneo (opcional)"),
      "lineas": {"type": "integer"}}, ["carpeta"], t_escaneo),
    ("imagenes", "Devuelve hasta 4 imágenes de un escaneo. tipo: 'comparacion' (izquierda el modelo terminado visto "
     "desde una foto real, derecha esa foto), 'fotos' (cuadros normales), 'fotos_12mp', 'texturas' (páginas de textura "
     "del resultado). 'recorte' [x0,y0,x1,y1] en fracciones 0–1 para acercarse a un detalle.",
     {"carpeta": S(), "tipo": S(enum=["comparacion", "fotos", "fotos_12mp", "texturas"]),
      "prueba": S(description="para ver la comparación o texturas de una prueba"),
      "indices": {"type": "array", "items": {"type": "integer"}},
      "max_lado": {"type": "integer", "description": "lado mayor en píxeles (def. 1400, máx. 2000)"},
      "recorte": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
     ["carpeta"], t_imagenes),
    ("procesar_prueba", "Vuelve a crear el modelo de un escaneo que conserva sus fotos, en la carpeta pruebas/<nombre> "
     "de ese escaneo (no toca el resultado de Navisworks). 'env': ajustes LEVANTA_*/RGBD_* (p. ej. "
     "LEVANTA_LAMBDA_VECINOS, LEVANTA_AFINAR_FOTOS, LEVANTA_MM_POR_PIXEL); 'args': opciones extra de procesar_rgbd "
     "(p. ej. ['--voxel','0.003']). Corre en segundo plano; consulta 'trabajo'.",
     {"carpeta": S(), "nombre": S(), "env": {"type": "object", "additionalProperties": {"type": "string"}},
      "args": {"type": "array", "items": {"type": "string"}}}, ["carpeta"], t_procesar_prueba),
    ("trabajo", "Avance de la prueba lanzada con procesar_prueba: si sigue corriendo, etapa, minutos y últimas líneas.",
     {"lineas": {"type": "integer"}}, [], t_trabajo),
    ("proceso", "Por dentro del programa ahora: si está creando un modelo, etapa y avance, CPU/RAM de todos sus "
     "procesos, RAM libre de la PC, segundos sin mensajes (atorado) y las últimas líneas que escribió el proceso.",
     {"lineas": {"type": "integer", "description": "líneas de salida (def. 60, máx. 300)"}}, [], t_proceso),
    ("cancelar", "Detiene un proceso que salió mal. que='programa': la creación del modelo / vista foto del "
     "programa (igual que el botón Cancelar; las fotos del escaneo no se borran). que='prueba': la prueba lanzada "
     "con procesar_prueba. Pide confirmación al usuario antes de cancelar algo que él lanzó.",
     {"que": S(enum=["programa", "prueba"])}, ["que"], t_cancelar),
    ("registro", "Últimas líneas del registro del programa Levantamiento 3D (errores, conexiones del iPhone).",
     {"lineas": {"type": "integer"}}, [], t_registro),
    ("borrar_prueba", "Borra la carpeta de una prueba (pruebas/<nombre>) para liberar disco. No toca el escaneo.",
     {"carpeta": S(), "prueba": S()}, ["carpeta", "prueba"], t_borrar_prueba),
]


def atender(msg):
    metodo, mid = msg.get("method"), msg.get("id")
    if metodo == "initialize":
        pv = (msg.get("params") or {}).get("protocolVersion") or "2024-11-05"
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": pv, "capabilities": {"tools": {}},
            "serverInfo": {"name": "levantamiento3d", "version": "1"},
            "instructions": "Escaneos reales de Levantamiento 3D en esta PC. Empieza con 'estado'. Para revisar un "
                            "escaneo con sus fotos hace falta 'modo_prueba' activo ANTES de escanear."}}
    if metodo == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if metodo == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [
            {"name": n, "description": d, "inputSchema": {"type": "object", "properties": p, "required": r}}
            for n, d, p, r, _f in HERRAMIENTAS]}}
    if metodo == "tools/call":
        params = msg.get("params") or {}
        f = next((h[4] for h in HERRAMIENTAS if h[0] == params.get("name")), None)
        if f is None:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "Herramienta desconocida"}}
        try:
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": f(params.get("arguments") or {}), "isError": False}}
        except Fallo as e:
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": _texto(str(e)), "isError": True}}
        except Exception as e:      # noqa: BLE001
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": _texto(f"Error: {type(e).__name__}: {e}"),
                                                            "isError": True}}
    if mid is None:
        return None                 # notificaciones (initialized, cancelled...): sin respuesta
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Método no soportado: {metodo}"}}


def main():
    entrada = sys.stdin.buffer
    salida = sys.stdout.buffer
    sys.stdout = sys.stderr          # cualquier print accidental va a stderr, no al canal del protocolo
    for linea in iter(entrada.readline, b""):
        linea = linea.strip()
        if not linea:
            continue
        try:
            msg = json.loads(linea.decode("utf-8"))
        except ValueError:
            continue
        resp = atender(msg)
        if resp is not None:
            salida.write(json.dumps(resp, ensure_ascii=False).encode("utf-8") + b"\n")
            salida.flush()


if __name__ == "__main__":
    main()
