"""
Funciones de la ventana del programa: lista de escaneos, abrir carpetas, actualizaciones
(del programa de la PC y de la app del iPhone) y cierre.

Las actualizaciones vienen de las versiones publicadas en GitHub:
  pc-<n>     Levantamiento3D_Setup.exe, actualizacion_pc.zip, version.json
  build-<n>  Levanta.ipa (app del iPhone)
"""
import asyncio
import hashlib
import importlib.metadata
import io
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime

from aiohttp import web

REPO = "ingenieriaimza-gif/levantamiento"
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.normpath(os.path.join(AQUI, ".."))          # carpeta de instalación
VERSION_FILE = os.path.join(RAIZ, "version.json")
UA = {"User-Agent": "Levantamiento3D-PC", "Accept": "application/vnd.github+json"}
PENDIENTE = os.path.join(RAIZ, "pendiente")              # actualización lista para aplicarse al reiniciar
ULTIMA = os.path.join(RAIZ, "ultima_actualizacion.json")
BUNDLE_ID = "com.levanta.lidar"


# ------------------------------------------------------------------ requisitos

def parse_requirements(text):
    """Líneas tipo 'paquete==1.2', 'paquete>=1.2' o 'paquete'. Devuelve [(nombre, op, versión, línea)]."""
    out = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=)?\s*([^;\s]*)", line)
        if m:
            out.append((m.group(1), m.group(2) or "", m.group(3) or "", line))
    return out


def _vt(v):
    parts = []
    for p in v.split("."):
        m = re.match(r"\d+", p)
        if not m:
            break
        parts.append(int(m.group()))
    return tuple(parts)


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def satisfies(installed, op, ver):
    if installed is None:
        return False
    if op == "==":
        return _vt(installed) == _vt(ver)
    if op == ">=":
        return _vt(installed) >= _vt(ver)
    return True


def _norm(name):
    return re.sub(r"[-_.]+", "_", name).lower()


def ipa_info(path):
    """Lee nombre, versión y build de una .ipa (Payload/*.app/Info.plist)."""
    with zipfile.ZipFile(path) as z:
        plists = [n for n in z.namelist() if re.match(r"^Payload/[^/]+\.app/Info\.plist$", n)]
        if not plists:
            raise ValueError("No es una app de iPhone (.ipa) válida")
        info = plistlib.loads(z.read(plists[0]))
    return {"bundle": info.get("CFBundleIdentifier", ""),
            "nombre": info.get("CFBundleDisplayName") or info.get("CFBundleName", ""),
            "version": info.get("CFBundleShortVersionString", ""),
            "build": int(re.sub(r"\D", "", str(info.get("CFBundleVersion", "0"))) or 0)}


CONFIG_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "Levantamiento3D")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


def load_config():
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)


def default_folder():
    return load_config().get("carpeta") or os.path.join(os.path.expanduser("~"), "Levantamientos3D")


def pick_folder_dialog(initial):
    """Selector de carpetas de Windows (sin librerías extra: usa PowerShell y Windows Forms)."""
    if not sys.platform.startswith("win"):
        return None
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$o = New-Object System.Windows.Forms.Form; $o.TopMost = $true; $o.ShowInTaskbar = $false;"
        "$f = New-Object System.Windows.Forms.FolderBrowserDialog;"
        "$f.Description = 'Carpeta donde guardar los escaneos de Levantamiento 3D';"
        "$f.ShowNewFolderButton = $true;"
        f"$f.SelectedPath = '{initial.replace(chr(39), chr(39) * 2)}';"
        "if ($f.ShowDialog($o) -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($f.SelectedPath) }"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    path = r.stdout.strip()
    return path or None


# ------------------------------------------------------------------ IA (módulo opcional)

IA_DIR = os.path.join(RAIZ, "ia")
IA_PAQUETES = os.path.join(IA_DIR, "paquetes")
IA_MODELOS = os.path.join(IA_DIR, "modelos")
IA_MODELO = "depth-anything/prompt-depth-anything-vits-hf"
IA_TORCH = "2.6.0"
IA_TRANSFORMERS = "5.22.2"


def gpu_nvidia():
    """Nombre de la tarjeta NVIDIA o None."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=10, creationflags=0x08000000 if sys.platform.startswith("win") else 0)
        name = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
        return name
    except Exception:       # noqa: BLE001
        return None


def ia_instalada():
    modelo = os.path.join(IA_MODELOS, "models--" + IA_MODELO.replace("/", "--"))
    return (os.path.isdir(os.path.join(IA_PAQUETES, "torch")) and
            os.path.isdir(os.path.join(IA_PAQUETES, "transformers")) and os.path.isdir(modelo))


def ia_config():
    # configuración fija: la IA no mejoró la exactitud de los tubos en las pruebas, así que no se usa
    return {"final": False, "vivo": False}


def local_version():
    try:
        with open(VERSION_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"pc": 0, "req": ""}


def requirements_hash():
    try:
        with open(os.path.join(AQUI, "requirements.txt"), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return ""


def _get_json(url, timeout=15):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _download(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def latest_releases():
    """Última versión del programa de la PC (pc-N) y de la app del iPhone (build-N)."""
    rels = _get_json(f"https://api.github.com/repos/{REPO}/releases?per_page=40")
    best = {"pc": None, "build": None}
    for r in rels:
        tag = r.get("tag_name", "")
        for prefix in ("pc", "build"):
            if tag.startswith(prefix + "-"):
                try:
                    n = int(tag.split("-", 1)[1])
                except ValueError:
                    continue
                if best[prefix] is None or n > best[prefix][0]:
                    best[prefix] = (n, r)
    return best


def _asset(release, name):
    for a in release.get("assets", []):
        if a["name"] == name:
            return a["browser_download_url"]
    return None


def open_path(path):
    if sys.platform.startswith("win"):
        os.startfile(path)                         # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def find_sideloadly():
    for base in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles", ""),
                 os.environ.get("ProgramFiles(x86)", "")):
        cand = os.path.join(base, "Sideloadly", "sideloadly.exe")
        if base and os.path.exists(cand):
            return cand
    return None


def reveal(path):
    """Abre el Explorador con el archivo seleccionado (Windows) o la carpeta."""
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        open_path(os.path.dirname(path))


def downloads_dir():
    d = os.path.join(os.path.expanduser("~"), "Downloads")
    return d if os.path.isdir(d) else os.path.expanduser("~")


class Gestion:
    def __init__(self, server, root):
        self.server = server
        self.root = root
        self.busy = None

    # ---------------------------------------------------------- carpeta de escaneos

    def _set_root(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".prueba_escritura")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        self.root = path
        self.server.root = path
        cfg = load_config()
        cfg["carpeta"] = path
        save_config(cfg)
        self.publish_app_status()
        return path

    async def carpeta_elegir(self, request):
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(None, pick_folder_dialog, self.root)
        if not path:
            return web.json_response({"ok": False, "mensaje": "No se eligió ninguna carpeta"})
        return await self._carpeta_aplicar(path)

    async def carpeta_fijar(self, request):
        data = await request.json()
        return await self._carpeta_aplicar(str(data.get("ruta", "")).strip().strip('"'))

    async def _carpeta_aplicar(self, path):
        if not path:
            return web.json_response({"ok": False, "mensaje": "Escribe una carpeta"})
        if self.server.session is not None:
            return web.json_response({"ok": False, "mensaje": "Termina el escaneo en curso antes de cambiar la carpeta"})
        try:
            p = self._set_root(path)
        except OSError as e:
            return web.json_response({"ok": False, "mensaje": f"No se puede usar esa carpeta: {e}"})
        return web.json_response({"ok": True, "carpeta": p,
                                  "mensaje": f"Los escaneos nuevos se guardarán en {p}"})

    # ---------------------------------------------------------- escaneos

    async def escaneos(self, request):
        items = []
        if os.path.isdir(self.root):
            for name in sorted(os.listdir(self.root), reverse=True):
                d = os.path.join(self.root, name)
                if not (os.path.isdir(d) and name.startswith("RGBD_")):
                    continue
                res = os.path.join(d, "resultado")
                files = []
                if os.path.isdir(res):
                    for fn in ("modelo_texturizado.usdz", "nube_fusionada.e57", "malla_solidworks.stl",
                               "nube_navisworks.pts", "modelo_texturizado.glb", "reporte.txt"):
                        p = os.path.join(res, fn)
                        if os.path.exists(p):
                            files.append({"nombre": fn, "mb": round(os.path.getsize(p) / 1e6, 1)})
                frames = 0
                try:
                    with open(os.path.join(d, "frames.jsonl"), encoding="utf-8") as f:
                        frames = sum(1 for line in f if line.strip())
                except OSError:
                    pass
                live = self.server.session is not None and os.path.normpath(self.server.session.dir) == os.path.normpath(d)
                processing = (self.server.status.get("procesando") and
                              os.path.normpath(getattr(self.server, "processing_dir", "") or "") == os.path.normpath(d))
                items.append({
                    "en_curso": live,
                    "iphone_conectado": self.server.phone is not None,
                    "procesando": bool(processing),
                    "con_modelo": any(f["nombre"] == "malla_solidworks.stl" for f in files),
                    "nombre": name.replace("RGBD_", ""),
                    "carpeta": name,
                    "fecha": datetime.fromtimestamp(os.path.getmtime(d)).strftime("%Y-%m-%d %H:%M"),
                    "cuadros": frames,
                    "procesado": os.path.isdir(res),
                    "splat": os.path.isdir(os.path.join(d, "splat")),
                    "descartado": name.endswith("_descartado"),
                    "archivos": files,
                })
        return web.json_response({"carpeta": self.root, "escaneos": items})

    # ---------- para el iPhone: modelos terminados (solo lectura en la red local)

    async def tel_modelos(self, request):
        items = []
        if os.path.isdir(self.root):
            live = self.server.session.dir if self.server.session is not None else None
            for name in sorted(os.listdir(self.root), reverse=True):
                d = os.path.join(self.root, name)
                p = os.path.join(d, "resultado", "modelo_texturizado.usdz")
                if not (name.startswith("RGBD_") and os.path.exists(p)):
                    continue
                if live and os.path.normpath(live) == os.path.normpath(d):
                    continue
                st = os.stat(p)
                items.append({"carpeta": name, "nombre": name.replace("RGBD_", ""),
                              "fecha": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                              "version": int(st.st_mtime), "mb": round(st.st_size / 1e6, 1)})
                if len(items) >= 20:
                    break
        return web.json_response({"modelos": items})

    async def tel_modelo(self, request):
        d = self._safe(request.match_info["carpeta"])
        p = os.path.join(d, "resultado", "modelo_texturizado.usdz")
        if not os.path.exists(p):
            raise web.HTTPNotFound(text="Este escaneo no tiene modelo terminado")
        return web.FileResponse(p, headers={"Content-Type": "model/vnd.usdz+zip"})

    async def tel_medidas(self, request):
        d = self._safe(request.match_info["carpeta"])
        try:
            with open(os.path.join(d, "medidas_iphone.json"), encoding="utf-8") as f:
                return web.json_response(json.load(f))
        except (OSError, ValueError):
            return web.json_response({"medidas": []})

    def _safe(self, rel):
        p = os.path.realpath(os.path.join(self.root, rel or ""))
        root = os.path.realpath(self.root)
        if p != root and not p.startswith(root + os.sep):
            raise web.HTTPForbidden(text="Ruta fuera de la carpeta de escaneos")
        if not os.path.exists(p):
            raise web.HTTPNotFound(text="No existe")
        return p

    async def abrir(self, request):
        data = await request.json()
        p = self._safe(data.get("ruta", ""))
        if data.get("mostrar") and sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", p])      # carpeta con el archivo seleccionado
        else:
            open_path(p)
        return web.json_response({"ok": True})

    async def reprocesar(self, request):
        data = await request.json()
        d = self._safe(data.get("carpeta", ""))
        if self.server.status.get("procesando"):
            return web.json_response({"ok": False, "mensaje": "Ya se está creando otro modelo; espera a que termine"})
        sess = self.server.session
        if sess is not None and os.path.normpath(sess.dir) == os.path.normpath(d):
            if self.server.phone is not None:
                return web.json_response({"ok": False, "mensaje": "El iPhone sigue conectado: termínalo desde el iPhone"})
            # el iPhone se desconectó sin terminar: se cierra aquí y se crea el modelo con lo recibido
            self.server.session = None
            await asyncio.get_running_loop().run_in_executor(self.server.pool, sess.close)
        escala = data.get("escala")
        try:
            escala = float(escala) if escala else None
        except (TypeError, ValueError):
            escala = None
        asyncio.create_task(self.server.final_processing(d, escala=escala))
        return web.json_response({"ok": True})

    # ---------------------------------------------------------- actualizaciones

    async def version(self, request):
        v = local_version()
        return web.json_response({"pc": v.get("pc", 0), "req": v.get("req", ""), "repo": REPO})

    async def buscar(self, request):
        loop = asyncio.get_running_loop()
        try:
            best = await loop.run_in_executor(None, latest_releases)
        except Exception as e:      # noqa: BLE001
            return web.json_response({"ok": False, "mensaje": f"No se pudo consultar GitHub: {e}"})
        local = local_version()
        out = {"ok": True, "pc_local": local.get("pc", 0)}
        if best["pc"]:
            n, rel = best["pc"]
            remote_req = ""
            vurl = _asset(rel, "version.json")
            if vurl:
                try:
                    remote = json.loads(await loop.run_in_executor(None, _download, vurl))
                    remote_req = remote.get("req", "")
                except Exception:   # noqa: BLE001
                    pass
            out.update(pc_disponible=n, pc_fecha=rel.get("published_at", "")[:10],
                       pc_instalador=bool(remote_req and remote_req != local.get("req", "")))
        if best["build"]:
            n, rel = best["build"]
            out.update(app_build=n, app_fecha=rel.get("published_at", "")[:10],
                       app_url=_asset(rel, "Levanta.ipa"))
        return web.json_response(out)

    async def actualizar_pc(self, request):
        if self.busy:
            return web.json_response({"ok": False, "mensaje": "Ya hay una actualización en curso"})
        self.busy = "pc"
        loop = asyncio.get_running_loop()
        try:
            best = await loop.run_in_executor(None, latest_releases)
            if not best["pc"]:
                return web.json_response({"ok": False, "mensaje": "No hay versiones publicadas"})
            n, rel = best["pc"]
            vurl = _asset(rel, "version.json")
            remote = json.loads(await loop.run_in_executor(None, _download, vurl)) if vurl else {}
            local = local_version()
            if remote.get("req") and remote.get("req") != local.get("req"):
                # cambiaron las librerías: hace falta el instalador completo
                url = _asset(rel, "Levantamiento3D_Setup.exe")
                data = await loop.run_in_executor(None, _download, url, 1800)
                dest = os.path.join(downloads_dir(), f"Levantamiento3D_Setup_{n}.exe")
                with open(dest, "wb") as f:
                    f.write(data)
                subprocess.Popen([dest], close_fds=True)
                asyncio.get_running_loop().call_later(1.5, os._exit, 0)
                return web.json_response({"ok": True, "mensaje": "Se abrió el instalador de la versión nueva."})
            url = _asset(rel, "actualizacion_pc.zip")
            if not url:
                return web.json_response({"ok": False, "mensaje": "La versión publicada no trae actualización"})
            data = await loop.run_in_executor(None, _download, url)
            await loop.run_in_executor(None, self._apply_code_update, data, remote or {"pc": n})
            asyncio.get_running_loop().call_later(1.0, self._restart)
            return web.json_response({"ok": True, "mensaje": f"Actualizado a la versión {n}. Reiniciando…"})
        except Exception as e:      # noqa: BLE001
            return web.json_response({"ok": False, "mensaje": f"No se pudo actualizar: {e}"})
        finally:
            self.busy = None

    def _apply_code_update(self, data, remote):
        """Reemplaza las carpetas pc/ y tools/ con las del zip (con respaldo por si algo falla)."""
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for member in z.namelist():
                if member.startswith("/") or ".." in member.split("/"):
                    raise ValueError(f"Ruta no permitida en la actualización: {member}")
            stage = tempfile.mkdtemp(prefix="act_", dir=RAIZ)
            z.extractall(stage)
        backup = os.path.join(RAIZ, "respaldo_anterior")
        shutil.rmtree(backup, ignore_errors=True)
        os.makedirs(backup)
        for part in ("pc", "tools"):
            src = os.path.join(stage, part)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(RAIZ, part)
            if os.path.isdir(dst):
                shutil.copytree(dst, os.path.join(backup, part))
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        with open(VERSION_FILE, "w", encoding="utf-8") as f:
            json.dump(remote, f)
        shutil.rmtree(stage, ignore_errors=True)

    def _restart(self):
        exe = sys.executable
        lanzador = os.path.join(AQUI, "lanzador.py")
        flags = 0x08000000 if sys.platform.startswith("win") else 0      # sin ventana de consola
        subprocess.Popen([exe, "-I", lanzador, "--esperar", "--sin-ventana", "--carpeta", self.root],
                         close_fds=True, creationflags=flags)
        os._exit(0)

    async def descargar_app(self, request):
        loop = asyncio.get_running_loop()
        try:
            best = await loop.run_in_executor(None, latest_releases)
            n, rel = best["build"]
            url = _asset(rel, "Levanta.ipa")
            data = await loop.run_in_executor(None, _download, url)
            dest = os.path.join(downloads_dir(), f"Levanta_build{n}.ipa")
            with open(dest, "wb") as f:
                f.write(data)
            await loop.run_in_executor(None, self._store_app, dest)
            sideloadly = None
            for base in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles", ""),
                         os.environ.get("ProgramFiles(x86)", "")):
                cand = os.path.join(base, "Sideloadly", "sideloadly.exe")
                if base and os.path.exists(cand):
                    sideloadly = cand
                    break
            if sideloadly:
                subprocess.Popen([sideloadly], close_fds=True)
            open_path(downloads_dir())
            return web.json_response({"ok": True, "ruta": dest, "sideloadly": bool(sideloadly),
                                      "mensaje": f"Descargada en {dest}"})
        except Exception as e:      # noqa: BLE001
            return web.json_response({"ok": False, "mensaje": f"No se pudo descargar: {e}"})

    async def salir(self, request):
        asyncio.get_running_loop().call_later(0.5, os._exit, 0)
        return web.json_response({"ok": True})

    # ---------------------------------------------------------- modelo y medidas de un escaneo

    async def modelo(self, request):
        d = self._safe(request.match_info["carpeta"])
        p = os.path.join(d, "resultado", "modelo_texturizado.glb")
        if not os.path.exists(p):
            raise web.HTTPNotFound(text="Este escaneo no tiene modelo procesado")
        return web.FileResponse(p)

    async def medidas_get(self, request):
        d = self._safe(request.match_info["carpeta"])
        out = {"escala": 1.0, "medidas": []}
        for fn in ("medidas.json", "medidas_iphone.json"):
            try:
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    data = json.load(f)
                if fn == "medidas.json":
                    out.update(data)
                else:
                    known = {m.get("id") for m in out["medidas"]}
                    out["medidas"] += [m for m in data.get("medidas", []) if m.get("id") not in known]
            except (OSError, ValueError):
                pass
        return web.json_response(out)

    async def medidas_set(self, request):
        d = self._safe(request.match_info["carpeta"])
        data = await request.json()
        clean = {"escala": float(data.get("escala", 1.0)), "medidas": []}
        for m in data.get("medidas", [])[:500]:
            try:
                clean["medidas"].append({
                    "id": str(m["id"])[:40], "nombre": str(m.get("nombre", ""))[:60],
                    "a": [float(x) for x in m["a"]][:3], "b": [float(x) for x in m["b"]][:3],
                    "origen": str(m.get("origen", "pc"))[:10],
                    "real": float(m["real"]) if m.get("real") not in (None, "") else None,
                    "calibra": bool(m.get("calibra", False))})
            except (KeyError, TypeError, ValueError):
                continue
        with open(os.path.join(d, "medidas.json"), "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=1)
        return web.json_response({"ok": True})

    # ---------------------------------------------------------- IA

    async def ia_estado(self, request):
        loop = asyncio.get_running_loop()
        gpu = await loop.run_in_executor(None, gpu_nvidia)
        return web.json_response({"instalada": ia_instalada(), "gpu": gpu, "instalando": self.busy == "ia",
                                  **ia_config()})

    async def ia_config_set(self, request):
        data = await request.json()
        cfg = load_config()
        if "final" in data:
            cfg["ia_final"] = bool(data["final"])
        if "vivo" in data:
            cfg["ia_vivo"] = bool(data["vivo"])
        save_config(cfg)
        return web.json_response({"ok": True, **ia_config()})

    async def ia_instalar(self, request):
        if self.busy:
            return web.json_response({"ok": False, "mensaje": "Hay otra instalación en curso"})
        self.busy = "ia"
        asyncio.create_task(self._ia_instalar())
        return web.json_response({"ok": True, "mensaje": "Instalando la IA. El avance se ve en el registro (Visor)."})

    async def _ia_instalar(self):
        srv = self.server
        from servidor_levanta import PYTHON, NO_WINDOW, TOOLS
        try:
            loop = asyncio.get_running_loop()
            gpu = await loop.run_in_executor(None, gpu_nvidia)
            os.makedirs(IA_PAQUETES, exist_ok=True)
            os.makedirs(IA_MODELOS, exist_ok=True)
            pip = [PYTHON, "-I", "-m", "pip", "install", "--target", IA_PAQUETES, "--upgrade",
                   "--no-warn-script-location", "--disable-pip-version-check", "--progress-bar", "off"]
            if gpu:
                torch_cmd = pip + [f"torch=={IA_TORCH}", "--index-url", "https://download.pytorch.org/whl/cu124"]
                await srv.send_status(f"IA: tarjeta {gpu} encontrada. Descargando PyTorch para NVIDIA (~2.5 GB)…")
            else:
                torch_cmd = pip + [f"torch=={IA_TORCH}"]
                await srv.send_status("IA: no hay tarjeta NVIDIA; se instala la versión para procesador (más lenta).")
            pasos = [
                ("PyTorch", torch_cmd),
                ("Transformers", pip + [f"transformers=={IA_TRANSFORMERS}", "huggingface_hub"]),
                ("modelo Prompt Depth Anything", [PYTHON, "-I", "-c",
                    "import sys; sys.path.append(r'%s'); from huggingface_hub import snapshot_download; "
                    "p = snapshot_download('%s', cache_dir=r'%s'); print('Modelo en', p)"
                    % (IA_PAQUETES, IA_MODELO, IA_MODELOS)]),
                ("prueba", [PYTHON, "-I", "-c",
                    "import sys; sys.path.insert(0, r'%s'); import ia_profundidad as I; "
                    "import torch; print('IA lista:', I.disponible(), '| GPU:', torch.cuda.is_available())" % TOOLS]),
            ]
            for nombre, cmd in pasos:
                await srv.send_status(f"IA: instalando {nombre}…")
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.STDOUT, creationflags=NO_WINDOW)
                last = ""
                async for line in proc.stdout:
                    t = line.decode("utf-8", "replace").strip()
                    if t and (t.startswith(("Successfully", "Modelo en", "IA lista", "ERROR", "Downloading torch"))):
                        await srv.send_status("IA: " + t[:200])
                    last = t or last
                await proc.wait()
                if proc.returncode != 0:
                    await srv.send_status(f"IA: falló la instalación de {nombre}: {last[:200]}")
                    return
            await srv.send_status("IA instalada. Se usará al procesar los escaneos (puedes cambiarlo en Actualizaciones).")
        except Exception as e:      # noqa: BLE001
            await srv.send_status(f"IA: no se pudo instalar: {e}")
        finally:
            self.busy = None

    # ---------------------------------------------------------- actualizar desde un archivo

    def _incoming(self):
        d = os.path.join(self.root, "_entrante")
        os.makedirs(d, exist_ok=True)
        return d

    async def archivo_validar(self, request):
        """Recibe el archivo elegido (se guarda por partes, sin límite de tamaño) y dice qué es y qué falta."""
        reader = await request.multipart()
        field = await reader.next()
        if field is None or not field.filename:
            return web.json_response({"ok": False, "mensaje": "No llegó ningún archivo"})
        name = re.sub(r"[^A-Za-z0-9_.\-]", "_", os.path.basename(field.filename))[:120]
        dest = os.path.join(self._incoming(), name)
        with open(dest, "wb") as f:
            while True:
                chunk = await field.read_chunk(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        loop = asyncio.get_running_loop()
        try:
            rep = await loop.run_in_executor(None, self._analizar, dest)
        except Exception as e:      # noqa: BLE001
            rep = {"ok": False, "mensaje": f"No se pudo leer el archivo: {e}"}
        rep["token"] = name
        return web.json_response(rep)

    def _analizar(self, path):
        low = path.lower()
        size_mb = round(os.path.getsize(path) / 1e6, 1)
        if low.endswith(".ipa"):
            info = ipa_info(path)
            if info["bundle"] != BUNDLE_ID:
                return {"ok": False, "mensaje": f"Esta app ({info['bundle']}) no es la app 3D."}
            actual = self._stored_app()
            return {"ok": True, "tipo": "ipa", "mb": size_mb, **info,
                    "actual_build": actual.get("build", 0) if actual else 0,
                    "resumen": f"App del iPhone {info['nombre']} {info['version']} (build {info['build']})",
                    "puede": True}
        if low.endswith(".exe"):
            with open(path, "rb") as f:
                if f.read(2) != b"MZ":
                    return {"ok": False, "mensaje": "El archivo no es un instalador de Windows válido"}
            return {"ok": True, "tipo": "exe", "mb": size_mb, "puede": True,
                    "resumen": "Instalador completo del programa: se cerrará el programa y se abrirá el instalador."}
        if not zipfile.is_zipfile(path):
            return {"ok": False, "mensaje": "Formato no reconocido. Usa actualizacion_pc.zip, "
                                            "Levantamiento3D_Setup.exe o Levanta.ipa"}
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            for n in names:
                if n.startswith("/") or ".." in n.replace("\\", "/").split("/"):
                    return {"ok": False, "mensaje": "El paquete trae rutas no permitidas"}
            prefix = next((n[: -len("pc/servidor_levanta.py")] for n in names
                           if n.endswith("pc/servidor_levanta.py")), None)
            if prefix is None or (prefix + "tools/procesar_rgbd.py") not in names:
                return {"ok": False, "mensaje": "El zip no es una actualización del programa (faltan pc/ o tools/)"}
            remote = {}
            if prefix + "version.json" in names:
                try:
                    remote = json.loads(z.read(prefix + "version.json"))
                except ValueError:
                    remote = {}
            reqs = parse_requirements(z.read(prefix + "pc/requirements.txt").decode("utf-8", "replace")) \
                if prefix + "pc/requirements.txt" in names else []
            wheels = [os.path.basename(n) for n in names if n.startswith(prefix + "wheels/") and n.endswith(".whl")]
        wheel_names = {_norm(w.split("-")[0]) for w in wheels}
        checks, faltan, sin_rueda = [], [], []
        for name, op, ver, line in reqs:
            inst = installed_version(name)
            ok = satisfies(inst, op, ver)
            checks.append({"paquete": name, "requiere": f"{op}{ver}" if op else "cualquiera",
                           "instalado": inst or "no instalado", "ok": ok})
            if not ok:
                faltan.append(line)
                if _norm(name) not in wheel_names:
                    sin_rueda.append(name)
        tiene_pip = "pip" in wheel_names or installed_version("pip") is not None
        puede = not faltan or (not sin_rueda and tiene_pip)
        local = local_version()
        motivo = ""
        if faltan and sin_rueda:
            motivo = ("Faltan librerías que este paquete no incluye: " + ", ".join(sin_rueda) +
                      ". Usa el instalador completo (Levantamiento3D_Setup.exe).")
        elif faltan and not tiene_pip:
            motivo = "El paquete trae librerías nuevas pero no el instalador de paquetes (pip)."
        return {"ok": True, "tipo": "pc", "mb": size_mb, "puede": puede, "motivo": motivo,
                "version_actual": local.get("pc", 0), "version_nueva": remote.get("pc", "desconocida"),
                "requisitos": checks, "faltan": faltan, "ruedas": len(wheels),
                "resumen": (f"Programa de la PC versión {remote.get('pc', '?')}"
                            + (f" · instalará {len(faltan)} librería(s) desde el paquete" if faltan and puede else ""))}

    async def archivo_instalar(self, request):
        data = await request.json()
        name = os.path.basename(str(data.get("token", "")))
        path = os.path.join(self._incoming(), name)
        if not name or not os.path.exists(path):
            return web.json_response({"ok": False, "mensaje": "Vuelve a elegir el archivo"})
        loop = asyncio.get_running_loop()
        rep = await loop.run_in_executor(None, self._analizar, path)
        if not rep.get("ok") or not rep.get("puede"):
            return web.json_response({"ok": False, "mensaje": rep.get("motivo") or rep.get("mensaje", "No se puede instalar")})
        if rep["tipo"] == "ipa":
            await loop.run_in_executor(None, self._store_app, path)
            return web.json_response({"ok": True, "mensaje": f"Listo: {rep['resumen']} guardada. "
                                      "El iPhone verá el aviso al conectarse al programa."})
        if rep["tipo"] == "exe":
            subprocess.Popen([path], close_fds=True)
            loop.call_later(1.5, os._exit, 0)
            return web.json_response({"ok": True, "mensaje": "Se abrió el instalador. El programa se cerrará."})
        await loop.run_in_executor(None, self._stage_zip, path, rep)
        loop.call_later(1.0, self._restart)
        return web.json_response({"ok": True, "reinicia": True,
                                  "mensaje": "Actualización preparada. El programa se reinicia para aplicarla…"})

    def _stage_zip(self, path, rep):
        """Deja la actualización en 'pendiente/'; el lanzador la aplica al reiniciar
        (así ninguna librería está en uso mientras se reemplaza)."""
        shutil.rmtree(PENDIENTE, ignore_errors=True)
        stage = PENDIENTE + "_tmp"
        shutil.rmtree(stage, ignore_errors=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(stage)
        names = os.listdir(stage)
        root = stage
        if "pc" not in names and len(names) == 1 and os.path.isdir(os.path.join(stage, names[0])):
            root = os.path.join(stage, names[0])
        with open(os.path.join(root, "aplicar.json"), "w", encoding="utf-8") as f:
            json.dump({"faltan": rep.get("faltan", []), "version": rep.get("version_nueva")}, f)
        os.replace(root, PENDIENTE)
        shutil.rmtree(stage, ignore_errors=True)
        try:
            os.remove(path)
        except OSError:
            pass

    async def ultima(self, request):
        try:
            with open(ULTIMA, encoding="utf-8") as f:
                return web.json_response(json.load(f))
        except (OSError, ValueError):
            return web.json_response({})

    # ---------------------------------------------------------- app del iPhone guardada en la PC

    def _app_dir(self):
        d = os.path.join(self.root, "app_iphone")
        os.makedirs(d, exist_ok=True)
        return d

    def _stored_app(self):
        try:
            with open(os.path.join(self._app_dir(), "info.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _store_app(self, path):
        info = ipa_info(path)
        dest = os.path.join(self._app_dir(), "Levanta.ipa")
        shutil.copyfile(path, dest)
        info["guardada"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        info["bytes"] = os.path.getsize(dest)
        with open(os.path.join(self._app_dir(), "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f)
        self.publish_app_status()
        return info

    def publish_app_status(self):
        info = self._stored_app() or {}
        self.server.status["app_build"] = info.get("build", 0)
        self.server.status["app_version"] = info.get("version", "")

    async def instalar_iphone(self, request):
        p = os.path.join(self._app_dir(), "Levanta.ipa")
        if not os.path.exists(p):
            return web.json_response({"ok": False, "mensaje": "Primero guarda una app: descárgala o elige el archivo .ipa."})
        sl = find_sideloadly()
        if sl:
            subprocess.Popen([sl], close_fds=True)
        reveal(p)
        return web.json_response({"ok": True, "sideloadly": bool(sl),
                                  "mensaje": ("Se abrió Sideloadly y la carpeta con Levanta.ipa: arrástrala a Sideloadly y pulsa Start."
                                              if sl else "Se abrió la carpeta con Levanta.ipa. Instala Sideloadly (sideloadly.io) "
                                                         "y arrastra el archivo.")})

    async def app_info(self, request):
        return web.json_response(self._stored_app() or {})

    async def app_ipa(self, request):
        p = os.path.join(self._app_dir(), "Levanta.ipa")
        if not os.path.exists(p):
            raise web.HTTPNotFound(text="No hay app guardada en la PC")
        return web.FileResponse(p, headers={"Content-Disposition": "attachment; filename=Levanta.ipa"})

    @staticmethod
    @web.middleware
    async def solo_local(request, handler):
        """Las funciones de la ventana solo responden a esta PC (no a otros equipos de la red),
        y las acciones exigen un encabezado propio para que ninguna página web pueda dispararlas."""
        if request.path.startswith("/api/"):
            if request.remote not in ("127.0.0.1", "::1"):
                raise web.HTTPForbidden(text="Solo disponible en esta PC")
            if request.method == "POST" and request.headers.get("X-Levanta") != "1":
                raise web.HTTPForbidden(text="Encabezado faltante")
        return await handler(request)

    def register(self, app):
        app.router.add_get("/api/escaneos", self.escaneos)
        app.router.add_post("/api/abrir", self.abrir)
        app.router.add_post("/api/reprocesar", self.reprocesar)
        app.router.add_get("/api/version", self.version)
        app.router.add_get("/api/actualizaciones", self.buscar)
        app.router.add_post("/api/actualizar_pc", self.actualizar_pc)
        app.router.add_post("/api/descargar_app", self.descargar_app)
        app.router.add_post("/api/salir", self.salir)
        app.router.add_post("/api/archivo/validar", self.archivo_validar)
        app.router.add_post("/api/archivo/instalar", self.archivo_instalar)
        app.router.add_get("/api/ultima", self.ultima)
        app.router.add_post("/api/instalar_iphone", self.instalar_iphone)
        app.router.add_get("/api/modelo/{carpeta}", self.modelo)
        app.router.add_post("/api/carpeta/elegir", self.carpeta_elegir)
        app.router.add_post("/api/carpeta", self.carpeta_fijar)
        app.router.add_get("/api/ia", self.ia_estado)
        app.router.add_post("/api/ia/config", self.ia_config_set)
        app.router.add_post("/api/ia/instalar", self.ia_instalar)
        app.router.add_get("/api/medidas/{carpeta}", self.medidas_get)
        app.router.add_post("/api/medidas/{carpeta}", self.medidas_set)
        app.router.add_get("/telefono/modelos.json", self.tel_modelos)        # para el iPhone (solo lectura)
        app.router.add_get("/telefono/modelo/{carpeta}", self.tel_modelo)
        app.router.add_get("/telefono/medidas/{carpeta}", self.tel_medidas)
        app.router.add_get("/app/info.json", self.app_info)          # visible para el iPhone en la red local
        app.router.add_get("/app/Levanta.ipa", self.app_ipa)
        self.publish_app_status()
