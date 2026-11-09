"""
Funciones de la ventana del programa: lista de escaneos, abrir carpetas, actualizaciones
(del programa de la PC y de la app del iPhone) y cierre.

Las actualizaciones vienen de las versiones publicadas en GitHub:
  pc-<n>     Levantamiento3D_Setup.exe, actualizacion_pc.zip, version.json
  build-<n>  Levanta.ipa (app del iPhone)
"""
import asyncio
import hashlib
import io
import json
import os
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


def downloads_dir():
    d = os.path.join(os.path.expanduser("~"), "Downloads")
    return d if os.path.isdir(d) else os.path.expanduser("~")


class Gestion:
    def __init__(self, server, root):
        self.server = server
        self.root = root
        self.busy = None

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
                               "modelo_texturizado.glb", "reporte.txt"):
                        p = os.path.join(res, fn)
                        if os.path.exists(p):
                            files.append({"nombre": fn, "mb": round(os.path.getsize(p) / 1e6, 1)})
                frames = 0
                try:
                    with open(os.path.join(d, "frames.jsonl"), encoding="utf-8") as f:
                        frames = sum(1 for line in f if line.strip())
                except OSError:
                    pass
                items.append({
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
        open_path(self._safe(data.get("ruta", "")))
        return web.json_response({"ok": True})

    async def reprocesar(self, request):
        data = await request.json()
        d = self._safe(data.get("carpeta", ""))
        if self.server.status.get("procesando"):
            return web.json_response({"ok": False, "mensaje": "Ya hay un procesamiento en curso"})
        asyncio.create_task(self.server.final_processing(d))
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
        subprocess.Popen([exe, "-I", lanzador, "--esperar", "--sin-ventana"], close_fds=True, creationflags=flags)
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
