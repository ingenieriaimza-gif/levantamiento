#!/usr/bin/env python3
"""
Levantamiento 3D — programa de la PC.

Recibe por red local (Wi-Fi, hotspot del iPhone, hotspot de Windows o USB) lo que captura la app 3D,
lo guarda en formato RGB-D, fusiona en vivo, regresa al iPhone la superficie fusionada para verla
en realidad aumentada y la muestra también en un visor en el navegador de la PC
(http://localhost:8765). Al terminar el escaneo procesa todo en alta calidad: nube E57 para
Navisworks, STL para SolidWorks, modelo con fotos (USDZ/GLB/OBJ) y datos para Gaussian Splatting.

Protocolo (WebSocket binario en /ws):
  mensaje = [1 byte tipo][4 bytes longitud del encabezado, little-endian][encabezado JSON][datos]
  iPhone -> PC:  1 cuadro  {i,t,fx,fy,cx,cy,w,h,T[16],dw,dh,kf,nj,nd,nc} + JPEG + profundidad(mm,u16) + confianza(u8)
                 2 inicio  {nombre, dispositivo, sistema}
                 3 fin     {}  (procesar en alta calidad)
                 4 descartar {}  (el usuario borró el escaneo en el iPhone)
                 5 medidas {medidas:[{id,a[3],b[3],nombre}]}  (medidas hechas en el iPhone, mundo ARKit)
  PC -> visores: 12 medidas {medidas, origen}
  PC -> iPhone:  10 malla  {nv,nt} + float32 xyz*nv + uint8 rgb*nv + uint32 índices*nt*3
                 11 estado {cuadros, clave, mensaje, procesando}
"""
import argparse
import asyncio
import concurrent.futures
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(AQUI, "..", "tools"))
sys.path.insert(0, TOOLS)
sys.path.insert(0, AQUI)
# en Windows: usar python.exe (no pythonw) para los procesos y sin abrir ventanas de consola
_py = os.path.join(os.path.dirname(sys.executable), "python.exe")
PYTHON = _py if os.path.exists(_py) else sys.executable
NO_WINDOW = 0x08000000 if sys.platform.startswith("win") else 0

import cv2  # noqa: E402
import open3d as o3d  # noqa: E402
from aiohttp import WSMsgType, web  # noqa: E402

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
MSG_FRAME, MSG_START, MSG_END, MSG_DISCARD, MSG_MEASURES = 1, 2, 3, 4, 5
MSG_MESH, MSG_STATUS, MSG_MEASURES_OUT = 10, 11, 12


def pack(kind, header, payload=b""):
    h = json.dumps(header).encode()
    return struct.pack("<BI", kind, len(h)) + h + payload


def unpack(data):
    kind, hl = struct.unpack_from("<BI", data, 0)
    header = json.loads(data[5:5 + hl].decode())
    return kind, header, memoryview(data)[5 + hl:]


def local_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    for probe in ("10.255.255.255", "172.20.10.1", "192.168.137.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((probe, 1))
            ips.add(s.getsockname()[0])
            s.close()
        except OSError:
            pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


# ------------------------------------------------------------------ sesión

class Session:
    """Una sesión de escaneo: guarda cuadros en disco y mantiene una fusión en vivo de baja resolución."""

    def __init__(self, root, name, info, live_voxel=0.012, dmax=3.5):
        self.name = name
        self.dir = os.path.join(root, f"RGBD_{name}")
        os.makedirs(os.path.join(self.dir, "frames"), exist_ok=True)
        self.jsonl = open(os.path.join(self.dir, "frames.jsonl"), "a", encoding="utf-8")
        self.info = info
        self.frames = 0
        self.keyframes = 0
        self.dims = {}
        self.dirty = False
        self.dmax = dmax
        o3c = o3d.core
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=live_voxel, block_resolution=8, block_count=20000,
            device=o3c.Device("CPU:0"))
        self.trunc_mult = 4.0

    def add_frame(self, h, jpeg, depth, conf):
        i = int(h["i"])
        name = f"{i:06d}"
        fdir = os.path.join(self.dir, "frames")
        if len(jpeg):
            with open(os.path.join(fdir, name + ".jpg"), "wb") as f:
                f.write(jpeg)
        with open(os.path.join(fdir, name + ".depth"), "wb") as f:
            f.write(depth)
        with open(os.path.join(fdir, name + ".conf"), "wb") as f:
            f.write(conf)
        line = {k: h[k] for k in ("i", "t", "fx", "fy", "cx", "cy", "w", "h", "T") if k in h}
        line["kf"] = bool(h.get("kf", True))
        self.jsonl.write(json.dumps(line) + "\n")
        self.jsonl.flush()
        self.frames += 1
        self.keyframes += int(line["kf"])
        dw, dh = int(h["dw"]), int(h["dh"])
        self.dims = {"imageWidth": int(h["w"]), "imageHeight": int(h["h"]), "depthWidth": dw, "depthHeight": dh}
        self._integrate(h, jpeg, depth, conf, dw, dh)

    def _integrate(self, h, jpeg, depth, conf, dw, dh):
        d = np.frombuffer(depth, dtype="<u2").reshape(dh, dw).copy()
        c = np.frombuffer(conf, dtype=np.uint8).reshape(dh, dw)
        d[c < 1] = 0
        if len(jpeg):
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            col = cv2.cvtColor(cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        else:
            col = np.full((dh, dw, 3), 160, np.uint8)
        sx, sy = dw / float(h["w"]), dh / float(h["h"])
        K = np.array([[h["fx"] * sx, 0, (h["cx"] + 0.5) * sx - 0.5 + 0.5],
                      [0, h["fy"] * sy, (h["cy"] + 0.5) * sy - 0.5 + 0.5],
                      [0, 0, 1]], dtype=np.float64)        # +0.5: Open3D trunca la proyección
        T = np.array(h["T"], float).reshape(4, 4).T @ ARKIT_TO_CV
        o3c = o3d.core
        Kt, Et = o3c.Tensor(K), o3c.Tensor(np.linalg.inv(T))
        dimg = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(d)))
        cimg = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(col)))
        coords = self.vbg.compute_unique_block_coordinates(dimg, Kt, Et, 1000.0, self.dmax,
                                                           trunc_voxel_multiplier=self.trunc_mult)
        self.vbg.integrate(coords, dimg, cimg, Kt, Et, 1000.0, self.dmax, trunc_voxel_multiplier=self.trunc_mult)
        self.dirty = True

    def preview_mesh(self, max_triangles=220_000):
        """Malla de vista previa (mundo ARKit) empaquetada para el iPhone y el visor."""
        self.dirty = False
        tm = self.vbg.extract_triangle_mesh(weight_threshold=2.0)
        if tm.triangle.indices.shape[0] == 0:
            return None
        m = tm.to_legacy()
        if len(m.triangles) > max_triangles:
            m = m.simplify_vertex_clustering(voxel_size=0.012 * (len(m.triangles) / max_triangles) ** 0.5)
        V = np.asarray(m.vertices, dtype=np.float32)
        C = np.asarray(m.vertex_colors)
        if C.size and C.max() > 1.5:
            C = C / 255.0
        C = (np.clip(C, 0, 1) * 255).astype(np.uint8) if C.size else np.full((len(V), 3), 180, np.uint8)
        F = np.asarray(m.triangles, dtype=np.uint32)
        payload = V.tobytes() + C.tobytes() + F.tobytes()
        return pack(MSG_MESH, {"nv": int(len(V)), "nt": int(len(F))}, payload)

    def close(self):
        self.jsonl.close()
        meta = {"format": "levanta-rgbd", "version": 2, "frames": self.frames, "depthUnits": "mm",
                "poseConvention": "ARKit camera-to-world, column-major 4x4; camera x right, y up, z back; "
                                  "world y up (gravity), meters",
                "intrinsicsAt": "image resolution", **self.dims, **self.info}
        with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# ------------------------------------------------------------------ servidor

class Server:
    def __init__(self, root, port):
        self.root = root
        self.port = port
        self.session = None
        self.phone = None
        self.viewers = set()
        self.last_mesh = None
        self.phone_measures = []
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)   # fusión: un hilo
        self.status = {"cuadros": 0, "clave": 0, "mensaje": "Esperando al iPhone", "procesando": False,
                       "resultados": []}

    async def broadcast(self, data, phone=True):
        targets = list(self.viewers) + ([self.phone] if phone and self.phone is not None else [])
        for ws in targets:
            try:
                await ws.send_bytes(data)
            except Exception:
                self.viewers.discard(ws)

    async def send_status(self, msg=None):
        if msg:
            self.status["mensaje"] = msg
            try:
                print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
            except Exception:       # noqa: BLE001 — el registro nunca debe detener el programa
                pass
        await self.broadcast(pack(MSG_STATUS, self.status))

    async def ws_phone(self, request):
        ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024, heartbeat=20)
        await ws.prepare(request)
        self.phone = ws
        peer = request.remote
        await self.send_status(f"iPhone conectado ({peer})")
        loop = asyncio.get_running_loop()
        try:
            async for msg in ws:
                if msg.type != WSMsgType.BINARY:
                    continue
                kind, h, data = unpack(msg.data)
                if kind == MSG_START:
                    if self.session is None:
                        name = h.get("nombre") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        self.session = Session(self.root, name,
                                               {"device": h.get("dispositivo", "iPhone"),
                                                "system": h.get("sistema", "")})
                        self.status.update(cuadros=0, clave=0)
                        await self.send_status(f"Escaneo iniciado: {name}")
                    else:
                        await self.send_status("Escaneo reanudado")
                elif kind == MSG_FRAME and self.session is not None:
                    nj, nd, nc = int(h["nj"]), int(h["nd"]), int(h["nc"])
                    jpeg = bytes(data[:nj])
                    depth = bytes(data[nj:nj + nd])
                    conf = bytes(data[nj + nd:nj + nd + nc])
                    await loop.run_in_executor(self.pool, self.session.add_frame, h, jpeg, depth, conf)
                    self.status["cuadros"] = self.session.frames
                    self.status["clave"] = self.session.keyframes
                elif kind == MSG_MEASURES:
                    meds = [dict(m, origen="iphone") for m in h.get("medidas", [])][:500]
                    self.phone_measures = meds
                    if self.session is not None:
                        with open(os.path.join(self.session.dir, "medidas_iphone.json"), "w", encoding="utf-8") as f:
                            json.dump({"medidas": meds}, f)
                    await self.broadcast(pack(MSG_MEASURES_OUT, {"medidas": meds, "origen": "iphone"}), phone=False)
                elif kind == MSG_DISCARD and self.session is not None:
                    sess = self.session
                    self.session = None
                    self.last_mesh = None
                    await loop.run_in_executor(self.pool, sess.close)
                    try:
                        os.replace(sess.dir, sess.dir + "_descartado")
                    except OSError:
                        pass
                    self.status.update(cuadros=0, clave=0)
                    await self.send_status("Escaneo descartado desde el iPhone")
                elif kind == MSG_END and self.session is not None:
                    sess = self.session
                    self.session = None
                    await loop.run_in_executor(self.pool, sess.close)
                    await self.send_status(f"Escaneo recibido: {sess.frames} cuadros. Procesando en alta calidad…")
                    asyncio.create_task(self.final_processing(sess.dir))
        finally:
            if self.phone is ws:
                self.phone = None
            await self.send_status("iPhone desconectado (lo recibido está guardado)")
        return ws

    async def ws_viewer(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.viewers.add(ws)
        await ws.send_bytes(pack(MSG_STATUS, self.status))
        if self.last_mesh:
            await ws.send_bytes(self.last_mesh)
        if self.phone_measures:
            await ws.send_bytes(pack(MSG_MEASURES_OUT, {"medidas": self.phone_measures, "origen": "iphone"}))
        async for _ in ws:
            pass
        self.viewers.discard(ws)
        return ws

    async def mesh_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(2.0)
            s = self.session
            if s is None or not s.dirty:
                continue
            try:
                data = await loop.run_in_executor(self.pool, s.preview_mesh)
            except Exception as e:      # noqa: BLE001
                print("Error al extraer la malla:", e)
                continue
            if data:
                self.last_mesh = data
                await self.broadcast(data)
                await self.broadcast(pack(MSG_STATUS, self.status))

    async def final_processing(self, sdir, escala=None):
        self.status["procesando"] = True
        out = os.path.join(sdir, "resultado")
        py = PYTHON
        # -X utf8: salida en UTF-8 aunque Windows use otra codificación por defecto
        cmds = [
            [py, "-I", "-X", "utf8", os.path.join(TOOLS, "procesar_rgbd.py"), sdir, "--salida", out,
             "--voxel", "0.004", "--dmax", "3.5", "--conf", "2", "--lecturas", "2",
             "--triangulos-textura", "1500000"] + (["--escala", str(escala), "1"] if escala else []),
            [py, "-I", "-X", "utf8", os.path.join(TOOLS, "exportar_splat.py"), sdir,
             "--salida", os.path.join(sdir, "splat"), "--nube", os.path.join(out, "nube_fusionada.ply")],
        ]
        try:
            for cmd in cmds:
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.STDOUT,
                                                            creationflags=NO_WINDOW)
                async for line in proc.stdout:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text and not text.startswith("Warning"):
                        await self.send_status(text)
                await proc.wait()
                if proc.returncode != 0:
                    await self.send_status(f"El procesamiento falló (código {proc.returncode}). "
                                           f"Los datos están en {sdir}")
                    break
            else:
                self.status["resultados"].append(sdir)
                await self.send_status(f"Listo. Resultados en {out}")
        except Exception as e:      # noqa: BLE001
            await self.send_status(f"El procesamiento falló: {e}. Los datos están en {sdir}")
        finally:
            self.status["procesando"] = False
            await self.send_status()

    async def index(self, request):
        return web.FileResponse(os.path.join(AQUI, "web", "visor.html"))

    async def estado(self, request):
        return web.json_response({**self.status, "ips": local_ips(), "puerto": self.port})


def advertise(port):
    """Anuncia el servicio en la red local para que la app lo encuentre sola."""
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        return None
    ips = local_ips()
    if not ips:
        return None
    zc = Zeroconf()
    name = socket.gethostname().split(".")[0][:40] or "PC"
    info = ServiceInfo("_levanta._tcp.local.", f"{name}._levanta._tcp.local.",
                       addresses=[socket.inet_aton(ip) for ip in ips], port=port,
                       properties={"app": "levantamiento3d", "v": "1"}, server=f"{name}.local.")
    zc.register_service(info)
    return zc


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Receptor de escaneos de la app 3D")
    ap.add_argument("--puerto", type=int, default=8765)
    from gestion import default_folder
    ap.add_argument("--carpeta", default=None, help="carpeta de escaneos (por defecto la elegida en el programa)")
    args = ap.parse_args()
    args.carpeta = args.carpeta or default_folder()
    os.makedirs(args.carpeta, exist_ok=True)

    srv = Server(args.carpeta, args.puerto)
    from gestion import Gestion
    gestion = Gestion(srv, args.carpeta)
    app = web.Application(client_max_size=64 * 1024 * 1024, middlewares=[Gestion.solo_local])
    gestion.register(app)
    app.router.add_get("/", srv.index)
    app.router.add_get("/estado", srv.estado)
    app.router.add_get("/ws", srv.ws_phone)
    app.router.add_get("/visor", srv.ws_viewer)
    app.router.add_static("/web/", os.path.join(AQUI, "web"))

    async def on_start(_app):
        _app["mesh_loop"] = asyncio.create_task(srv.mesh_loop())
        _app["zc"] = await asyncio.get_running_loop().run_in_executor(None, advertise, args.puerto)

    async def on_stop(_app):
        _app["mesh_loop"].cancel()
        if _app.get("zc"):
            _app["zc"].close()

    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)

    print("=" * 60)
    print(" Levantamiento 3D — receptor en la PC")
    print(f" Escaneos en: {args.carpeta}")
    print(" Direcciones para la app (si no la encuentra sola):")
    for ip in local_ips():
        print(f"     {ip}:{args.puerto}")
    print(f" Visor en esta PC:  http://localhost:{args.puerto}")
    print("=" * 60, flush=True)
    web.run_app(app, host="0.0.0.0", port=args.puerto, print=None)


if __name__ == "__main__":
    main()
