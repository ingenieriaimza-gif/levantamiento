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
import collections
import socket
import struct
import subprocess
import sys
import threading
import shutil
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
MSG_FRAME, MSG_START, MSG_END, MSG_DISCARD, MSG_MEASURES, MSG_ZONES, MSG_PHOTO = 1, 2, 3, 4, 5, 6, 7
MSG_MESH, MSG_STATUS, MSG_MEASURES_OUT = 10, 11, 12


from diagnostico import explicar_codigo, matar_arbol, medir_arbol  # noqa: E402,F401


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

    def __init__(self, root, name, info, live_voxel=0.012, dmax=3.5, refiner=None):
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
        self.refiner = refiner          # IA en vivo (opcional)
        self.ia_fallos = 0

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

    def add_photo(self, h, jpeg):
        """Foto de máxima resolución (12 MP) con su propia posición: solo sirve para la textura."""
        if not len(jpeg):
            return
        n = getattr(self, "fotos_hr", 0)
        fdir = os.path.join(self.dir, "fotos_hr")
        os.makedirs(fdir, exist_ok=True)
        name = f"{n:06d}"
        with open(os.path.join(fdir, name + ".jpg"), "wb") as f:
            f.write(jpeg)
        line = {k: h[k] for k in ("t", "fx", "fy", "cx", "cy", "w", "h", "T") if k in h}
        line["i"] = n
        with open(os.path.join(self.dir, "fotos_hr.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        self.fotos_hr = n + 1

    def _integrate(self, h, jpeg, depth, conf, dw, dh):
        d = np.frombuffer(depth, dtype="<u2").reshape(dh, dw).copy()
        c = np.frombuffer(conf, dtype=np.uint8).reshape(dh, dw)
        d[c < 1] = 0
        ow, oh = dw, dh
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if len(jpeg) else None
        if img is not None and self.refiner is not None and self.ia_fallos < 5:
            try:
                from ia_profundidad import refinar_cuadro
                ow, oh = 512, int(round(512 * img.shape[0] / img.shape[1]))
                d, _ = refinar_cuadro(self.refiner, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), d, c, (ow, oh))
            except Exception as e:      # noqa: BLE001
                self.ia_fallos += 1
                print("IA en vivo falló:", e)
                ow, oh = dw, dh
                d = np.frombuffer(depth, dtype="<u2").reshape(dh, dw).copy()
                d[c < 1] = 0
        if img is not None:
            col = cv2.cvtColor(cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        else:
            col = np.full((oh, ow, 3), 160, np.uint8)
        sx, sy = ow / float(h["w"]), oh / float(h["h"])
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
        # soltar el volumen de la vista en vivo (cientos de MB): al crear el modelo hace falta toda la memoria
        self.vbg = None
        import gc
        gc.collect()
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
        self.disco_lleno = False
        # trabajo pesado en curso (crear el modelo, vista foto, HTML): para poder verlo por dentro y cancelarlo
        self.proc_actual = None
        self.cancelado = False
        self.ultima_salida = None
        self.salida_viva = collections.deque(maxlen=300)    # últimas líneas del proceso, con hora
        self._bytes_cuadros = 0
        self._n_cuadros = 0
        self._t_primero = None
        self.revisar_disco(forzar=True)

    # Sin tope de cuadros: el límite es el espacio libre del disco. Se dejan 3 GB para crear los modelos.
    RESERVA_DISCO = 3 * 1024 ** 3

    def revisar_disco(self, forzar=False):
        """Actualiza en el estado el espacio libre y cuánto tiempo de escaneo cabe. Devuelve True si hay lugar."""
        if not forzar and self._n_cuadros % 25:
            return not self.disco_lleno
        try:
            libre = shutil.disk_usage(self.root).free
        except OSError:
            return True
        self.status["libre_gb"] = round(libre / 1024 ** 3, 1)
        # ritmo real de este escaneo (bytes por segundo); si aún no hay, ~1.8 MB/s (4 cuadros/s)
        ritmo = 1.8e6
        if self._t_primero and self._n_cuadros > 20:
            dt = max(1.0, time.time() - self._t_primero)
            ritmo = max(2e5, self._bytes_cuadros / dt)
        self.status["libre_horas"] = round(max(0, libre - self.RESERVA_DISCO) / ritmo / 3600, 1)
        lleno = libre < self.RESERVA_DISCO
        self.status["disco_lleno"] = lleno
        self.disco_lleno = lleno
        return not lleno

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
                        refiner = await self.live_refiner()
                        self.session = Session(self.root, name,
                                               {"device": h.get("dispositivo", "iPhone"),
                                                "system": h.get("sistema", "")}, refiner=refiner)
                        self.status.update(cuadros=0, clave=0)
                        self._n_cuadros, self._bytes_cuadros, self._t_primero = 0, 0, None
                        self.revisar_disco(forzar=True)
                        if getattr(self, "zones", None):      # piezas marcadas antes de empezar a grabar
                            with open(os.path.join(self.session.dir, "zonas.json"), "w", encoding="utf-8") as f:
                                json.dump({"zonas": self.zones}, f)
                        await self.send_status(f"Escaneo iniciado: {name}")
                    else:
                        await self.send_status("Escaneo reanudado")
                elif kind == MSG_FRAME and self.session is not None:
                    nj, nd, nc = int(h["nj"]), int(h["nd"]), int(h["nc"])
                    jpeg = bytes(data[:nj])
                    depth = bytes(data[nj:nj + nd])
                    conf = bytes(data[nj + nd:nj + nd + nc])
                    self._n_cuadros += 1
                    self._bytes_cuadros += nj + nd + nc
                    if self._t_primero is None:
                        self._t_primero = time.time()
                    antes = self.disco_lleno
                    if not self.revisar_disco():
                        if not antes:
                            await self.send_status("Disco de la PC casi lleno: ya no se guardan cuadros. "
                                                   "Libera espacio o termina el escaneo.")
                        continue
                    if antes:
                        await self.send_status("Hay espacio de nuevo en la PC: se siguen guardando los cuadros")
                    await loop.run_in_executor(self.pool, self.session.add_frame, h, jpeg, depth, conf)
                    self.status["cuadros"] = self.session.frames
                    self.status["clave"] = self.session.keyframes
                elif kind == MSG_PHOTO and self.session is not None:
                    if not self.disco_lleno:
                        nj = int(h["nj"])
                        await loop.run_in_executor(self.pool, self.session.add_photo, h, bytes(data[:nj]))
                        self._bytes_cuadros += nj
                        self.status["fotos_hr"] = getattr(self.session, "fotos_hr", 0)
                elif kind == MSG_ZONES:
                    # piezas enfocadas en el iPhone: se reconstruyen a 2 mm al crear el modelo
                    zonas = [z for z in h.get("zonas", []) if "centro" in z and "medio" in z][:50]
                    self.zones = zonas
                    if self.session is not None:
                        with open(os.path.join(self.session.dir, "zonas.json"), "w", encoding="utf-8") as f:
                            json.dump({"zonas": zonas}, f)
                    if zonas:
                        await self.send_status(f"Piezas enfocadas: {', '.join(z.get('nombre', '') for z in zonas)}")
                elif kind == MSG_MEASURES:
                    meds = [dict(m, origen="iphone") for m in h.get("medidas", [])][:500]
                    carpeta = h.get("carpeta")
                    if carpeta:
                        # medidas hechas en Revisar 3D sobre un modelo terminado: se guardan con ese escaneo
                        root = os.path.realpath(self.root)
                        d = os.path.realpath(os.path.join(root, str(carpeta)))
                        if d.startswith(root + os.sep) and os.path.isdir(d):
                            with open(os.path.join(d, "medidas_iphone.json"), "w", encoding="utf-8") as f:
                                json.dump({"medidas": meds}, f)
                            await self.broadcast(pack(MSG_MEASURES_OUT, {"medidas": meds, "origen": "iphone",
                                                                         "carpeta": os.path.basename(d)}), phone=False)
                        continue
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

    async def live_refiner(self):
        """Carga la IA para la vista en vivo si está instalada y activada (se carga una sola vez)."""
        from gestion import ia_config, ia_instalada
        if not (ia_instalada() and ia_config()["vivo"]):
            return None
        if getattr(self, "_refiner", None) is None:
            await self.send_status("Cargando la IA para la vista en vivo…")
            try:
                from ia_profundidad import RefinadorIA
                self._refiner = await asyncio.get_running_loop().run_in_executor(self.pool, RefinadorIA)
                await self.send_status(f"IA en vivo lista ({self._refiner.device})")
            except Exception as e:      # noqa: BLE001
                await self.send_status(f"No se pudo cargar la IA en vivo: {e}")
                return None
        return self._refiner

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
        pausa = 2.0
        while True:
            await asyncio.sleep(pausa)
            s = self.session
            if s is None or not s.dirty:
                continue
            t_ini = time.time()
            try:
                data = await loop.run_in_executor(self.pool, s.preview_mesh)
            except Exception as e:      # noqa: BLE001
                print("Error al extraer la malla:", e)
                continue
            # en un espacio grande sacar la vista previa tarda más: se espacia para que recibir y guardar los
            # cuadros del iPhone (lo importante) tenga siempre al menos el 80 % del tiempo
            pausa = min(30.0, max(2.0, 4.0 * (time.time() - t_ini)))
            if data:
                self.last_mesh = data
                await self.broadcast(data)
                await self.broadcast(pack(MSG_STATUS, self.status))

    # ---------------------------------------------------------- trabajo en curso: ver por dentro y cancelar

    def _registrar_proc(self, proc, nombre):
        self.proc_actual = proc
        self.ultima_salida = time.time()
        self.status["diagnostico"] = {"pid": proc.pid, "trabajo": nombre, "inicio": time.time()}
        self._linea(f"» inicia {nombre} (pid {proc.pid})")

    def _linea(self, texto):
        self.ultima_salida = time.time()
        self.salida_viva.append(f"[{datetime.now():%H:%M:%S}] {texto}")

    async def cancelar(self):
        """Detiene el trabajo en curso (y todos sus procesos hijos). Lo recibido del iPhone no se borra."""
        if not self.status.get("procesando"):
            return False, "No hay nada procesándose"
        self.cancelado = True
        proc = self.proc_actual
        self._linea("» cancelado por el usuario")
        await self.send_status("Cancelando… (deteniendo los procesos)")
        if proc is not None and proc.returncode is None:
            await asyncio.get_running_loop().run_in_executor(None, matar_arbol, proc.pid)
        return True, "Cancelado"

    async def vigilar(self):
        """Cada 5 s: CPU, RAM y segundos sin noticias del proceso en curso. Va en el estado (visor y Claude)
        y cada minuto queda en el registro del escaneo, para saber qué pasó si se cae sin avisar."""
        ultimo_reg = 0.0
        aviso_ram = False
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(5.0)
            proc = self.proc_actual
            if not self.status.get("procesando") or proc is None or proc.returncode is not None:
                aviso_ram = False
                continue
            m = await loop.run_in_executor(None, medir_arbol, proc.pid)
            d = self.status.get("diagnostico") or {}
            d.update(m or {})
            d["sin_salida_s"] = round(time.time() - (self.ultima_salida or time.time()))
            d["lleva_s"] = round(time.time() - d.get("inicio", time.time()))
            self.status["diagnostico"] = d
            if m and m["ram_libre_mb"] < 0.06 * m["ram_total_mb"] and not aviso_ram:
                aviso_ram = True
                self._linea(f"⚠ RAM casi agotada: quedan {m['ram_libre_mb']} MB libres; el proceso usa {m['ram_mb']} MB")
                await self.send_status(f"Atención: la PC se está quedando sin memoria RAM "
                                       f"(libres {m['ram_libre_mb']} MB). Si se congela, cancela.")
            if m and time.time() - ultimo_reg > 60:
                ultimo_reg = time.time()
                reg = getattr(self, "_reg_escaneo", None)
                if reg:
                    reg(f"diag: {m['procesos']} procesos · CPU {m['cpu_pct']} % · RAM {m['ram_mb']} MB "
                        f"(pedida {m.get('privada_mb', '?')} MB; libre {m['ram_libre_mb']} de {m['ram_total_mb']} MB) · "
                        f"fallos de página {m.get('fallos_pag_s', '?')}/s · {d['sin_salida_s']} s sin mensajes"
                        + (" · PAGINANDO AL DISCO" if m.get("paginando") else ""))
            await self.send_status()

    def info_proceso(self):
        """Todo lo que se sabe del trabajo en curso (para /api/proceso y el conector de Claude)."""
        return {"procesando": bool(self.status.get("procesando")), "carpeta": os.path.basename(
                    getattr(self, "processing_dir", "") or "") or None,
                "mensaje": self.status.get("mensaje"), "progreso": self.status.get("progreso"),
                "diagnostico": self.status.get("diagnostico"), "cancelando": self.cancelado,
                "salida": list(self.salida_viva)}

    async def final_processing(self, sdir, escala=None, desde_cero=False):
        self.status["procesando"] = True
        self.cancelado = False
        self.salida_viva.clear()
        self.processing_dir = sdir
        out = os.path.join(sdir, "resultado")
        py = PYTHON
        from gestion import ia_config, ia_instalada
        usar_ia = ia_instalada() and ia_config()["final"]
        if usar_ia:
            await self.send_status("Procesando con IA de profundidad (Prompt Depth Anything)…")
        # -X utf8: salida en UTF-8 aunque Windows use otra codificación por defecto
        cmds = [
            [py, "-I", "-X", "utf8", os.path.join(TOOLS, "procesar_rgbd.py"), sdir, "--salida", out,
             "--voxel", "0.004", "--dmax", "3.5", "--conf", "2", "--lecturas", "2",
             # triángulos, texturas y vóxel final los decide procesar_rgbd según la RAM de la PC y el tamaño
             # del espacio (tools/perfil.py); el avance se guarda para continuar si se cancela o falla
             "--avance", os.path.join(sdir, "avance"),
             "--verificacion", os.path.join(sdir, "verificacion.txt")]
            + (["--desde-cero"] if desde_cero else [])
            + (["--comparar", os.path.join(sdir, "comparacion")] if self.modo_prueba() else []) + (["--escala", str(escala), "1"] if escala else [])
            + (["--ia"] if usar_ia else []),
        ]
        # solo el modelo para Navisworks (FBX) y SolidWorks (3MF); sin vista foto-realista (tardaba mucho)
        tramos = [(0.0, 1.0)]
        t0 = time.time()
        ultimas = collections.deque(maxlen=40)
        err_path = os.path.join(out, "error.txt")
        try:
            os.remove(err_path)
        except OSError:
            pass
        self.status["progreso"] = {"carpeta": os.path.basename(sdir), "fraccion": 0.0,
                                   "etapa": "Empezando", "restante_s": None}
        await self.send_status("Creando el modelo 3D…")
        last_sent = 0.0
        fallo = None
        # registro de ESTE escaneo (etapas y tiempos), para copiarlo desde la lista de escaneos
        try:
            reg_f = open(os.path.join(sdir, "registro_escaneo.txt"), "a", encoding="utf-8", buffering=1)
        except OSError:
            reg_f = None

        def reg(msg):
            if reg_f:
                try:
                    reg_f.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")
                except (OSError, ValueError):
                    pass
        self._reg_escaneo = reg
        reg(f"--- creación del modelo {os.path.basename(sdir)} · {os.cpu_count()} núcleos · IA: {'sí' if usar_ia else 'no'} ---")
        _sin_ruta = {"--salida", "--verificacion", "--comparar", "--avance"}
        reg("opciones: " + " ".join(x for x in cmds[0][6:] if x not in _sin_ruta and not (os.path.isabs(x) or os.sep in x)))
        etapa_prev = None
        proc = None
        try:
            for (a, b), cmd in zip(tramos, cmds):
                if self.cancelado:
                    break
                # toda la PC: todos los núcleos y prioridad alta (la PC sigue usable, pero esto va primero)
                nuc = str(os.cpu_count() or 4)
                # PYTHONFAULTHANDLER: si una librería se cae (Open3D, numpy) queda en el registro en qué línea iba;
                # PYTHONUNBUFFERED: los mensajes llegan al momento, no en bloques
                env = dict(os.environ, OMP_NUM_THREADS=nuc, MKL_NUM_THREADS=nuc, OPENBLAS_NUM_THREADS=nuc,
                           NUMEXPR_NUM_THREADS=nuc, PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
                prioridad = 0x00008000 if sys.platform.startswith("win") else 0     # ABOVE_NORMAL_PRIORITY_CLASS
                try:
                    from perfil import perfil
                    if not perfil()["prioridad_alta"]:
                        # PC con poca memoria: prioridad normal, para que la ventana (y Cancelar) sigan respondiendo
                        prioridad = 0
                except Exception:       # noqa: BLE001
                    pass
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.STDOUT, env=env,
                                                            limit=4 * 1024 * 1024,     # renglones largos (tracebacks)
                                                            creationflags=NO_WINDOW | prioridad)
                self._registrar_proc(proc, "crear el modelo 3D")
                async for line in proc.stdout:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if not text:
                        continue
                    if not text.startswith("@@PROGRESO "):
                        self._linea(text)
                    else:
                        self.ultima_salida = time.time()
                    if text.startswith("Warning"):
                        continue
                    # volcado automático cuando el proceso lleva rato sin decir nada (faulthandler):
                    # va al registro del escaneo y a "Por dentro", no al iPhone
                    if text.startswith(("Timeout (", "Thread 0x", "Current thread 0x", '  File "', "Stack (most")):
                        if text.startswith("Timeout ("):
                            d = self.status.get("diagnostico") or {}
                            reg(f"sin mensajes hace rato (minuto {(time.time() - t0) / 60:.0f}); si la CPU sigue "
                                f"trabajando es una etapa larga, no un atasco. "
                                f"RAM del proceso {d.get('ram_mb')} MB, pedida {d.get('privada_mb')} MB, "
                                f"libre {d.get('ram_libre_mb')} MB, fallos de página {d.get('fallos_pag_s')}/s. Dónde va:")
                        elif text.startswith('  File "'):
                            reg("   " + text.strip())
                        continue
                    if text.startswith("@@PROGRESO "):
                        try:
                            f, etapa = text[11:].split("|", 1)
                            frac = a + (b - a) * float(f)
                        except ValueError:
                            continue
                        el = time.time() - t0
                        if etapa != etapa_prev:
                            etapa_prev = etapa
                            reg(f"+{el / 60:.1f} min · {etapa} ({frac * 100:.0f} %)")
                            self._linea(f"» etapa: {etapa} ({frac * 100:.0f} %)")
                        # tiempo restante: lo que ha tardado, repartido según el avance (se estabiliza solo)
                        rest = el / frac - el if frac > 0.07 else None
                        prev = self.status["progreso"].get("restante_s")
                        if rest is not None and prev is not None:
                            rest = 0.7 * prev + 0.3 * rest
                        self.status["progreso"] = {"carpeta": os.path.basename(sdir), "fraccion": round(frac, 4),
                                                   "etapa": etapa, "restante_s": None if rest is None else round(rest),
                                                   "transcurrido_s": round(el)}
                        if time.time() - last_sent > 1.0:
                            last_sent = time.time()
                            await self.send_status()
                        continue
                    ultimas.append(text)
                    reg(text)
                    await self.send_status(text)
                await proc.wait()
                self._linea(f"» terminó: {explicar_codigo(proc.returncode)}")
                if self.cancelado:
                    break
                if proc.returncode != 0:
                    fallo = f"El programa terminó con error ({explicar_codigo(proc.returncode)})."
                    break
        except Exception as e:      # noqa: BLE001
            fallo = f"{type(e).__name__}: {e}"
            self._linea(f"» error en el programa de la PC: {fallo}")
        finally:
            # nunca dejar procesos huérfanos trabajando (si algo falló aquí o se canceló)
            if proc is not None and proc.returncode is None:
                await asyncio.get_running_loop().run_in_executor(None, matar_arbol, proc.pid)
            self.proc_actual = None
        if self.cancelado:
            fallo = "Cancelado por el usuario."
        reg((f"TERMINÓ CON ERROR: {fallo}" if fallo else "Listo") + f" · tiempo total {(time.time() - t0) / 60:.1f} min")
        self._reg_escaneo = None
        if reg_f:
            reg_f.close()
        try:
            if self.cancelado:
                os.makedirs(out, exist_ok=True)
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write(fallo + " Las fotos del escaneo siguen guardadas: puedes volver a crear el modelo.\n\n"
                            + "\n".join(ultimas) + "\n")
                self.status["progreso"] = None
                await self.send_status("Creación del modelo cancelada. Lo recibido del iPhone sigue guardado.")
            elif fallo:
                os.makedirs(out, exist_ok=True)
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write(fallo + "\n\n" + "\n".join(ultimas) + "\n")
                self.status["progreso"] = None
                await self.send_status(f"No se pudo crear el modelo: {fallo} Detalle en resultado\\error.txt")
            else:
                loop_ = asyncio.get_running_loop()
                if await loop_.run_in_executor(None, self._dejar_solo_navisworks, out):
                    if not self.modo_prueba():        # en modo de prueba se conservan las fotos para revisar y repetir
                        await loop_.run_in_executor(None, self._borrar_capturas, sdir)
                    # versión ligera para "Revisar 3D" y "Mis escaneos" del iPhone (si falla, el modelo ya quedó)
                    self.status["progreso"] = {"carpeta": os.path.basename(sdir), "fraccion": 0.99,
                                               "etapa": "Preparando el modelo para el iPhone", "restante_s": 90,
                                               "transcurrido_s": round(time.time() - t0)}
                    await self.send_status("Modelo creado. Preparándolo para verlo en el iPhone…")
                    try:
                        await self.preparar_iphone(sdir, registrar=True)
                    except Exception as e:      # noqa: BLE001
                        self._linea(f"» no se pudo preparar el modelo para el iPhone: {e}")
                self.status["resultados"].append(sdir)
                self.status["progreso"] = {"carpeta": os.path.basename(sdir), "fraccion": 1.0,
                                           "etapa": "Listo", "restante_s": 0,
                                           "transcurrido_s": round(time.time() - t0)}
                resumen = ""
                try:
                    with open(os.path.join(sdir, "verificacion.json"), encoding="utf-8") as f:
                        v = json.load(f)
                    con = [m for m in v.get("medidas", []) if "diferencia_mm" in m]
                    if con:
                        resumen = (f" Verificación: {len(con)} medida(s) del iPhone comprobada(s), mayor diferencia "
                                   f"{max(abs(m['diferencia_mm']) for m in con):.1f} mm.")
                    elif "sesgo_mm" in v:
                        resumen = f" Verificación: desviación del modelo respecto al LiDAR {v['sesgo_mm']:+.1f} mm."
                    resumen += " Detalle en verificacion.txt"
                except (OSError, ValueError):
                    pass
                await self.send_status(f"Listo. Resultados en {out}.{resumen}")
        finally:
            self.status["procesando"] = False
            self.status["diagnostico"] = None
            self.cancelado = False
            await self.send_status()

    # ---------------------------------------------------------- modelo ligero para el iPhone

    @staticmethod
    def iphone_info(sdir):
        """info.json del modelo para el iPhone, si corresponde al FBX actual (None = todavía no se prepara)."""
        fbx = os.path.join(sdir, "resultado", "obj", "modelo_navisworks.fbx")
        try:
            with open(os.path.join(sdir, "iphone", "info.json"), encoding="utf-8") as f:
                info = json.load(f)
            if info.get("fbx_mtime") != int(os.path.getmtime(fbx)):
                return None                 # el modelo se volvió a crear: hay que prepararlo otra vez
            if info.get("ok") and not os.path.exists(os.path.join(sdir, "iphone", "modelo.usdz")):
                return None
            return info
        except (OSError, ValueError):
            return None

    async def preparar_iphone(self, sdir, registrar=False):
        """Deja en <escaneo>/iphone/ el modelo terminado en versión ligera (USDZ + miniatura) para verlo en
        "Revisar 3D" y "Mis escaneos" del iPhone. Uno a la vez. Devuelve True si quedó listo."""
        fbx = os.path.join(sdir, "resultado", "obj", "modelo_navisworks.fbx")
        if not os.path.exists(fbx):
            return False
        lock = self.__dict__.setdefault("_iphone_lock", asyncio.Lock())
        async with lock:
            info = self.iphone_info(sdir)
            if info is not None:
                return bool(info.get("ok"))
            cmd = [PYTHON, "-I", "-X", "utf8", os.path.join(TOOLS, "fbx_a_usdz.py"), fbx, os.path.join(sdir, "iphone")]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.STDOUT, creationflags=NO_WINDOW)
            if registrar:
                self._registrar_proc(proc, "modelo para el iPhone")
            try:
                out, _ = await proc.communicate()
            finally:
                if registrar:
                    self.proc_actual = None
            for t in out.decode("utf-8", errors="replace").splitlines():
                if t.strip() and not t.startswith("Warning"):
                    self._linea(t.strip())
            return proc.returncode == 0

    def iphone_ocupado(self):
        lock = self.__dict__.get("_iphone_lock")
        return bool(lock and lock.locked())

    def modo_prueba(self):
        """Modo de prueba (lo activa Claude por el conector): no se borran las fotos al terminar el modelo
        y se guarda la comparación del modelo contra fotos reales."""
        return os.path.exists(os.path.join(self.root, "modo_prueba.txt"))

    @staticmethod
    def _dejar_solo_navisworks(out):
        """Ya con el FBX listo, borra lo demás del resultado (GLB, USDZ, reporte, poses...): solo queda la
        carpeta obj con el FBX y sus texturas. Si el FBX no se creó, no se borra nada."""
        import shutil
        if not os.path.exists(os.path.join(out, "obj", "modelo_navisworks.fbx")):
            return False
        import time as _t
        obj = os.path.join(out, "obj")
        # en Windows un archivo puede estar bloqueado unos segundos (antivirus, indexador): se reintenta
        for _intento in range(10):
            pendientes = 0
            for fn in os.listdir(obj):
                if not (fn == "modelo_navisworks.fbx" or (fn.startswith("textura_") and fn.endswith(".jpg"))):
                    try:
                        os.remove(os.path.join(obj, fn))
                    except OSError:
                        pendientes += 1
            for fn in os.listdir(out):
                if fn == "obj":
                    continue
                p = os.path.join(out, fn)
                try:
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
                except OSError:
                    pendientes += 1
            if not pendientes:
                break
            _t.sleep(1.5)
        return True

    @staticmethod
    def _borrar_capturas(sdir):
        """Con el FBX ya hecho, borra lo que mandó el iPhone (fotos, profundidad, poses): es lo más pesado.
        Se conservan las medidas y el resultado. Guarda cuántos cuadros eran para mostrarlo en la lista."""
        import shutil
        try:
            with open(os.path.join(sdir, "frames.jsonl"), encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            with open(os.path.join(sdir, "cuadros.txt"), "w") as f:
                f.write(str(n))
        except OSError:
            pass
        for fn in ("frames", "fotos_hr", "avance"):
            shutil.rmtree(os.path.join(sdir, fn), ignore_errors=True)
        for fn in ("frames.jsonl", "fotos_hr.jsonl", "meta.json", "zonas.json"):
            try:
                os.remove(os.path.join(sdir, fn))
            except OSError:
                pass

    async def photo_view(self, sdir):
        """Vista foto-realista (Gaussian Splatting) en la tarjeta de video. Si falla, el modelo ya quedó listo."""
        import vista_foto
        res = os.path.join(sdir, "resultado")
        err_path = os.path.join(res, "vista_foto_error.txt")
        try:
            os.remove(err_path)
        except OSError:
            pass
        carpeta = os.path.basename(sdir)
        if not vista_foto.disponible():
            with open(err_path, "w", encoding="utf-8") as f:
                f.write("Esta PC no tiene tarjeta de video NVIDIA: la vista foto-realista tardaría horas.\n")
            return
        t0 = time.time()
        state = {"sent": 0.0, "rest": None}
        self.status["fase"] = "foto"

        def on(frac, etapa):
            el = time.time() - t0
            rest = el / frac - el if frac > 0.05 else None
            if rest is not None and state["rest"] is not None:
                rest = 0.8 * state["rest"] + 0.2 * rest
            state["rest"] = rest
            self.status["progreso"] = {"carpeta": carpeta, "fraccion": round(frac, 4), "etapa": etapa, "fase": "foto",
                                       "restante_s": None if rest is None else round(rest),
                                       "transcurrido_s": round(el)}
            if time.time() - state["sent"] > 1.0:
                state["sent"] = time.time()
                asyncio.ensure_future(self.send_status())

        try:
            on(0.0, "Preparando la vista foto-realista")
            await self.send_status("Creando la vista foto-realista (los archivos del modelo ya se pueden usar)…")
            await vista_foto.entrenar(sdir, on, on_linea=self._linea,
                                      on_proc=lambda p: self._registrar_proc(p, "vista foto-realista"))
            if self.cancelado:
                raise RuntimeError("cancelado por el usuario")
            self.status["progreso"] = {"carpeta": carpeta, "fraccion": 1.0, "etapa": "Listo", "fase": "foto",
                                       "restante_s": 0, "transcurrido_s": round(time.time() - t0)}
            on(0.995, "Creando el archivo HTML para compartir")
            await self.make_html(sdir)
            await self.send_status("Vista foto-realista lista: ábrela en Revisar 3D o con el archivo HTML")
        except Exception as e:      # noqa: BLE001
            if self.cancelado:
                e = "cancelado por el usuario"
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(f"{e}\n")
            self.status["progreso"] = None
            await self.send_status(f"No se pudo crear la vista foto-realista: {e}")
        finally:
            self.status["fase"] = None
            self.proc_actual = None

    async def make_html(self, sdir):
        """Archivo HTML único (vista foto-realista + medir). Si falla, no afecta lo demás."""
        cmd = [PYTHON, "-I", "-X", "utf8", os.path.join(TOOLS, "exportar_html.py"), sdir]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT, creationflags=NO_WINDOW,
                                                    env=dict(os.environ, PYTHONFAULTHANDLER="1"))
        self._registrar_proc(proc, "archivo HTML")
        try:
            out, _ = await proc.communicate()
        finally:
            self.proc_actual = None
        text = out.decode("utf-8", errors="replace").strip().splitlines()
        for t in text[-40:]:
            self._linea(t)
        if self.cancelado:
            await self.send_status("Archivo HTML cancelado")
            return False
        if proc.returncode != 0:
            await self.send_status(f"No se pudo crear el archivo HTML: {text[-1] if text else proc.returncode}")
            return False
        return True

    async def only_html(self, sdir):
        self.status["procesando"] = True
        self.cancelado = False
        self.salida_viva.clear()
        self.processing_dir = sdir
        try:
            await self.send_status("Creando el archivo HTML…")
            if await self.make_html(sdir):
                await self.send_status("Archivo HTML listo (en la carpeta resultado del escaneo)")
        finally:
            self.status["procesando"] = False
            self.status["diagnostico"] = None
            self.cancelado = False
            await self.send_status()

    async def only_photo_view(self, sdir):
        """Crea solo la vista foto-realista de un escaneo que ya tiene modelo."""
        self.status["procesando"] = True
        self.cancelado = False
        self.salida_viva.clear()
        self.processing_dir = sdir
        try:
            if not os.path.isdir(os.path.join(sdir, "splat", "sparse", "0")):
                await self.send_status("Preparando las fotos para la vista foto-realista…")
                cmd = [PYTHON, "-I", "-X", "utf8", os.path.join(TOOLS, "exportar_splat.py"), sdir,
                       "--salida", os.path.join(sdir, "splat"),
                       "--nube", os.path.join(sdir, "resultado", "nube_fusionada.ply")]
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.STDOUT, creationflags=NO_WINDOW,
                                                            limit=4 * 1024 * 1024,
                                                            env=dict(os.environ, PYTHONFAULTHANDLER="1",
                                                                     PYTHONUNBUFFERED="1"))
                self._registrar_proc(proc, "preparar fotos (exportar_splat)")
                try:
                    async for line in proc.stdout:
                        t = line.decode("utf-8", errors="replace").rstrip()
                        if t:
                            self._linea(t)
                    await proc.wait()
                finally:
                    self.proc_actual = None
                self._linea(f"» terminó: {explicar_codigo(proc.returncode)}")
            if self.cancelado:
                await self.send_status("Vista foto-realista cancelada")
                return
            await self.photo_view(sdir)
        finally:
            self.status["procesando"] = False
            self.status["diagnostico"] = None
            self.cancelado = False
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
    try:
        # si el nombre quedó anunciado por un arranque anterior, se usa otro en vez de cerrar el programa
        zc.register_service(info, allow_name_change=True)
    except Exception as e:      # noqa: BLE001 — sin anuncio la app aún conecta escribiendo la dirección
        print("No se pudo anunciar el programa en la red (la app puede conectarse con la dirección):", e)
        try:
            zc.close()
        except Exception:       # noqa: BLE001
            pass
        return None
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
        _app["vigilar"] = asyncio.create_task(srv.vigilar())
        _app["zc"] = await asyncio.get_running_loop().run_in_executor(None, advertise, args.puerto)

    async def on_stop(_app):
        _app["mesh_loop"].cancel()
        _app["vigilar"].cancel()
        if srv.proc_actual is not None and srv.proc_actual.returncode is None:
            matar_arbol(srv.proc_actual.pid)        # al cerrar el programa no quedan procesos trabajando
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
