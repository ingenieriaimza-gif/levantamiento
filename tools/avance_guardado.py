"""
Avance guardado: si crear el modelo se cancela, se va la luz o falla a la mitad, la siguiente vez
continúa desde la última etapa terminada en vez de empezar de cero.

Se guarda en <escaneo>/avance/ :
  estado.json     clave del trabajo (qué escaneo y con qué ajustes) y las etapas terminadas
  <etapa>.npz     mallas y tablas de esa etapa
  <etapa>.json    datos chicos de esa etapa (posiciones corregidas, etc.)
Si el escaneo o los ajustes cambian, la clave ya no coincide y el avance viejo se descarta solo.
Al terminar bien el modelo, la carpeta se borra.
"""
import hashlib
import json
import os
import shutil
import time

import numpy as np

VERSION = 3          # súbela si cambia lo que se guarda: los avances de versiones anteriores se descartan


def clave_de(*partes):
    return hashlib.sha256(json.dumps([VERSION, *partes], sort_keys=True, default=str).encode()).hexdigest()[:20]


class Avance:
    def __init__(self, carpeta, clave, log=print):
        self.dir = carpeta
        self.log = log
        self.estado = {"clave": clave, "etapas": {}}
        if not carpeta:
            return
        try:
            os.makedirs(carpeta, exist_ok=True)
            p = os.path.join(carpeta, "estado.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    viejo = json.load(f)
                if viejo.get("clave") == clave:
                    self.estado = viejo
                else:
                    self._vaciar()
        except (OSError, ValueError):
            self._vaciar()

    def _vaciar(self):
        if not self.dir:
            return
        for fn in os.listdir(self.dir) if os.path.isdir(self.dir) else []:
            try:
                os.remove(os.path.join(self.dir, fn))
            except OSError:
                pass

    def _p(self, nombre, ext):
        return os.path.join(self.dir, f"{nombre}.{ext}")

    def tiene(self, etapa):
        return bool(self.dir) and etapa in self.estado["etapas"]

    def _marcar(self, etapa, texto):
        self.estado["etapas"][etapa] = {"hora": time.strftime("%Y-%m-%d %H:%M:%S"), "texto": texto}
        tmp = self._p("estado", "json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.estado, f, ensure_ascii=False)
        os.replace(tmp, self._p("estado", "json"))

    def guardar(self, etapa, texto, datos=None, **tablas):
        """Guarda una etapa terminada: `datos` (JSON) y `tablas` (arreglos numpy). Nunca detiene el proceso."""
        if not self.dir:
            return
        try:
            t0 = time.time()
            if tablas:
                tmp = self._p(etapa, "tmp.npz")
                np.savez(tmp, **{k: v for k, v in tablas.items() if v is not None})
                os.replace(tmp, self._p(etapa, "npz"))
            tmp = self._p(etapa, "json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(datos if datos is not None else {}, f)
            os.replace(tmp, self._p(etapa, "json"))
            self._marcar(etapa, texto)
            self.log(f"  avance guardado: {texto} ({time.time() - t0:.0f} s). Si se cancela, continúa desde aquí.")
        except Exception as e:      # noqa: BLE001
            self.log(f"  (no se pudo guardar el avance de '{texto}': {e})")

    def cargar(self, etapa):
        """(datos, tablas) de una etapa guardada, o None."""
        if not self.tiene(etapa):
            return None
        try:
            with open(self._p(etapa, "json"), encoding="utf-8") as f:
                datos = json.load(f)
            tablas = {}
            if os.path.exists(self._p(etapa, "npz")):
                with np.load(self._p(etapa, "npz")) as z:
                    tablas = {k: z[k] for k in z.files}
            return datos, tablas
        except Exception as e:      # noqa: BLE001 — archivo a medias o dañado: esa etapa se rehace
            self.log(f"  (el avance de '{etapa}' no se pudo leer y se rehace: {e})")
            self.estado["etapas"].pop(etapa, None)
            return None

    def olvidar(self, etapa):
        self.estado["etapas"].pop(etapa, None)

    def borrar(self):
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)


def malla_a_tablas(mesh, prefijo=""):
    import open3d as o3d  # noqa: F401
    out = {prefijo + "V": np.asarray(mesh.vertices), prefijo + "F": np.asarray(mesh.triangles).astype(np.int32)}
    if mesh.has_vertex_colors():
        out[prefijo + "C"] = np.asarray(mesh.vertex_colors).astype(np.float32)
    return out


def tablas_a_malla(tablas, prefijo=""):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tablas[prefijo + "V"]),
                                  o3d.utility.Vector3iVector(tablas[prefijo + "F"]))
    if prefijo + "C" in tablas:
        m.vertex_colors = o3d.utility.Vector3dVector(tablas[prefijo + "C"].astype(np.float64))
    m.compute_vertex_normals()
    return m
