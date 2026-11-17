#!/usr/bin/env python3
"""
Convierte el FBX de texto que escribe Levantamiento 3D (obj/modelo_navisworks.fbx, milímetros, Z arriba)
en un GLB para el visor del programa (metros, Y arriba), usando las imágenes textura_*.jpg que están junto
al FBX. Solo entiende el FBX de este programa (formatos_cad.write_fbx), no FBX en general.

Uso:  python fbx_a_glb.py modelo_navisworks.fbx salida.glb
No necesita librerías extra (solo numpy); el GLB se escribe a mano.
"""
import json
import os
import re
import struct
import sys

import numpy as np

_RE_GEOM = re.compile(r'Geometry:\s*\d+,\s*"Geometry::([^"]*)"')
_RE_MODEL_VIDEO = re.compile(r'(Video|Texture):\s*\d+,\s*"(?:Video|Texture)::([^"]*)"')
_RE_REL = re.compile(r'RelativeFilename:\s*"([^"]*)"')


def _leer_fbx(path):
    """Devuelve [(nombre, vértices N×3 mm, caras M×3, uv N×2)] y {nombre: archivo_jpg}."""
    paginas = []
    texturas = {}
    actual = None          # datos de la geometría en curso
    clave = None           # 'V', 'P' o 'U' mientras se lee un arreglo
    trozos = []
    nombre_obj = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for linea in f:
            s = linea.strip()
            if clave is not None:
                if s == "}":
                    txt = "".join(trozos)
                    arr = np.fromstring(txt, dtype=np.float64, sep=",") if txt else np.zeros(0)
                    actual[clave] = arr
                    clave, trozos = None, []
                elif s.startswith("a:"):
                    trozos.append(s[2:].strip())
                else:
                    trozos.append(s)
                continue
            if s.startswith("Geometry:"):
                m = _RE_GEOM.match(s)
                if m:
                    actual = {"nombre": m.group(1)}
                    paginas.append(actual)
                continue
            if s.startswith("Vertices: *") and actual is not None:
                clave = "V"
            elif s.startswith("PolygonVertexIndex: *") and actual is not None:
                clave = "P"
            elif s.startswith("UV: *") and actual is not None:
                clave = "U"
            else:
                m = _RE_MODEL_VIDEO.match(s)
                if m:
                    nombre_obj = m.group(2)
                    actual = None if s.startswith(("Video:", "Texture:")) else actual
                    continue
                m = _RE_REL.match(s)
                if m and nombre_obj and nombre_obj not in texturas:
                    texturas[nombre_obj] = os.path.basename(m.group(1).replace("\\", "/"))
    salida = []
    for p in paginas:
        if "V" not in p or "P" not in p:
            continue
        v = p["V"].reshape(-1, 3)
        poly = p["P"].astype(np.int64).reshape(-1, 3)
        poly[:, 2] = -poly[:, 2] - 1                    # el último índice de cada triángulo viene negado
        uv = p["U"].reshape(-1, 2) if "U" in p and len(p["U"]) == 2 * len(v) else np.zeros((len(v), 2))
        salida.append((p["nombre"], v, poly, uv))
    return salida, texturas


def convertir(fbx, glb, log=print):
    carpeta = os.path.dirname(os.path.abspath(fbx))
    paginas, texturas = _leer_fbx(fbx)
    if not paginas:
        raise RuntimeError("el FBX no tiene mallas que el visor pueda leer")
    log(f"FBX leído: {len(paginas)} página(s)")

    # --- piezas binarias, cada una alineada a 4 bytes
    piezas = []            # (kind, dato) kind: 'np' arreglo o 'file' ruta
    vistas = []            # bufferViews
    accesores = []
    mallas, nodos, materiales, imagenes, texs = [], [], [], [], []
    desplazamiento = 0

    def agregar_np(arr, target):
        nonlocal desplazamiento
        arr = np.ascontiguousarray(arr)
        n = arr.nbytes
        vistas.append({"buffer": 0, "byteOffset": desplazamiento, "byteLength": n, **({"target": target} if target else {})})
        piezas.append(("np", arr))
        desplazamiento += n + (-n) % 4
        return len(vistas) - 1

    def agregar_archivo(ruta):
        nonlocal desplazamiento
        n = os.path.getsize(ruta)
        vistas.append({"buffer": 0, "byteOffset": desplazamiento, "byteLength": n})
        piezas.append(("file", ruta))
        desplazamiento += n + (-n) % 4
        return len(vistas) - 1

    for nombre, v, caras, uv in paginas:
        # mm, Z arriba  ->  m, Y arriba (rotación: x, z, -y; conserva el sentido de las caras)
        pos = np.empty((len(v), 3), dtype=np.float32)
        pos[:, 0] = v[:, 0] / 1000.0
        pos[:, 1] = v[:, 2] / 1000.0
        pos[:, 2] = -v[:, 1] / 1000.0
        coord = np.empty((len(uv), 2), dtype=np.float32)
        coord[:, 0] = uv[:, 0]
        coord[:, 1] = 1.0 - uv[:, 1]                     # glTF tiene el origen de la imagen arriba
        idx = caras.astype(np.uint32).reshape(-1)

        vp = agregar_np(pos, 34962)
        vt = agregar_np(coord, 34962)
        vi = agregar_np(idx, 34963)
        base = len(accesores)
        accesores.append({"bufferView": vp, "componentType": 5126, "count": len(pos), "type": "VEC3",
                          "min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
        accesores.append({"bufferView": vt, "componentType": 5126, "count": len(coord), "type": "VEC2"})
        accesores.append({"bufferView": vi, "componentType": 5125, "count": len(idx), "type": "SCALAR"})

        prim = {"attributes": {"POSITION": base, "TEXCOORD_0": base + 1}, "indices": base + 2, "mode": 4}
        jpg = texturas.get(nombre) or texturas.get(nombre.replace("pagina_", "")) or \
            "textura_" + nombre.replace("pagina_", "") + ".jpg"
        ruta_jpg = os.path.join(carpeta, jpg)
        if os.path.exists(ruta_jpg):
            imagenes.append({"bufferView": agregar_archivo(ruta_jpg), "mimeType": "image/jpeg"})
            texs.append({"sampler": 0, "source": len(imagenes) - 1})
            materiales.append({"name": nombre, "doubleSided": True,
                               "pbrMetallicRoughness": {"baseColorTexture": {"index": len(texs) - 1},
                                                        "metallicFactor": 0.0, "roughnessFactor": 1.0}})
        else:
            log(f"  (falta la imagen {jpg}: esa página se verá gris)")
            materiales.append({"name": nombre, "doubleSided": True,
                               "pbrMetallicRoughness": {"baseColorFactor": [0.7, 0.7, 0.7, 1.0],
                                                        "metallicFactor": 0.0, "roughnessFactor": 1.0}})
        prim["material"] = len(materiales) - 1
        mallas.append({"name": nombre, "primitives": [prim]})
        nodos.append({"name": nombre, "mesh": len(mallas) - 1})

    gltf = {"asset": {"version": "2.0", "generator": "Levantamiento 3D (fbx_a_glb)"},
            "scene": 0, "scenes": [{"nodes": list(range(len(nodos)))}], "nodes": nodos, "meshes": mallas,
            "materials": materiales, "accessors": accesores, "bufferViews": vistas,
            "buffers": [{"byteLength": desplazamiento}]}
    if imagenes:
        gltf["images"] = imagenes
        gltf["textures"] = texs
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + desplazamiento

    tmp = glb + ".parte"
    with open(tmp, "wb") as out:
        out.write(struct.pack("<4sII", b"glTF", 2, total))
        out.write(struct.pack("<I4s", len(js), b"JSON"))
        out.write(js)
        out.write(struct.pack("<I4s", desplazamiento, b"BIN\x00"))
        for kind, dato in piezas:
            if kind == "np":
                out.write(dato.tobytes())
                n = dato.nbytes
            else:
                with open(dato, "rb") as f:
                    b = f.read()
                out.write(b)
                n = len(b)
            out.write(b"\x00" * ((-n) % 4))
    os.replace(tmp, glb)
    log(f"GLB guardado: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    convertir(sys.argv[1], sys.argv[2])
