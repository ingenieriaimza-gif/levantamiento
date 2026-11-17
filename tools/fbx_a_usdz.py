#!/usr/bin/env python3
"""
Deja el modelo terminado listo para verlo en el iPhone ("Revisar 3D" y "Mis escaneos" de la app).

Lee el FBX que entrega Levantamiento 3D (obj/modelo_navisworks.fbx + textura_*.jpg; milímetros, Z arriba)
y escribe en la carpeta de salida:
  modelo.usdz     el mismo modelo con texturas reducidas (metros, Y arriba; mismas coordenadas que el visor
                  de la PC, así las medidas hechas en el iPhone caen en el mismo lugar)
  miniatura.jpg   vista en planta del modelo, para la lista de escaneos del iPhone
  info.json       triángulos, páginas, tamaño; o el motivo si no se pudo

El FBX completo (decenas de MB y texturas de 4096²) cerraba la app del iPhone por memoria. Aquí las texturas
se reducen a ~50 millones de píxeles en total y los modelos de más de 1.3 millones de triángulos no se ofrecen.

Uso:  python fbx_a_usdz.py <modelo_navisworks.fbx> <carpeta de salida>
No carga Open3D (arranca rápido y gasta poca memoria).
"""
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbx_a_glb import _leer_fbx  # noqa: E402

MAX_TRIANGULOS = 1_300_000
PIXELES_TEXTURA = 50e6


def _pagina(nombre):
    d = "".join(ch for ch in nombre if ch.isdigit())
    return int(d) if d else 0


def _a_metros_y_arriba(v_mm):
    pos = np.empty((len(v_mm), 3), dtype=np.float32)
    pos[:, 0] = v_mm[:, 0] / 1000.0
    pos[:, 1] = v_mm[:, 2] / 1000.0
    pos[:, 2] = -v_mm[:, 1] / 1000.0
    return pos


def _miniatura(partes, imagenes, destino, lado=640):
    """Vista en planta (desde arriba, sin el 20 % más alto para que el techo no tape): cada vértice se pinta
    con el color de su textura; los más altos encima."""
    import cv2
    P, C = [], []
    for pg, pos, _f, uv in partes:
        im = imagenes.get(pg)
        if im is None or not len(pos):
            continue
        h, w = im.shape[:2]
        x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        y = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)
        P.append(pos)
        C.append(im[y, x])
    if not P:
        return False
    P, C = np.concatenate(P), np.concatenate(C)
    lo, hi = P.min(0), P.max(0)
    ok = P[:, 1] <= lo[1] + 0.8 * (hi[1] - lo[1])
    P, C = P[ok], C[ok]
    if not len(P):
        return False
    ext = max(hi[0] - lo[0], hi[2] - lo[2], 1e-3)
    esc = (lado - 24) / ext
    ancho = int((hi[0] - lo[0]) * esc) + 24
    alto = int((hi[2] - lo[2]) * esc) + 24
    px = np.clip(((P[:, 0] - lo[0]) * esc).astype(int) + 12, 0, ancho - 1)
    py = np.clip(((P[:, 2] - lo[2]) * esc).astype(int) + 12, 0, alto - 1)
    orden = np.argsort(P[:, 1], kind="stable")            # los más altos se pintan al final
    img = np.zeros((alto, ancho, 3), np.uint8)
    lleno = np.zeros((alto, ancho), np.uint8)
    img[py[orden], px[orden]] = C[orden]
    lleno[py, px] = 255
    for _ in range(2):                                    # tapa los huequitos entre vértices
        vecino = cv2.dilate(img, np.ones((3, 3), np.uint8))
        hueco = lleno == 0
        img[hueco] = vecino[hueco]
        lleno = cv2.dilate(lleno, np.ones((3, 3), np.uint8))
    fondo = np.full_like(img, (29, 24, 20))               # BGR: gris azulado oscuro, como la app
    img = np.where(lleno[..., None] > 0, img, fondo)
    return bool(cv2.imwrite(destino, img, [cv2.IMWRITE_JPEG_QUALITY, 85]))


def _escribir_usdz(partes, carpeta_tex, usdz, trabajo):
    import cv2
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

    n_pag = max(1, len(partes))
    lado = 512
    while lado < 2048 and (lado * 2) ** 2 * n_pag <= PIXELES_TEXTURA:
        lado *= 2
    imagenes = {}
    for pg, _v, _f, _uv in partes:
        im = cv2.imread(os.path.join(carpeta_tex, f"textura_{pg}.jpg"), cv2.IMREAD_COLOR)
        if im is None:
            continue
        if im.shape[1] > lado:
            im = cv2.resize(im, (lado, lado), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(trabajo, f"textura_{pg}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        imagenes[pg] = im
    usdc = os.path.join(trabajo, "modelo.usdc")
    stage = Usd.Stage.CreateNew(usdc)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Modelo")
    stage.SetDefaultPrim(root.GetPrim())
    for pg, verts, faces, uv in partes:
        mesh = UsdGeom.Mesh.Define(stage, f"/Modelo/pagina_{pg}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(verts, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1), dtype=np.int32)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        lo, hi = verts.min(0), verts.max(0)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]))
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv, dtype=np.float32)))

        mpath = f"/Modelo/Materiales/pagina_{pg}"
        mat = UsdShade.Material.Define(stage, mpath)
        pbr = UsdShade.Shader.Define(stage, mpath + "/PBR")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        if pg in imagenes:
            reader = UsdShade.Shader.Define(stage, mpath + "/stReader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex = UsdShade.Shader.Define(stage, mpath + "/Textura")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"textura_{pg}.jpg")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        else:
            pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.7, 0.7, 0.7))
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    del stage
    tmp = usdz + ".parte.usdz"
    if os.path.exists(tmp):
        os.remove(tmp)
    UsdUtils.CreateNewARKitUsdzPackage(usdc, tmp)
    os.replace(tmp, usdz)
    return imagenes, lado


def convertir(fbx, salida, log=print):
    t0 = time.time()
    os.makedirs(salida, exist_ok=True)
    info_path = os.path.join(salida, "info.json")
    info = {"ok": False, "fbx_mtime": int(os.path.getmtime(fbx))}
    trabajo = os.path.join(salida, "_trabajo")
    try:
        paginas, _tex = _leer_fbx(fbx)
        if not paginas:
            raise RuntimeError("el modelo no tiene mallas")
        partes = []
        tri = 0
        for nombre, v, caras, uv in paginas:
            if not len(caras):
                continue
            partes.append((_pagina(nombre), _a_metros_y_arriba(v), caras.astype(np.int32), uv.astype(np.float32)))
            tri += len(caras)
        del paginas
        info["triangulos"] = int(tri)
        if tri > MAX_TRIANGULOS:
            raise RuntimeError(f"el modelo tiene {tri:,} triángulos: es demasiado pesado para abrirlo en el iPhone "
                               f"(máximo {MAX_TRIANGULOS:,}); revísalo en la PC")
        shutil.rmtree(trabajo, ignore_errors=True)
        os.makedirs(trabajo)
        usdz = os.path.join(salida, "modelo.usdz")
        imagenes, lado = _escribir_usdz(partes, os.path.dirname(os.path.abspath(fbx)), usdz, trabajo)
        V = np.concatenate([p[1] for p in partes])
        lo, hi = V.min(0), V.max(0)
        del V
        info.update(ok=True, paginas=len(partes), textura=lado, mb=round(os.path.getsize(usdz) / 1e6, 1),
                    medidas_m=[round(float(hi[0] - lo[0]), 2), round(float(hi[2] - lo[2]), 2),
                               round(float(hi[1] - lo[1]), 2)])
        try:
            info["miniatura"] = _miniatura(partes, imagenes, os.path.join(salida, "miniatura.jpg"))
        except Exception as e:      # noqa: BLE001 — la miniatura es un adorno: no impide entregar el modelo
            info["miniatura"] = False
            log(f"  (sin miniatura: {e})")
        log(f"Modelo para el iPhone: {tri:,} triángulos, {len(partes)} texturas de {lado}², "
            f"{info['mb']} MB ({time.time() - t0:.0f} s)")
    except Exception as e:      # noqa: BLE001
        info["motivo"] = str(e)
        log(f"No se pudo preparar el modelo para el iPhone: {e}")
    finally:
        shutil.rmtree(trabajo, ignore_errors=True)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
    return info


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if convertir(sys.argv[1], sys.argv[2]).get("ok") else 1)
