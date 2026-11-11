#!/usr/bin/env python3
"""
Modelo con fotos para programas de CAD, en el mismo sistema que el STL (milímetros, Z arriba):

  - FBX (texto, 7.4) con texturas: Navisworks lo abre con sus fotos (el OBJ y el 3DS no las muestran).
  - 3MF con texturas: SolidWorks 2021+ lo importa como cuerpo de gráficos con color y textura.

`parts` es una lista de (página, vértices_mm (N×3), caras (M×3), uv (N×2), archivo_jpg):
cada página de textura es una malla con su propia imagen.
"""
import os
import zipfile

import numpy as np


# ---------------------------------------------------------------- FBX (Navisworks)

def _arr(vals, fmt, per_line=2000):
    """Arreglo FBX en texto: varias líneas para no hacerlas enormes."""
    flat = np.asarray(vals).reshape(-1)
    lines = []
    for i in range(0, len(flat), per_line):
        lines.append(",".join(fmt % v for v in flat[i:i + per_line].tolist()))
    return len(flat), ",\n\t\t\t\t".join(lines)


def write_fbx(path, parts):
    """FBX 7.4 en texto. Las texturas van como archivos .jpg junto al .fbx (mismo nombre relativo)."""
    out = []
    w = out.append
    w("; FBX 7.4.0 project file")
    w("; Levantamiento 3D: modelo con fotos (milímetros, Z arriba)")
    w("FBXHeaderExtension:  {")
    w("\tFBXHeaderVersion: 1003")
    w("\tFBXVersion: 7400")
    w('\tCreator: "Levantamiento 3D"')
    w("}")
    w("GlobalSettings:  {")
    w("\tVersion: 1000")
    w("\tProperties70:  {")
    for name, typ, val in (("UpAxis", "int", 2), ("UpAxisSign", "int", 1), ("FrontAxis", "int", 1),
                           ("FrontAxisSign", "int", -1), ("CoordAxis", "int", 0), ("CoordAxisSign", "int", 1),
                           ("OriginalUpAxis", "int", 2), ("OriginalUpAxisSign", "int", 1)):
        w(f'\t\tP: "{name}", "{typ}", "Integer", "",{val}')
    # unidad de FBX = cm; 0.1 => 1 unidad = 1 mm
    w('\t\tP: "UnitScaleFactor", "double", "Number", "",0.1')
    w('\t\tP: "OriginalUnitScaleFactor", "double", "Number", "",0.1')
    w("\t}")
    w("}")
    n = len(parts)
    # Un solo objeto "Escaneo" que agrupa todas las páginas, centrado en el origen (X,Y al centro, piso en Z=0):
    # en Navisworks se selecciona completo y se puede mover/girar como una sola pieza.
    allv = np.vstack([p_[1] for p_ in parts]) if parts else np.zeros((1, 3))
    lo, hi = allv.min(0), allv.max(0)
    centro = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    parts = [(pg, v - centro, f, uv, tex) for pg, v, f, uv, tex in parts]
    gid_raiz = 999_999
    w("Definitions:  {")
    w("\tVersion: 100")
    w(f"\tCount: {2 + 5 * n}")
    w('\tObjectType: "GlobalSettings" {')
    w("\t\tCount: 1")
    w("\t}")
    for t in ("Model", "Geometry", "Material", "Texture", "Video"):
        w(f'\tObjectType: "{t}" {{')
        w(f"\t\tCount: {n + 1 if t == 'Model' else n}")
        w("\t}")
    w("}")
    w("Objects:  {")
    conns = [f'\tC: "OO",{gid_raiz},0']
    w(f'\tModel: {gid_raiz}, "Model::Escaneo", "Null" {{')
    w("\t\tVersion: 232")
    w("\t\tProperties70:  {")
    w('\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0,0')
    w("\t\t}")
    w("\t}")
    for k, (pg, verts, faces, uv, tex) in enumerate(parts):
        gid, mid, matid, tid, vid = (1_000_000 + k * 10 + j for j in range(5))
        name = f"pagina_{pg}"
        poly = faces.astype(np.int64).copy()
        poly[:, 2] = -poly[:, 2] - 1                       # fin de cada polígono: -(índice+1)
        nv, sv = _arr(verts, "%.2f")
        npi, spi = _arr(poly, "%d")
        nuv, suv = _arr(uv, "%.6f")
        nui, sui = _arr(faces, "%d")
        w(f'\tGeometry: {gid}, "Geometry::{name}", "Mesh" {{')
        w(f"\t\tVertices: *{nv} {{")
        w(f"\t\t\ta: {sv}")
        w("\t\t}")
        w(f"\t\tPolygonVertexIndex: *{npi} {{")
        w(f"\t\t\ta: {spi}")
        w("\t\t}")
        w("\t\tGeometryVersion: 124")
        w("\t\tLayerElementUV: 0 {")
        w("\t\t\tVersion: 101")
        w('\t\t\tName: "UVMap"')
        w('\t\t\tMappingInformationType: "ByPolygonVertex"')
        w('\t\t\tReferenceInformationType: "IndexToDirect"')
        w(f"\t\t\tUV: *{nuv} {{")
        w(f"\t\t\t\ta: {suv}")
        w("\t\t\t}")
        w(f"\t\t\tUVIndex: *{nui} {{")
        w(f"\t\t\t\ta: {sui}")
        w("\t\t\t}")
        w("\t\t}")
        w("\t\tLayerElementMaterial: 0 {")
        w("\t\t\tVersion: 101")
        w('\t\t\tName: ""')
        w('\t\t\tMappingInformationType: "AllSame"')
        w('\t\t\tReferenceInformationType: "IndexToDirect"')
        w("\t\t\tMaterials: *1 {")
        w("\t\t\t\ta: 0")
        w("\t\t\t}")
        w("\t\t}")
        w("\t\tLayer: 0 {")
        w("\t\t\tVersion: 100")
        w("\t\t\tLayerElement:  {")
        w('\t\t\t\tType: "LayerElementMaterial"')
        w("\t\t\t\tTypedIndex: 0")
        w("\t\t\t}")
        w("\t\t\tLayerElement:  {")
        w('\t\t\t\tType: "LayerElementUV"')
        w("\t\t\t\tTypedIndex: 0")
        w("\t\t\t}")
        w("\t\t}")
        w("\t}")
        w(f'\tModel: {mid}, "Model::{name}", "Mesh" {{')
        w("\t\tVersion: 232")
        w("\t\tProperties70:  {")
        w('\t\t\tP: "DefaultAttributeIndex", "int", "Integer", "",0')
        w("\t\t}")
        w("\t\tShading: T")
        w('\t\tCulling: "CullingOff"')
        w("\t}")
        w(f'\tMaterial: {matid}, "Material::{name}", "" {{')
        w("\t\tVersion: 102")
        w('\t\tShadingModel: "lambert"')
        w("\t\tMultiLayer: 0")
        w("\t\tProperties70:  {")
        w('\t\t\tP: "DiffuseColor", "Color", "", "A",1,1,1')
        w('\t\t\tP: "DiffuseFactor", "Number", "", "A",1')
        w('\t\t\tP: "AmbientColor", "Color", "", "A",1,1,1')
        w('\t\t\tP: "EmissiveColor", "Color", "", "A",0,0,0')
        w('\t\t\tP: "Opacity", "double", "Number", "",1')
        w("\t\t}")
        w("\t}")
        w(f'\tVideo: {vid}, "Video::{name}", "Clip" {{')
        w('\t\tType: "Clip"')
        w("\t\tProperties70:  {")
        w(f'\t\t\tP: "Path", "KString", "XRefUrl", "", "{tex}"')
        w("\t\t}")
        w("\t\tUseMipMap: 0")
        w(f'\t\tFilename: "{tex}"')
        w(f'\t\tRelativeFilename: "{tex}"')
        w("\t}")
        w(f'\tTexture: {tid}, "Texture::{name}", "" {{')
        w('\t\tType: "TextureVideoClip"')
        w("\t\tVersion: 202")
        w(f'\t\tTextureName: "Texture::{name}"')
        w("\t\tProperties70:  {")
        w('\t\t\tP: "UVSet", "KString", "", "", "UVMap"')
        w('\t\t\tP: "UseMaterial", "bool", "", "",1')
        w("\t\t}")
        w(f'\t\tMedia: "Video::{name}"')
        w(f'\t\tFileName: "{tex}"')
        w(f'\t\tRelativeFilename: "{tex}"')
        w("\t\tModelUVTranslation: 0,0")
        w("\t\tModelUVScaling: 1,1")
        w('\t\tTexture_Alpha_Source: "None"')
        w("\t\tCropping: 0,0,0,0")
        w("\t}")
        conns += [f'\tC: "OO",{mid},{gid_raiz}', f'\tC: "OO",{gid},{mid}', f'\tC: "OO",{matid},{mid}',
                  f'\tC: "OP",{tid},{matid}, "DiffuseColor"', f'\tC: "OO",{vid},{tid}']
    w("}")
    w("Connections:  {")
    out.extend(conns)
    w("}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")


# ---------------------------------------------------------------- 3MF (SolidWorks)

_CT = ('<?xml version="1.0" encoding="UTF-8"?>\n'
       '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
       '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
       '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
       '<Default Extension="jpg" ContentType="image/jpeg"/></Types>')
_RELS = ('<?xml version="1.0" encoding="UTF-8"?>\n'
         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
         'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')


def write_3mf(path, parts, tex_dir):
    """3MF en milímetros con una malla y una textura por página (las imágenes van dentro del archivo)."""
    tmp = path + ".parte"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", _CT)
        z.writestr("_rels/.rels", _RELS)
        rels = ['<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
        with z.open("3D/3dmodel.model", "w", force_zip64=True) as f:
            def put(s):
                f.write(s.encode("utf-8"))
            put('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<model unit="millimeter" xml:lang="es-MX" '
                'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
                'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">\n'
                '<metadata name="Application">Levantamiento 3D</metadata>\n<resources>\n')
            items = []
            rid = 1
            for k, (pg, verts, faces, uv, tex) in enumerate(parts):
                tex_id, grp_id, obj_id = rid, rid + 1, rid + 2
                rid += 3
                tpath = f"/3D/Textures/{tex}"
                rels.append(f'<Relationship Target="{tpath}" Id="tex{k}" '
                            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dtexture"/>')
                put(f'<m:texture2d id="{tex_id}" path="{tpath}" contenttype="image/jpeg" '
                    'tilestyleu="clamp" tilestylev="clamp" filter="auto"/>\n')
                put(f'<m:texture2dgroup id="{grp_id}" texid="{tex_id}">\n')
                for i in range(0, len(uv), 200_000):
                    put("".join(f'<m:tex2coord u="{a:.6f}" v="{b:.6f}"/>' for a, b in uv[i:i + 200_000].tolist()))
                put("\n</m:texture2dgroup>\n")
                put(f'<object id="{obj_id}" type="model" name="pagina_{pg}">\n<mesh>\n<vertices>\n')
                for i in range(0, len(verts), 200_000):
                    put("".join(f'<vertex x="{x:.2f}" y="{y:.2f}" z="{zz:.2f}"/>'
                                for x, y, zz in verts[i:i + 200_000].tolist()))
                put("\n</vertices>\n<triangles>\n")
                for i in range(0, len(faces), 200_000):
                    put("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="{grp_id}" p1="{a}" p2="{b}" p3="{c}"/>'
                                for a, b, c in faces[i:i + 200_000].tolist()))
                put("\n</triangles>\n</mesh>\n</object>\n")
                items.append(obj_id)
            put("</resources>\n<build>\n" + "".join(f'<item objectid="{o}"/>' for o in items) + "\n</build>\n</model>\n")
        rels.append("</Relationships>")
        z.writestr("3D/_rels/3dmodel.model.rels", "".join(rels))
        for pg, _v, _f, _u, tex in parts:
            z.write(os.path.join(tex_dir, tex), f"3D/Textures/{tex}", compress_type=zipfile.ZIP_STORED)
    os.replace(tmp, path)
