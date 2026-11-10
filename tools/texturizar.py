#!/usr/bin/env python3
"""
Texturizado fotográfico: pega las fotos de los cuadros RGB-D sobre la malla fusionada.

Para cada triángulo se elige la foto que lo ve mejor (de frente, de cerca, nítida y sin
obstrucciones), se suavizan las elecciones entre vecinos para reducir costuras, se recortan las
zonas usadas de cada foto y se empaquetan en páginas de textura (atlas). Exporta:
  modelo_texturizado.glb          glTF binario, metros, Y arriba (visores, web, Blender)
  obj/modelo_texturizado.obj      OBJ + MTL + JPG, metros, Z arriba
"""
import io
import math
import os
import time

import cv2
import numpy as np
import open3d as o3d

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def _log(msg):
    print(msg, flush=True)


# avance (0–1) de esta etapa; procesar_rgbd lo conecta a la barra del programa
AVANCE = None


def _avance(f, texto):
    if AVANCE is not None:
        AVANCE(f, texto)


def _face_adjacency(F):
    """Pares de caras que comparten arista."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    fid = np.tile(np.arange(len(F)), 3)
    key = e[:, 0].astype(np.int64) * (int(F.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    k, f = key[order], fid[order]
    same = k[1:] == k[:-1]
    return np.stack([f[:-1][same], f[1:][same]], 1)


def _shelf_pack(sizes, page):
    """Empaqueta rectángulos (w, h) en páginas de page×page. Devuelve (página, x, y) por rectángulo."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    out = [None] * len(sizes)
    pg, x, y, shelf_h = 0, 0, 0, 0
    for i in order:
        w, h = sizes[i]
        if x + w > page:
            x, y, shelf_h = 0, y + shelf_h, 0
        if y + h > page:
            pg, x, y, shelf_h = pg + 1, 0, 0, 0
        out[i] = (pg, x, y)
        x += w
        shelf_h = max(shelf_h, h)
    return out, pg + 1


def texture_mesh(mesh, frames, out_dir, dmax=3.0, target_triangles=700_000,
                 page_size=4096, zbuf_scale=2, smooth_iters=6, to_z_up=None, scale=1.0,
                 harmonize=True, texture_scale=1.0, consume=False):
    """
    mesh: TriangleMesh de Open3D en el mundo de ARKit (Y arriba, metros).
    frames: lista de cuadros (dicts con jpg, fx, fy, cx, cy, w, h, T) como en procesar_rgbd.
    to_z_up: matriz 4×4 opcional para el OBJ (Z arriba).
    """
    t0 = time.time()
    _avance(0.0, "Simplificando la malla para las fotos")
    # consume=True: la malla es solo para esto y se puede soltar (ahorra memoria con mallas grandes)
    m = mesh if consume else o3d.geometry.TriangleMesh(mesh)
    del mesh
    if len(m.triangles) > target_triangles:
        m = m.simplify_quadric_decimation(target_triangles)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    m.compute_triangle_normals()
    V = np.asarray(m.vertices)
    F = np.asarray(m.triangles)
    N = np.asarray(m.triangle_normals)
    VC = np.asarray(m.vertex_colors) if m.has_vertex_colors() else np.full((len(V), 3), 0.6)
    C = V[F].mean(1)
    nF = len(F)
    _log(f"Texturizado: {nF:,} triángulos, {len(frames)} fotos candidatas")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

    K = len(frames)
    best_score = np.zeros(nF)
    best_frame = np.full(nF, -1, np.int32)
    second_score = np.zeros(nF)
    second_frame = np.full(nF, -1, np.int32)
    proj_cache = {}

    _avance(0.38, "Revisando la nitidez de las fotos")
    sharp = []
    for fr in frames:
        g = cv2.imread(fr["jpg"], cv2.IMREAD_GRAYSCALE)
        sharp.append(cv2.Laplacian(cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2)), cv2.CV_64F).var() if g is not None else 0)
    sharp = np.array(sharp)
    # la nitidez pesa fuerte: una foto movida casi nunca gana si hay otra nítida
    sharp_w = np.clip((sharp / (np.median(sharp) + 1e-9)) ** 1.5, 0.1, 2.5)

    for k, fr in enumerate(frames):
        w, h = int(fr["w"]), int(fr["h"])
        fx, fy, cx, cy = fr["fx"], fr["fy"], fr["cx"], fr["cy"]
        Tc = np.array(fr["T"], float).reshape(4, 4).T @ ARKIT_TO_CV      # cámara (CV) -> mundo
        R, t = Tc[:3, :3], Tc[:3, 3]

        # z-buffer reducido desde esta cámara
        zw, zh = w // zbuf_scale, h // zbuf_scale
        uu, vv = np.meshgrid(np.arange(zw), np.arange(zh))
        uf = (uu + 0.5) * zbuf_scale - 0.5
        vf = (vv + 0.5) * zbuf_scale - 0.5
        dcam = np.stack([(uf - cx) / fx, (vf - cy) / fy, np.ones_like(uf)], -1).reshape(-1, 3)
        dw_ = dcam @ R.T
        rays = np.concatenate([np.broadcast_to(t, dw_.shape), dw_], 1).astype(np.float32)
        zbuf = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(zh, zw)

        # proyección de centroides
        Pc = (C - t) @ R
        z = Pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * Pc[:, 0] / z + cx
            v = fy * Pc[:, 1] / z + cy
        margin = 3
        ok = (z > 0.15) & (z < dmax) & (u > margin) & (u < w - 1 - margin) & (v > margin) & (v < h - 1 - margin)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        zu = np.clip(((u[idx] + 0.5) / zbuf_scale - 0.5).round().astype(int), 0, zw - 1)
        zv = np.clip(((v[idx] + 0.5) / zbuf_scale - 0.5).round().astype(int), 0, zh - 1)
        zb = zbuf[zv, zu]
        vis = np.abs(zb - z[idx]) < np.maximum(0.012, 0.012 * z[idx])
        idx = idx[vis]
        if len(idx) == 0:
            continue
        # vértices dentro de la imagen y visibles (sin tubos u objetos delante)
        Pv = (V[F[idx]].reshape(-1, 3) - t) @ R
        zvv = Pv[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv_u = fx * Pv[:, 0] / zvv + cx
            uv_v = fy * Pv[:, 1] / zvv + cy
        inside = (zvv > 0.1) & (uv_u > 1) & (uv_u < w - 2) & (uv_v > 1) & (uv_v < h - 2)
        qu = np.clip(np.nan_to_num((uv_u + 0.5) / zbuf_scale - 0.5).round().astype(int), 0, zw - 1)
        qv = np.clip(np.nan_to_num((uv_v + 0.5) / zbuf_scale - 0.5).round().astype(int), 0, zh - 1)
        vz = zbuf[qv, qu]
        vvis = (np.abs(vz - zvv) < np.maximum(0.012, 0.01 * np.abs(zvv))).reshape(-1, 3).all(1)
        inside = inside.reshape(-1, 3).all(1)
        idx = idx[inside]
        vvis = vvis[inside]
        if len(idx) == 0:
            continue
        view = C[idx] - t
        dist = np.linalg.norm(view, axis=1)
        cosang = np.abs(np.sum(N[idx] * view, 1)) / dist
        # de frente, cerca, y lejos del borde de la foto (menos distorsión)
        edge = np.minimum.reduce([u[idx], w - u[idx], v[idx], h - v[idx]]) / (0.5 * min(w, h))
        score = (cosang ** 2) / (dist ** 1.5) * sharp_w[k] * np.clip(edge, 0.25, 1.0)
        # si algún vértice queda tapado, la foto sólo sirve como último recurso
        score = np.where(vvis, score, score * 0.02)
        better = score > best_score[idx]
        # el mejor anterior pasa a segundo
        ii = idx[better]
        second_score[ii] = best_score[ii]
        second_frame[ii] = best_frame[ii]
        best_score[ii] = score[better]
        best_frame[ii] = k
        jj = idx[~better]
        sec = score[~better] > second_score[jj]
        second_score[jj[sec]] = score[~better][sec]
        second_frame[jj[sec]] = k
        if k % 5 == 0:
            _avance(0.40 + 0.22 * (k + 1) / K, f"Eligiendo la mejor foto para cada zona ({k + 1} de {K})")
        if (k + 1) % 50 == 0:
            _log(f"  {k + 1}/{K} fotos evaluadas ({time.time() - t0:.0f} s)")

    labels = best_frame.copy()
    _log(f"  con foto: {(labels >= 0).mean() * 100:.1f} % de los triángulos")

    # suavizado de etiquetas: cambiar a la etiqueta mayoritaria de los vecinos si esa foto también
    # ve bien el triángulo (es su mejor o segunda mejor opción), para hacer parches grandes
    _avance(0.62, "Uniendo zonas con la misma foto")
    adj = _face_adjacency(F)
    for _ in range(smooth_iters):
        a, b = adj[:, 0], adj[:, 1]
        changed = 0
        for src, dst in ((a, b), (b, a)):
            nb_label = labels[src]
            cand = (second_frame[dst] == nb_label) & (labels[dst] != nb_label) & (nb_label >= 0) \
                & (second_score[dst] > 0.6 * best_score[dst])
            # contar votos por (cara, etiqueta)
            if cand.any():
                d = dst[cand]
                cnt = np.bincount(d, minlength=nF)
                flip = cnt[d] >= 2
                labels[d[flip]] = nb_label[cand][flip]
                changed += int(flip.sum())
        if changed == 0:
            break

    # --- igualación de exposición/balance entre fotos
    gains = np.ones((K, 3))
    if harmonize:
        gains = _harmonize_gains(frames, C, labels, second_frame, K)

    # --- UV de cada triángulo en su foto
    used = np.unique(labels[labels >= 0])
    corner_uv = np.zeros((nF, 3, 2))
    for k in used:
        fr = frames[k]
        fx, fy, cx, cy = fr["fx"], fr["fy"], fr["cx"], fr["cy"]
        Tc = np.array(fr["T"], float).reshape(4, 4).T @ ARKIT_TO_CV
        R, t = Tc[:3, :3], Tc[:3, 3]
        sel = np.nonzero(labels == k)[0]
        Pv = (V[F[sel]].reshape(-1, 3) - t) @ R
        corner_uv[sel] = np.stack([fx * Pv[:, 0] / Pv[:, 2] + cx, fy * Pv[:, 1] / Pv[:, 2] + cy], 1).reshape(-1, 3, 2)

    # --- parches: grupos de triángulos vecinos que usan la misma foto. Solo se copia a la textura
    # el pedazo de foto que cada parche usa (no la foto entera): menos memoria y carga más rápida.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a, b = adj[:, 0], adj[:, 1]
    same = (labels[a] == labels[b]) & (labels[a] >= 0)
    G = coo_matrix((np.ones(int(same.sum()), bool), (a[same], b[same])), shape=(nF, nF))
    _, comp = connected_components(G, directed=False)
    lab_faces = np.nonzero(labels >= 0)[0]
    _, chart_of = np.unique(comp[lab_faces], return_inverse=True)      # parche de cada cara con foto
    nC = int(chart_of.max()) + 1 if len(lab_faces) else 0
    PAD = 3
    cmin = np.full((nC, 2), np.inf)
    cmax = np.full((nC, 2), -np.inf)
    uvf = corner_uv[lab_faces]
    np.minimum.at(cmin, chart_of, uvf.min(1))
    np.maximum.at(cmax, chart_of, uvf.max(1))
    chart_label = np.zeros(nC, np.int32)
    chart_label[chart_of] = labels[lab_faces]
    wh = np.array([[int(frames[k]["w"]), int(frames[k]["h"])] for k in range(K)])
    lo = np.maximum(np.floor(cmin).astype(int) - PAD, 0)
    hi = np.minimum(np.ceil(cmax).astype(int) + PAD, wh[chart_label] - 1)
    ts = float(texture_scale)
    csize = np.maximum(1, np.round((hi - lo + 1) * ts).astype(int))
    place, n_pages = _shelf_pack([tuple(x) for x in csize.tolist()], page_size)
    place = np.array(place, int).reshape(-1, 3)

    # triángulos sin foto: cada uno recibe un cuadrito con su propio color (del LiDAR), no un gris único
    none = np.nonzero(labels < 0)[0]
    CELL = 4
    cols_pp = page_size // CELL
    per_page = cols_pp * cols_pp
    n_color_pages = int(math.ceil(len(none) / per_page)) if len(none) else 0
    pages = [np.full((page_size, page_size, 3), 128, np.uint8) for _ in range(n_pages + n_color_pages)]
    page_of = np.zeros(nF, np.int32)
    face_uv = np.zeros((nF, 3, 2))

    _avance(0.66, "Armando la textura con las fotos")
    order = np.argsort(chart_label, kind="stable")
    bounds = np.searchsorted(chart_label[order], np.arange(K + 1))
    for j, k in enumerate(used):
        if j % 10 == 0:
            _avance(0.66 + 0.09 * j / max(1, len(used)), "Armando la textura con las fotos")
        img = cv2.imread(frames[k]["jpg"], cv2.IMREAD_COLOR)
        if harmonize:
            img = np.clip(img.astype(np.float32) * gains[k][::-1], 0, 255).astype(np.uint8)
        for c in order[bounds[k]:bounds[k + 1]]:
            crop = img[lo[c, 1]:hi[c, 1] + 1, lo[c, 0]:hi[c, 0] + 1]
            if ts != 1.0:
                crop = cv2.resize(crop, (int(csize[c, 0]), int(csize[c, 1])), interpolation=cv2.INTER_AREA)
            pg, px, py = place[c]
            pages[pg][py:py + crop.shape[0], px:px + crop.shape[1]] = crop
    if nC:
        lo_f, pl_f = lo[chart_of], place[chart_of]
        cu = (uvf[..., 0] - lo_f[:, None, 0] + 0.5) * ts + pl_f[:, None, 1]
        cv_ = (uvf[..., 1] - lo_f[:, None, 1] + 0.5) * ts + pl_f[:, None, 2]
        face_uv[lab_faces] = np.stack([cu / page_size, 1.0 - cv_ / page_size], -1)
        page_of[lab_faces] = pl_f[:, 0]
    if len(none):
        col = (np.clip(VC[F[none]].mean(1), 0, 1) * 255).astype(np.uint8)[:, ::-1]      # BGR
        n = np.arange(len(none))
        pg = n_pages + n // per_page
        gx = (n % per_page) % cols_pp
        gy = (n % per_page) // cols_pp
        for q in range(n_color_pages):
            m = pg == n_pages + q
            cell = np.full((cols_pp, cols_pp, 3), 128, np.uint8)
            cell[gy[m], gx[m]] = col[m]
            pages[n_pages + q][:] = np.repeat(np.repeat(cell, CELL, 0), CELL, 1)
        cuv = np.stack([(gx * CELL + CELL / 2) / page_size, 1.0 - (gy * CELL + CELL / 2) / page_size], -1)
        face_uv[none] = cuv[:, None, :]
        page_of[none] = pg
    n_photo_pages = n_pages
    n_pages = len(pages)
    # recortar la última página de color a lo usado no hace falta: JPEG la comprime casi a nada

    os.makedirs(out_dir, exist_ok=True)
    _avance(0.75, "Guardando el modelo con fotos")
    _export(V * scale, F, face_uv, page_of, pages, out_dir, to_z_up)
    _log(f"  {len(used)} fotos usadas en {nC:,} parches, {n_photo_pages} página(s) de fotos + "
         f"{n_color_pages} de color, {page_size}², {time.time() - t0:.0f} s")
    return {"triangles": nF, "photos_used": int(len(used)), "pages": n_pages,
            "coverage": float((labels >= 0).mean())}


def _project(fr, P):
    Tc = np.array(fr["T"], float).reshape(4, 4).T @ ARKIT_TO_CV
    Pc = (P - Tc[:3, 3]) @ Tc[:3, :3]
    z = np.maximum(Pc[:, 2], 1e-6)
    return fr["fx"] * Pc[:, 0] / z + fr["cx"], fr["fy"] * Pc[:, 1] / z + fr["cy"]


def _harmonize_gains(frames, C, labels, second, K, max_pairs=400_000):
    """Ganancia RGB por foto para que una misma superficie tenga el mismo color en todas.
    Usa pares (mejor foto, segunda mejor) del mismo triángulo y resuelve mínimos cuadrados en log."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    pairs = np.nonzero((labels >= 0) & (second >= 0) & (second != labels))[0]
    if len(pairs) < 1000:
        return np.ones((K, 3))
    rng = np.random.default_rng(0)
    if len(pairs) > max_pairs:
        pairs = rng.choice(pairs, max_pairs, replace=False)
    colA = np.zeros((len(pairs), 3))
    colB = np.zeros((len(pairs), 3))
    la, lb = labels[pairs], second[pairs]
    for k in range(K):
        ia = np.nonzero(la == k)[0]
        ib = np.nonzero(lb == k)[0]
        if len(ia) == 0 and len(ib) == 0:
            continue
        img = cv2.imread(frames[k]["jpg"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.blur(img, (9, 9)).astype(np.float64)[..., ::-1]
        h, w = img.shape[:2]
        for sel, out in ((ia, colA), (ib, colB)):
            if len(sel) == 0:
                continue
            u, v = _project(frames[k], C[pairs[sel]])
            u = np.clip(np.round(u).astype(int), 0, w - 1)
            v = np.clip(np.round(v).astype(int), 0, h - 1)
            out[sel] = img[v, u]
    ok = ((colA > 12) & (colA < 245) & (colB > 12) & (colB < 245)).all(1)
    la, lb, colA, colB = la[ok], lb[ok], colA[ok], colB[ok]
    n = len(la)
    if n < 1000:
        return np.ones((K, 3))
    lam = 0.05 * math.sqrt(n / K)
    rows = np.r_[np.arange(n), np.arange(n), n + np.arange(K)]
    cols = np.r_[la, lb, np.arange(K)]
    vals = np.r_[np.ones(n), -np.ones(n), np.full(K, lam)]
    A = coo_matrix((vals, (rows, cols)), shape=(n + K, K)).tocsr()
    gains = np.ones((K, 3))
    for ch in range(3):
        b = np.r_[np.log(colB[:, ch]) - np.log(colA[:, ch]), np.zeros(K)]
        g = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=2000)[0]
        gains[:, ch] = np.exp(np.clip(g, -0.5, 0.5))
    _log(f"  igualación de color: {n:,} pares, ganancia p5–p95 "
         f"{np.percentile(gains, 5):.2f}–{np.percentile(gains, 95):.2f}")
    return gains


def _export(V, F, face_uv, page_of, pages, out_dir, to_z_up):
    import trimesh
    from PIL import Image

    # GLB (Y arriba, como glTF): una malla por página
    scene = trimesh.Scene()
    obj_dir = os.path.join(out_dir, "obj")
    os.makedirs(obj_dir, exist_ok=True)
    Vz = V if to_z_up is None else (np.c_[V, np.ones(len(V))] @ to_z_up.T)[:, :3]
    obj_lines = ["mtllib modelo_texturizado.mtl"]
    mtl_lines = []
    usd_parts = []
    cad_parts = []          # para FBX (Navisworks) y 3MF (SolidWorks): mm, Z arriba
    v_off = 0
    for pg, img in enumerate(pages):
        sel = np.nonzero(page_of == pg)[0]
        if len(sel) == 0:
            continue
        # vértices únicos por (vértice original, uv)
        corners = F[sel].reshape(-1)
        uvs = face_uv[sel].reshape(-1, 2)
        key = np.c_[corners, np.round(uvs * 1e6).astype(np.int64)]
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        verts = V[uk[:, 0].astype(int)]
        uv_unique = np.zeros((len(uk), 2))
        uv_unique[inv] = uvs
        faces = inv.reshape(-1, 3)
        ok_, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rgb = Image.open(io.BytesIO(enc.tobytes()))          # conserva formato JPEG dentro del GLB
        rgb.load()
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=rgb, metallicFactor=0.0, roughnessFactor=1.0,
                                                  doubleSided=True)
        tm = trimesh.Trimesh(vertices=verts, faces=faces,
                             visual=trimesh.visual.TextureVisuals(uv=uv_unique, material=mat), process=False)
        scene.add_geometry(tm, geom_name=f"pagina_{pg}")
        usd_parts.append((pg, verts, faces, uv_unique))

        # OBJ (Z arriba, milímetros): Navisworks lo abre con sus fotos
        tex_name = f"textura_{pg}.jpg"
        cv2.imwrite(os.path.join(obj_dir, tex_name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        mtl_lines += [f"newmtl pagina_{pg}", "Ka 1 1 1", "Kd 1 1 1", "Ks 0 0 0", "d 1", "illum 1",
                      f"map_Kd {tex_name}", ""]
        vz = Vz[uk[:, 0].astype(int)]
        good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
        cad_parts.append((pg, vz * 1000.0, faces[good], uv_unique, tex_name))
        obj_lines.append(f"o pagina_{pg}")
        obj_lines += [f"v {p[0]:.2f} {p[1]:.2f} {p[2]:.2f}" for p in vz * 1000.0]   # milímetros, como el STL
        obj_lines += [f"vt {q[0]:.6f} {q[1]:.6f}" for q in uv_unique]
        obj_lines.append(f"usemtl pagina_{pg}")
        ff = faces + 1 + v_off
        obj_lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in ff]
        v_off += len(verts)

    scene.export(os.path.join(out_dir, "modelo_texturizado.glb"))
    try:
        _export_usdz(usd_parts, obj_dir, os.path.join(out_dir, "modelo_texturizado.usdz"))
    except ImportError:
        _log("  (sin usd-core: no se generó USDZ)")
    with open(os.path.join(obj_dir, "modelo_texturizado.obj"), "w") as f:
        f.write("\n".join(obj_lines) + "\n")
    with open(os.path.join(obj_dir, "modelo_texturizado.mtl"), "w") as f:
        f.write("\n".join(mtl_lines) + "\n")
    del obj_lines
    # mismos datos para Navisworks (FBX con texturas) y SolidWorks (3MF con texturas)
    from formatos_cad import write_3mf, write_fbx
    _avance(0.9, "Guardando el modelo con fotos para Navisworks (FBX)")
    write_fbx(os.path.join(obj_dir, "modelo_navisworks.fbx"), cad_parts)
    _avance(0.95, "Guardando el modelo con fotos para SolidWorks (3MF)")
    write_3mf(os.path.join(out_dir, "modelo_solidworks.3mf"), cad_parts, obj_dir)


def _export_usdz(parts, tex_dir, usdz_path):
    """USDZ para abrir directo en el iPhone (Vista rápida / AR). Metros, Y arriba."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

    usdc = os.path.join(tex_dir, "modelo_texturizado.usdc")
    stage = Usd.Stage.CreateNew(usdc)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Modelo")
    stage.SetDefaultPrim(root.GetPrim())
    for pg, verts, faces, uv in parts:
        mesh = UsdGeom.Mesh.Define(stage, f"/Modelo/pagina_{pg}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(verts, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1), dtype=np.int32)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        lo, hi = verts.min(0), verts.max(0)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]))
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv, dtype=np.float32)))

        mpath = f"/Modelo/Materiales/pagina_{pg}"
        mat = UsdShade.Material.Define(stage, mpath)
        pbr = UsdShade.Shader.Define(stage, mpath + "/PBR")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
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
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    del stage
    if os.path.exists(usdz_path):
        os.remove(usdz_path)
    UsdUtils.CreateNewARKitUsdzPackage(usdc, usdz_path)
    os.remove(usdc)
