#!/usr/bin/env python3
"""Mide la fidelidad de la textura: vuelve a "tomar" las fotos desde las posiciones reales sobre el modelo
texturizado (OBJ) y las compara con las fotos de entrada. Uso: evtex.py <resultado> <paquete> [n]"""
import sys, os, json, math
import numpy as np, cv2, open3d as o3d
sys.path.insert(0, "/home/claude/levantamiento/tools")
import sintetico as S
res, paq = sys.argv[1], sys.argv[2]
nsel = int(sys.argv[3]) if len(sys.argv) > 3 else 12
V, VT, pages = [], [], {}
cur = None
for line in open(os.path.join(res, "obj", "modelo_texturizado.obj")):
    p = line.split()
    if not p: continue
    if p[0] == "v": V.append(p[1:4])
    elif p[0] == "vt": VT.append(p[1:3])
    elif p[0] == "usemtl": cur = int(p[1].split("_")[1]); pages[cur] = []
    elif p[0] == "f": pages[cur].append([int(x.split("/")[0]) - 1 for x in p[1:4]])
V = np.array(V, float) / 1000.0; VT = np.array(VT, float)
ZtoY = np.linalg.inv(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float))
V = V @ ZtoY.T
F = np.concatenate([np.array(pages[k]) for k in sorted(pages)])
pg = np.concatenate([np.full(len(pages[k]), k) for k in sorted(pages)])
tex = {k: cv2.imread(os.path.join(res, "obj", f"textura_{k}.jpg")) for k in pages}
EW, EH = 1920, 1440
gt = S.build_scene(); gsc = o3d.t.geometry.RaycastingScene(); gsc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
gcol = np.asarray(gt.vertex_colors)[np.asarray(gt.triangles)[:, 0]]
sc = o3d.t.geometry.RaycastingScene()
sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
# posiciones reales (misma semilla que el simulador)
n = len(open(os.path.join(paq, "frames.jsonl")).read().strip().split("\n"))
rng = np.random.default_rng(7)
poses = []
for i in range(n):
    a = -1.1 + 2.2 * i / (n - 1); r = 1.35 + 0.15 * math.sin(i * 0.7); h = 1.15 + 0.35 * (i % 2)
    eye = np.array([1.6 * math.sin(a), h, 0.15 + r * math.cos(a)])
    target = np.array([0.9 * math.sin(a * 1.3), 0.95, -0.15]) + rng.normal(0, 0.05, 3)
    poses.append(S.look_at_arkit(eye, target))
ps, sh, cov, mae = [], [], [], []
for i in np.linspace(3, n - 4, nsel).astype(int):
    rays = S.pixel_rays(poses[i], EW, EH)
    hit = sc.cast_rays(o3d.core.Tensor(rays))
    ids = hit["primitive_ids"].numpy(); ok = ids != sc.INVALID_ID
    uvb = hit["primitive_uvs"].numpy()
    f = F[ids[ok]]; b1, b2 = uvb[ok, 0], uvb[ok, 1]; b0 = 1 - b1 - b2
    uv = VT[f[:, 0]] * b0[:, None] + VT[f[:, 1]] * b1[:, None] + VT[f[:, 2]] * b2[:, None]
    out = np.zeros((EH * EW, 3), np.float32)
    pp = pg[ids[ok]]; idx = np.nonzero(ok)[0]
    for k in tex:
        m = pp == k
        if not m.any(): continue
        T = tex[k]; H, W = T.shape[:2]
        mx = np.clip(uv[m, 0] * W - 0.5, 0, W - 1.001); my = np.clip((1 - uv[m, 1]) * H - 0.5, 0, H - 1.001)
        x0 = mx.astype(int); y0 = my.astype(int); ax = (mx - x0)[:, None]; ay = (my - y0)[:, None]
        out[idx[m]] = (T[y0, x0] * (1 - ax) + T[y0, x0 + 1] * ax) * (1 - ay) + (T[y0 + 1, x0] * (1 - ax) + T[y0 + 1, x0 + 1] * ax) * ay
    ren = out.reshape(EH, EW, 3)
    foto = S.render_color(gsc, gcol, poses[i], EW, EH)[..., ::-1].astype(np.float32)
    m2 = cv2.erode(ok.reshape(EH, EW).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    d = (ren - foto)[m2]
    ps.append(10 * math.log10(255 ** 2 / np.mean(d ** 2))); mae.append(np.abs(d).mean())
    g = lambda im: np.abs(cv2.Laplacian(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F))[m2].mean()
    sh.append(g(ren) / g(foto)); cov.append(ok.mean())
    if len(sys.argv) > 4: cv2.imwrite(f"{sys.argv[4]}_{i:03d}.jpg", np.hstack([ren, foto]).astype(np.uint8))
print(f"{res:14s} PSNR {np.mean(ps):5.2f} dB  error medio {np.mean(mae):5.2f}/255  nitidez {np.mean(sh):4.2f}  cubre {np.mean(cov)*100:4.1f} %")
