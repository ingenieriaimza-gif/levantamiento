#!/usr/bin/env python3
"""
Archivo HTML único con la vista foto-realista de un escaneo: se abre con doble clic en Edge o Chrome,
sin la PC ni internet, y permite girar, acercar y medir (distancia y X, Y, Z).

Lleva dentro:
  - la vista foto-realista (Gaussian Splatting) convertida a formato .splat (32 bytes por mancha);
  - la malla exacta del escaneo, oculta, para que cada medida caiga en la superficie real;
  - las medidas guardadas del escaneo;
  - three.js, OrbitControls y gaussian-splats-3d (licencia MIT).

Uso: python exportar_html.py <carpeta del escaneo> [--salida archivo.html] [--max-manchas N]
"""
import argparse
import base64
import json
import math
import os
import sys

import numpy as np
import open3d as o3d

AQUI = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.normpath(os.path.join(AQUI, "..", "pc", "web"))
sys.path.insert(0, AQUI)
from procesar_rgbd import Y_UP_TO_Z_UP  # noqa: E402

SH_C0 = 0.28209479177387814


def leer_ply_splat(path):
    """PLY de Gaussian Splatting (binario, propiedades float) -> dict nombre -> arreglo."""
    with open(path, "rb") as f:
        props, n = [], 0
        line = f.readline()
        if not line.startswith(b"ply"):
            raise ValueError("no es un PLY")
        while True:
            line = f.readline().strip()
            if line.startswith(b"format") and b"binary_little_endian" not in line:
                raise ValueError("solo PLY binario")
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property"):
                t, name = line.split()[1:3]
                if t not in (b"float", b"float32"):
                    raise ValueError(f"propiedad no soportada: {line!r}")
                props.append(name.decode())
            elif line == b"end_header":
                break
        data = np.frombuffer(f.read(n * 4 * len(props)), dtype="<f4").reshape(n, len(props))
    return {p: data[:, i] for i, p in enumerate(props)}


def a_splat(ply, max_manchas):
    """Convierte al formato .splat: xyz, escala, color RGBA y rotación (w,x,y,z) en 32 bytes."""
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float32)
    esc = np.exp(np.stack([ply["scale_0"], ply["scale_1"], ply["scale_2"]], 1)).astype(np.float32)
    rgb = 0.5 + SH_C0 * np.stack([ply["f_dc_0"], ply["f_dc_1"], ply["f_dc_2"]], 1)
    alpha = 1.0 / (1.0 + np.exp(-ply["opacity"]))
    q = np.stack([ply["rot_0"], ply["rot_1"], ply["rot_2"], ply["rot_3"]], 1)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    ok = np.isfinite(xyz).all(1) & np.isfinite(esc).all(1) & (alpha > 0.02)
    xyz, esc, rgb, alpha, q = xyz[ok], esc[ok], rgb[ok], alpha[ok], q[ok]
    # las más importantes primero (grandes y opacas): si hay que recortar, se quitan las que menos se ven
    imp = esc.prod(1) * alpha
    order = np.argsort(-imp)[:max_manchas]
    xyz, esc, rgb, alpha, q = xyz[order], esc[order], rgb[order], alpha[order], q[order]
    n = len(xyz)
    out = np.zeros(n, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    out["p"], out["s"] = xyz, esc
    out["c"][:, :3] = np.clip(rgb * 255, 0, 255).round()
    out["c"][:, 3] = np.clip(alpha * 255, 0, 255).round()
    out["r"] = np.clip(q * 128 + 128, 0, 255).round()
    return out.tobytes(), n


def malla_para_medir(res_dir, max_tri):
    """Malla exacta (la de SolidWorks) en el mundo del modelo (Y arriba, metros), reducida."""
    m = o3d.io.read_triangle_mesh(os.path.join(res_dir, "malla_color.ply"))
    if len(m.triangles) == 0:
        raise RuntimeError("el escaneo no tiene malla (malla_color.ply)")
    m.transform(np.linalg.inv(Y_UP_TO_Z_UP))
    if len(m.triangles) > max_tri:
        ratio = math.sqrt(len(m.triangles) / max_tri)
        m = m.simplify_vertex_clustering(voxel_size=0.004 * ratio)
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()
    V = np.asarray(m.vertices, dtype=np.float32)
    F = np.asarray(m.triangles, dtype=np.uint32)
    C = (np.clip(np.asarray(m.vertex_colors), 0, 1) * 255).astype(np.uint8) if m.has_vertex_colors() \
        else np.full((len(V), 3), 170, np.uint8)
    blob = (np.array([len(V), len(F)], "<u4").tobytes() + V.astype("<f4").tobytes()
            + F.astype("<u4").tobytes() + C.tobytes())
    return blob, len(F)


def medidas_guardadas(sdir):
    out = []
    for fn in ("medidas.json", "medidas_iphone.json"):
        try:
            with open(os.path.join(sdir, fn), encoding="utf-8") as f:
                for m in json.load(f).get("medidas", []):
                    if "a" in m and "b" in m:
                        out.append({"nombre": m.get("nombre", "Medida"), "a": m["a"], "b": m["b"],
                                    "origen": "iphone" if fn.endswith("iphone.json") else "pc"})
        except (OSError, ValueError):
            pass
    return out


def lib(nombre):
    with open(os.path.join(WEB, nombre), encoding="utf-8") as f:
        src = f.read()
    if "</script" in src:
        src = src.replace("</script", "<\\/script")
    return src


PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>__TITULO__</title>
<style>
  :root { --bg:#0f1012; --panel:rgba(24,25,28,.9); --line:rgba(255,255,255,.1); --fg:#e9e9ea; --muted:#9a9ca3;
    --accent:#f5a821; --mono:ui-monospace,"Cascadia Mono",Consolas,monospace; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg); overflow:hidden;
    font:14px system-ui,"Segoe UI",sans-serif; }
  canvas { display:block; touch-action:none; }
  .panel { position:fixed; background:var(--panel); border:1px solid var(--line); border-radius:14px;
    backdrop-filter:blur(10px); padding:12px 14px; }
  #top { top:14px; left:14px; right:14px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding:8px 10px; }
  #top h1 { font-size:14px; margin:0 auto 0 4px; font-weight:600; min-width:0; }
  .btn { background:rgba(255,255,255,.08); color:var(--fg); border:1px solid var(--line); border-radius:10px;
    padding:7px 12px; font:inherit; cursor:pointer; }
  .btn:hover { background:rgba(255,255,255,.14); }
  .btn.on { background:var(--accent); color:#111; border-color:var(--accent); font-weight:600; }
  #meas { top:76px; right:14px; width:min(330px,calc(100% - 28px)); max-height:calc(100% - 150px); overflow:auto;
    display:flex; flex-direction:column; gap:8px; }
  #meas h2 { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin:0; }
  .m { border-top:1px solid var(--line); padding-top:8px; display:flex; flex-direction:column; gap:4px; }
  .m-top { display:flex; gap:8px; align-items:baseline; }
  .m-top b { font:600 17px var(--mono); margin-left:auto; font-variant-numeric:tabular-nums; }
  .axes { display:flex; gap:6px; flex-wrap:wrap; font:12px var(--mono); font-variant-numeric:tabular-nums; }
  .axes span { padding:1px 7px; border-radius:6px; background:rgba(255,255,255,.06); }
  .x { color:#ff7b72; } .y { color:#7ee787; } .z { color:#79c0ff; }
  .muted { color:var(--muted); font-size:12px; line-height:1.4; }
  .del { margin-left:6px; background:none; border:0; color:var(--muted); cursor:pointer; font-size:12px; }
  #hint { bottom:14px; left:50%; transform:translateX(-50%); color:var(--accent); max-width:calc(100% - 28px);
    text-align:center; padding:8px 14px; }
  #hint[hidden], #msg[hidden] { display:none; }
  #msg { top:50%; left:50%; transform:translate(-50%,-50%); text-align:center; }
  #labels { position:fixed; inset:0; pointer-events:none; }
  .lab { position:absolute; transform:translate(-50%,-130%); background:rgba(10,10,12,.85); color:#fff;
    font:600 12px var(--mono); padding:2px 7px; border-radius:6px; white-space:nowrap; }
</style>
</head>
<body>
<div id="top" class="panel">
  <h1>__TITULO__</h1>
  <button class="btn" id="b-medir" type="button">Medir</button>
  <button class="btn" id="b-ver" type="button">Ver la malla</button>
  <button class="btn" id="b-encuadrar" type="button">Encuadrar</button>
  <button class="btn" id="b-planta" type="button">Planta</button>
</div>
<div id="meas" class="panel">
  <h2>Medidas</h2>
  <div class="muted" id="m-empty">Pulsa <b>Medir</b> y haz clic en dos puntos. Doble clic para acercarte a un punto.
    Ejes como en SolidWorks y Navisworks: Z hacia arriba.</div>
  <div id="m-list" style="display:flex;flex-direction:column;gap:8px"></div>
</div>
<div id="msg" class="panel">Cargando la vista foto-realista…</div>
<div id="hint" class="panel" hidden></div>
<div id="labels"></div>
<script type="text/plain" id="lib-three">__THREE__</script>
<script type="text/plain" id="lib-orbit">__ORBIT__</script>
<script type="text/plain" id="lib-gs">__GS__</script>
<script type="text/plain" id="data-splat">__SPLAT__</script>
<script type="text/plain" id="data-mesh">__MESH__</script>
<script type="application/json" id="data-meas">__MEDIDAS__</script>
<script type="text/plain" id="app">
import * as THREE from '__THREE_URL__';
import { OrbitControls } from '__ORBIT_URL__';
import * as GS3D from '__GS_URL__';
const $ = (id) => document.getElementById(id);
const b64 = (id) => { const s = atob($(id).textContent.trim()); const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; };

const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
// resolución adaptable: más baja mientras se mueve (fluido) y completa al soltar (nítido)
const PR_QUIETO = Math.min(devicePixelRatio || 1, 1.5), PR_MOVIENDO = Math.min(devicePixelRatio || 1, 1) * 0.6;
renderer.setPixelRatio(PR_QUIETO);
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x0f1012);
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 300);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.12;
let restTimer = null;
controls.addEventListener('start', () => { clearTimeout(restTimer); renderer.setPixelRatio(PR_MOVIENDO); });
controls.addEventListener('end', () => { restTimer = setTimeout(() => renderer.setPixelRatio(PR_QUIETO), 250); });
addEventListener('resize', () => { renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); });

// malla exacta (oculta): ahí caen las medidas
const mb = b64('data-mesh'); const hdr = new Uint32Array(mb, 0, 2); const nv = hdr[0], nt = hdr[1];
const pos = new Float32Array(mb, 8, nv * 3);
const idx = new Uint32Array(mb, 8 + nv * 12, nt * 3);
const col = new Uint8Array(mb, 8 + nv * 12 + nt * 12, nv * 3);
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
geo.setAttribute('color', new THREE.BufferAttribute(col, 3, true));
geo.setIndex(new THREE.BufferAttribute(idx, 1));
geo.computeBoundingBox(); geo.computeBoundingSphere();
const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
mesh.visible = false; scene.add(mesh);

// encuadre
const box = geo.boundingBox, center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
const span = Math.max(size.x, size.y, size.z, 0.5);
function frame(top) {
  controls.target.copy(center);
  if (top) camera.position.set(center.x, center.y + span * 1.6, center.z + 0.001);
  else camera.position.set(center.x + span * 0.7, center.y + span * 0.55, center.z + span * 0.9);
  controls.update();
}
frame(false);

// vista foto-realista
let splat = null, photo = true;
const viewer = new GS3D.DropInViewer({ gpuAcceleratedSort: false, sharedMemoryForWorkers: false,
  sphericalHarmonicsDegree: 0, antialiased: false, halfPrecisionCovariancesOnGPU: true, integerBasedSort: true,
  sceneRevealMode: GS3D.SceneRevealMode.Instant });
const splatURL = URL.createObjectURL(new Blob([b64('data-splat')]));
viewer.addSplatScene(splatURL, { format: GS3D.SceneFormat.Splat, showLoadingUI: false, splatAlphaRemovalThreshold: 5 })
  .then(() => { splat = viewer; scene.add(viewer); $('msg').hidden = true; look(); })
  .catch((e) => { console.warn(e); $('msg').textContent = 'No se pudo mostrar la vista foto-realista; se muestra la malla.';
    photo = false; look(); setTimeout(() => $('msg').hidden = true, 4000); });
function look() {
  if (splat) splat.visible = photo;
  mesh.visible = !photo || !splat;
  $('b-ver').textContent = photo && splat ? 'Ver la malla' : 'Ver foto-realista';
}
$('b-ver').onclick = () => { photo = !photo; look(); };
$('b-encuadrar').onclick = () => frame(false);
$('b-planta').onclick = () => frame(true);

// medir
let measures = JSON.parse($('data-meas').textContent || '[]'), measuring = false, pending = null;
const fmt = (m) => m >= 1 ? `${m.toFixed(3)} m` : `${(m * 1000).toFixed(1)} mm`;
const overlay = new THREE.Group(); scene.add(overlay);
const ray = new THREE.Raycaster();
function hitAt(x, y) {
  ray.setFromCamera(new THREE.Vector2((x / innerWidth) * 2 - 1, -(y / innerHeight) * 2 + 1), camera);
  const h = ray.intersectObject(mesh, false)[0];
  return h ? h.point.clone() : null;
}
function hint(t) { $('hint').hidden = !t; $('hint').textContent = t || ''; }
$('b-medir').onclick = () => { measuring = !measuring; pending = null; $('b-medir').classList.toggle('on', measuring);
  hint(measuring ? 'Clic en el primer punto' : ''); draw(); };
let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5 || !measuring) return;
  const p = hitAt(e.clientX, e.clientY);
  if (!p) { hint('Ahí no hay superficie medible: haz clic sobre el modelo'); return; }
  if (!pending) { pending = p; hint('Clic en el segundo punto'); }
  else { measures.push({ nombre: `Medida ${measures.length + 1}`, a: pending.toArray(), b: p.toArray(), origen: 'html' });
    pending = null; hint('Clic en el primer punto'); }
  draw(); list();
});
renderer.domElement.addEventListener('dblclick', (e) => {
  const p = hitAt(e.clientX, e.clientY); if (!p) return;
  const dir = camera.position.clone().sub(p).normalize();
  const d = Math.max(0.15, camera.position.distanceTo(p) * 0.45);
  controls.target.copy(p); camera.position.copy(p).addScaledVector(dir, d); controls.update();
});
const dot = (p, c) => { const s = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8),
  new THREE.MeshBasicMaterial({ color: c, depthTest: false })); s.position.copy(p); s.renderOrder = 10; s.userData.dot = true; return s; };
function draw() {
  overlay.clear(); $('labels').replaceChildren();
  for (const m of measures) {
    const a = new THREE.Vector3(...m.a), b = new THREE.Vector3(...m.b);
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    const ln = new THREE.Line(g, new THREE.LineBasicMaterial({ color: m.origen === 'iphone' ? 0x79c0ff : 0xf5a821, depthTest: false }));
    ln.renderOrder = 9; overlay.add(ln, dot(a, 0xffffff), dot(b, 0xffffff));
    const lab = document.createElement('div'); lab.className = 'lab'; lab.textContent = `${m.nombre} · ${fmt(a.distanceTo(b))}`;
    lab.dataset.x = (a.x + b.x) / 2; lab.dataset.y = (a.y + b.y) / 2; lab.dataset.z = (a.z + b.z) / 2;
    $('labels').append(lab);
  }
  if (pending) overlay.add(dot(pending, 0xf5a821));
}
function list() {
  const L = $('m-list'); L.replaceChildren(); $('m-empty').hidden = measures.length > 0;
  measures.forEach((m, i) => {
    const a = m.a, b = m.b, d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    // ejes como en SolidWorks/Navisworks (Z arriba): X = x, Y = -z, Z = y
    const dx = Math.abs(a[0] - b[0]), dy = Math.abs(a[2] - b[2]), dz = Math.abs(a[1] - b[1]);
    const it = document.createElement('div'); it.className = 'm';
    it.innerHTML = `<div class="m-top"><span></span><b>${fmt(d)}</b></div>
      <div class="axes"><span class="x">X ${fmt(dx)}</span><span class="y">Y ${fmt(dy)}</span><span class="z">Z ${fmt(dz)}</span></div>`;
    it.querySelector('.m-top span').textContent = m.nombre + (m.origen === 'iphone' ? ' (iPhone)' : '');
    const del = document.createElement('button'); del.className = 'del'; del.textContent = 'Borrar';
    del.onclick = () => { measures.splice(i, 1); draw(); list(); };
    it.querySelector('.m-top').append(del); L.append(it);
  });
}
draw(); list();
const v = new THREE.Vector3();
(function loop() {
  requestAnimationFrame(loop); controls.update();
  const s = camera.position.distanceTo(controls.target) * 0.009;
  overlay.children.forEach(o => { if (o.userData.dot) o.scale.setScalar(s); });
  for (const lab of $('labels').children) {
    v.set(+lab.dataset.x, +lab.dataset.y, +lab.dataset.z).project(camera);
    lab.style.display = v.z < 1 ? '' : 'none';
    lab.style.left = `${(v.x + 1) / 2 * innerWidth}px`; lab.style.top = `${(1 - v.y) / 2 * innerHeight}px`;
  }
  renderer.render(scene, camera);
})();
</script>
<script>
// Arranque: las librerías van dentro del archivo; se cargan como módulos desde blobs (funciona con doble clic).
(function () {
  const url = (code) => URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
  const txt = (id) => document.getElementById(id).textContent.replace(/<\\\/script/g, '<' + '/script');
  const three = url(txt('lib-three'));
  const orbit = url(txt('lib-orbit').split("'./three.module.min.js'").join("'" + three + "'"));
  const gs = url(txt('lib-gs').split('"./three.module.min.js"').join('"' + three + '"'));
  const app = url(txt('app').replace('__THREE_URL__', three).replace('__ORBIT_URL__', orbit).replace('__GS_URL__', gs));
  import(app).catch(function (e) {
    document.getElementById('msg').textContent = 'Este navegador no pudo abrir el archivo. Ábrelo con Microsoft Edge o Google Chrome.';
    console.error(e);
  });
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Vista foto-realista -> archivo HTML único con medición")
    ap.add_argument("carpeta", help="carpeta del escaneo (con resultado/vista_foto.ply)")
    ap.add_argument("--salida")
    ap.add_argument("--max-manchas", type=int, default=900_000)
    ap.add_argument("--max-triangulos", type=int, default=500_000)
    args = ap.parse_args()
    res = os.path.join(args.carpeta, "resultado")
    ply = leer_ply_splat(os.path.join(res, "vista_foto.ply"))
    splat, n = a_splat(ply, args.max_manchas)
    del ply
    malla, nt = malla_para_medir(res, args.max_triangulos)
    nombre = os.path.basename(os.path.normpath(args.carpeta)).replace("RGBD_", "")
    html = PLANTILLA
    for k, v in (("__THREE__", lib("three.module.min.js")), ("__ORBIT__", lib("OrbitControls.js")),
                 ("__GS__", lib("gaussian-splats-3d.module.js")),
                 ("__MEDIDAS__", json.dumps(medidas_guardadas(args.carpeta)).replace("</", "<\\/")),
                 ("__TITULO__", f"Levantamiento 3D · {nombre}")):
        html = html.replace(k, v)
    salida = args.salida or os.path.join(res, f"vista_foto_{nombre}.html")
    a, b = html.split("__SPLAT__")
    b1, b2 = b.split("__MESH__")
    with open(salida, "w", encoding="utf-8") as f:
        f.write(a)
        f.write(base64.b64encode(splat).decode("ascii"))
        f.write(b1)
        f.write(base64.b64encode(malla).decode("ascii"))
        f.write(b2)
    print(f"Listo: {salida} ({os.path.getsize(salida) / 1e6:.0f} MB, {n:,} manchas, {nt:,} triángulos para medir)")


if __name__ == "__main__":
    main()
