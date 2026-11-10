import { JSDOM } from 'jsdom';
import fs from 'fs';
const html = fs.readFileSync(process.argv[2], 'utf8');
let js = html.match(/<script type="module">([\s\S]*)<\/script>/)[1];
// sustituir importaciones por objetos simulados
js = js.replace(/^import .*$/gm, '');
const dom = new JSDOM(html.replace(/<script type="module">[\s\S]*<\/script>/, ''), { url: 'http://localhost:8765/', pretendToBeVisual: true, runScripts: 'outside-only' });
const w = dom.window;
const errors = [];
const mk = () => new Proxy(function () {}, {
  get: (t, k) => k === Symbol.toPrimitive ? () => 0 : (k === 'then' ? undefined : mk()),
  apply: () => mk(), construct: () => mk(),
});
const base = mk();
const THREE = new Proxy({}, { get: (t, k) => k === 'WebGLRenderer'
  ? function () { return { domElement: w.document.createElement('canvas'), setPixelRatio() {}, setSize() {}, setClearColor() {}, render() {} }; }
  : base[k] });
let rafCount = 0;
w.requestAnimationFrame = (f) => { if (rafCount++ < 5) setTimeout(() => { try { f(); } catch (e) { errors.push('frame: ' + e.message); } }, 1); };
let sock = null;
w.TextDecoder = TextDecoder; w.TextEncoder = TextEncoder;
w.WebSocket = class { constructor() { sock = this; setTimeout(() => this.onopen && this.onopen(), 1); } send() {} };
const pack = (kind, obj) => {
  const j = new TextEncoder().encode(JSON.stringify(obj));
  const b = new ArrayBuffer(5 + j.length); const dv = new DataView(b);
  dv.setUint8(0, kind); dv.setUint32(1, j.length, true); new Uint8Array(b, 5).set(j); return b;
};
w.fetch = async (u) => ({ ok: true, json: async () => (u.includes('escaneos') ? { carpeta: 'C:/x', escaneos: [
  { nombre: 'a', carpeta: 'RGBD_a', fecha: '', cuadros: 5, procesado: true, splat: true, descartado: false, en_curso: false, con_modelo: true,
    archivos: [{ nombre: 'modelo_texturizado.glb', mb: 1 }, { nombre: 'modelo_solidworks.3mf', mb: 2 }, { nombre: 'obj/modelo_navisworks.fbx', mb: 3 }] },
  { nombre: 'b', carpeta: 'RGBD_b', fecha: '', cuadros: 5, procesado: true, descartado: false, en_curso: false, con_modelo: false, error: 'sin memoria', archivos: [] },
  { nombre: 'c', carpeta: 'RGBD_c', fecha: '', cuadros: 5, procesado: false, descartado: false, en_curso: false, procesando: true, archivos: [] },
  { nombre: 'd', carpeta: 'RGBD_d', fecha: '', cuadros: 5, procesado: false, descartado: false, en_curso: true, iphone_conectado: false, archivos: [] }] } : { ips: ['1.2.3.4'], puerto: 8765, medidas: [], pc: 4 }) });
w.console.error = (...a) => errors.push('console.error: ' + a.join(' '));
w.THREE = THREE; w.OrbitControls = mk(); w.GLTFLoader = mk(); w.__errs = errors;
try { w.eval('(function(){' + js + '\n})()'); } catch (e) { errors.push('carga: ' + e.message); }
await new Promise(r => setTimeout(r, 300));
// avance de "crear modelo" llegando desde la PC
try {
  sock.onmessage({ data: pack(11, { cuadros: 10, clave: 2, mensaje: 'x', procesando: true,
    progreso: { carpeta: 'RGBD_c', fraccion: 0.42, etapa: 'Uniendo cuadros', restante_s: 200, transcurrido_s: 90 } }) });
} catch (e) { errors.push('avance: ' + e.message); }
// interacciones: pestañas y medir
try {
  for (const t of ['escaneos', 'actualizaciones', 'visor']) w.document.querySelector(`[data-tab="${t}"]`).click();
  w.document.getElementById('measure-btn').click();
} catch (e) { errors.push('clic: ' + e.message); }
await new Promise(r => setTimeout(r, 300));
const pct = w.document.querySelector('#progress .prog-pct');
if (!pct || pct.textContent !== '42 %') errors.push('barra de avance: ' + (pct ? pct.textContent : 'no aparece'));
const slot = w.document.querySelector('.prog-slot[data-carpeta="RGBD_c"] .prog-pct');
if (!slot) errors.push('barra en la tarjeta del escaneo: no aparece');
console.log('avance:', pct && pct.textContent, '|', w.document.querySelector('#progress .prog-time')?.textContent);
console.log('cuadros dibujados:', rafCount, '| visor oculto:', w.document.getElementById('viewing').hidden, '| errores:', errors.length ? errors : 'ninguno');
process.exit(errors.length || rafCount < 3 ? 1 : 0);   // salir aunque haya temporizadores
