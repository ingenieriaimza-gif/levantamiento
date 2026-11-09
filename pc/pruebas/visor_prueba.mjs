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
w.WebSocket = class { constructor() { setTimeout(() => this.onopen && this.onopen(), 1); } send() {} };
w.fetch = async (u) => ({ ok: true, json: async () => (u.includes('escaneos') ? { carpeta: 'C:/x', escaneos: [{ nombre: 'a', carpeta: 'RGBD_a', fecha: '', cuadros: 5, procesado: true, splat: true, descartado: false, en_curso: false, archivos: [{ nombre: 'modelo_texturizado.glb', mb: 1 }] }] } : { ips: ['1.2.3.4'], puerto: 8765, medidas: [], pc: 4 }) });
w.console.error = (...a) => errors.push('console.error: ' + a.join(' '));
w.THREE = THREE; w.OrbitControls = mk(); w.GLTFLoader = mk(); w.__errs = errors;
try { w.eval('(function(){' + js + '\n})()'); } catch (e) { errors.push('carga: ' + e.message); }
await new Promise(r => setTimeout(r, 300));
// interacciones: pestañas y medir
try {
  for (const t of ['escaneos', 'actualizaciones', 'visor']) w.document.querySelector(`[data-tab="${t}"]`).click();
  w.document.getElementById('measure-btn').click();
} catch (e) { errors.push('clic: ' + e.message); }
await new Promise(r => setTimeout(r, 300));
console.log('cuadros dibujados:', rafCount, '| visor oculto:', w.document.getElementById('viewing').hidden, '| errores:', errors.length ? errors : 'ninguno');
if (errors.length || rafCount < 3) process.exit(1);
