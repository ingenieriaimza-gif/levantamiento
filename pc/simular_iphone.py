#!/usr/bin/env python3
"""Simula la app del iPhone: envía un paquete RGB-D al programa de la PC por el protocolo de red."""
import argparse, asyncio, json, os, struct, sys, time
import aiohttp

def pack(kind, header, payload=b""):
    h = json.dumps(header).encode()
    return struct.pack("<BI", kind, len(h)) + h + payload

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paquete"); ap.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    ap.add_argument("--fps", type=float, default=8.0)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.paquete, "meta.json")))
    frames = [json.loads(l) for l in open(os.path.join(a.paquete, "frames.jsonl")) if l.strip()]
    meshes, last = 0, None
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(a.url, max_msg_size=64 * 1024 * 1024) as ws:
            async def reader():
                nonlocal meshes, last
                async for m in ws:
                    k, hl = struct.unpack_from("<BI", m.data, 0)
                    h = json.loads(m.data[5:5 + hl])
                    if k == 10:
                        meshes += 1; print(f"  malla en vivo #{meshes}: {h['nv']:,} vértices, {h['nt']:,} triángulos", flush=True)
                    elif k == 11 and h.get("mensaje") != last:
                        last = h.get("mensaje"); print("  estado:", last, flush=True)
            rt = asyncio.create_task(reader())
            await ws.send_bytes(pack(2, {"nombre": "prueba_simulada", "dispositivo": "simulado"}))
            t0 = time.time()
            for n, fr in enumerate(frames):
                base = os.path.join(a.paquete, "frames", f"{int(fr['i']):06d}")
                jpg = open(base + ".jpg", "rb").read(); dep = open(base + ".depth", "rb").read(); con = open(base + ".conf", "rb").read()
                h = {**fr, "dw": meta["depthWidth"], "dh": meta["depthHeight"], "kf": n % 2 == 0,
                     "nj": len(jpg), "nd": len(dep), "nc": len(con)}
                await ws.send_bytes(pack(1, h, jpg + dep + con))
                await asyncio.sleep(max(0, (n + 1) / a.fps - (time.time() - t0)))
            await asyncio.sleep(3)
            await ws.send_bytes(pack(3, {}))
            print(f"Enviados {len(frames)} cuadros en {time.time() - t0:.1f} s", flush=True)
            await asyncio.sleep(1.5)
            await ws.close()
            rt.cancel()
    print("mallas recibidas:", meshes)

asyncio.run(main())
