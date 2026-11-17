#!/usr/bin/env python3
"""
Corre un comando y deja en su registro cómo terminó (código de salida y qué significa).

Lo usa el conector de Claude para las pruebas: sin esto, si el proceso se caía (falla de memoria de Open3D,
la PC sin RAM, etc.) el registro solo se cortaba y no había forma de saber por qué.

Uso: python ejecutar_registrado.py <registro.txt> -- <comando...>
"""
import os
import subprocess
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)


def explicar(code):
    try:
        from diagnostico import explicar_codigo
        return explicar_codigo(code)
    except Exception:       # noqa: BLE001
        return f"código {code}"


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        print(__doc__)
        return 2
    reg, cmd = sys.argv[1], sys.argv[3:]
    t0 = time.time()
    flags = 0x08000000 if os.name == "nt" else 0
    with open(reg, "a", encoding="utf-8") as log:
        env = dict(os.environ, PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                creationflags=flags)
        code = proc.wait()
    texto = "bien" if code == 0 else explicar(code)
    with open(reg, "a", encoding="utf-8") as log:
        log.write(f"\n@@FIN codigo={code} minutos={(time.time() - t0) / 60:.1f} | {texto}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
