"""
Zona Wi‑Fi propia de la PC para que el iPhone se conecte siempre directo, sin depender del Wi‑Fi del lugar.

Usa la "Zona con cobertura inalámbrica móvil" de Windows (NetworkOperatorTetheringManager) por medio de
PowerShell, sin instalar nada. Nombre y contraseña fijos, así el iPhone la recuerda y se une solo la próxima vez.
En la red de la zona la PC siempre es 192.168.137.1 (la app del iPhone la busca ahí).

Windows apaga la zona cuando nadie la usa un rato; el programa la vuelve a encender cada minuto.
"""
import json
import subprocess
import sys

NOMBRE = "Levantamiento3D"
CLAVE = "levanta3d"
IP_PC = "192.168.137.1"

_PS = r"""
$ErrorActionPreference = 'Stop'
try {
  Add-Type -AssemblyName System.Runtime.WindowsRuntime
  $m = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
         $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
         $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
  function Esperar($op, $tipo) { $t = $m.MakeGenericMethod($tipo).Invoke($null, @($op)); $t.Wait(-1) | Out-Null; $t.Result }
  function EsperarAccion($op) { $t = [System.WindowsRuntimeSystemExtensions]::AsTask($op); $t.Wait(-1) | Out-Null }
  $null = [Windows.Networking.Connectivity.NetworkInformation, Windows.Networking.Connectivity, ContentType = WindowsRuntime]
  $null = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager, Windows.Networking.NetworkOperators, ContentType = WindowsRuntime]
  $perfil = [Windows.Networking.Connectivity.NetworkInformation]::GetInternetConnectionProfile()
  if ($perfil -eq $null) {
    $perfil = [Windows.Networking.Connectivity.NetworkInformation]::GetConnectionProfiles() |
              Where-Object { $_.IsWlanConnectionProfile -or $_.IsWwanConnectionProfile -or $_.NetworkAdapter -ne $null } |
              Select-Object -First 1
  }
  if ($perfil -eq $null) { @{ ok = $false; activa = $false; motivo = 'sin_red' } | ConvertTo-Json -Compress; exit }
  $tm = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager]::CreateFromConnectionProfile($perfil)
  $cfg = $tm.GetCurrentAccessPointConfiguration()
  if ($cfg.Ssid -ne '__NOMBRE__' -or $cfg.Passphrase -ne '__CLAVE__') {
    if ($tm.TetheringOperationalState -eq 'On') {
      $null = Esperar ($tm.StopTetheringAsync()) ([Windows.Networking.NetworkOperators.NetworkOperatorTetheringOperationResult])
    }
    $cfg.Ssid = '__NOMBRE__'; $cfg.Passphrase = '__CLAVE__'
    EsperarAccion ($tm.ConfigureAccessPointAsync($cfg))
  }
  if ($tm.TetheringOperationalState -ne 'On') {
    $r = Esperar ($tm.StartTetheringAsync()) ([Windows.Networking.NetworkOperators.NetworkOperatorTetheringOperationResult])
    @{ ok = ($r.Status -eq 'Success'); activa = ($r.Status -eq 'Success'); motivo = "$($r.Status) $($r.AdditionalErrorMessage)";
       clientes = $tm.ClientCount } | ConvertTo-Json -Compress
  } else {
    @{ ok = $true; activa = $true; motivo = 'ya_activa'; clientes = $tm.ClientCount } | ConvertTo-Json -Compress
  }
} catch {
  @{ ok = $false; activa = $false; motivo = $_.Exception.Message } | ConvertTo-Json -Compress
}
""".replace("__NOMBRE__", NOMBRE).replace("__CLAVE__", CLAVE)


def encender():
    """Enciende (o deja encendida) la zona Wi‑Fi de la PC. Devuelve un dict para mostrar en el programa."""
    base = {"nombre": NOMBRE, "clave": CLAVE, "ip": IP_PC}
    if not sys.platform.startswith("win"):
        return {**base, "activa": False, "mensaje": "La zona Wi‑Fi propia solo existe en Windows."}
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", _PS],
                           capture_output=True, text=True, timeout=60, creationflags=0x08000000)
        out = (r.stdout or "").strip().splitlines()
        res = json.loads(out[-1]) if out else {"activa": False, "motivo": (r.stderr or "").strip()[:200]}
    except Exception as e:      # noqa: BLE001
        res = {"activa": False, "motivo": str(e)}
    if res.get("activa"):
        n = res.get("clientes")
        msg = (f"Zona Wi‑Fi de la PC encendida: «{NOMBRE}», contraseña «{CLAVE}». "
               + ("El iPhone está conectado a ella." if n else "Conecta el iPhone a esa red."))
    elif res.get("motivo") == "sin_red":
        msg = ("Windows solo enciende la zona Wi‑Fi si la PC tiene alguna conexión (Wi‑Fi o cable). "
               "Conéctala a cualquier red y la zona se enciende sola.")
    else:
        msg = f"No se pudo encender la zona Wi‑Fi de la PC ({res.get('motivo', '')}). " \
              "Revisa en Configuración › Red › Zona con cobertura inalámbrica móvil."
    return {**base, "activa": bool(res.get("activa")), "clientes": res.get("clientes"), "mensaje": msg}
