@echo off
rem Ejecutar con clic derecho > "Ejecutar como administrador" si la app no logra conectarse.
netsh advfirewall firewall delete rule name="Levantamiento 3D" >nul 2>nul
netsh advfirewall firewall add rule name="Levantamiento 3D" dir=in action=allow protocol=TCP localport=8765 profile=any
netsh advfirewall firewall add rule name="Levantamiento 3D descubrimiento" dir=in action=allow protocol=UDP localport=5353 profile=any
echo Listo: puerto 8765 (TCP) y descubrimiento (UDP 5353) permitidos.
pause
