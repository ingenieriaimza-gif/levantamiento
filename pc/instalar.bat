@echo off
chcp 65001 >nul
title Levantamiento 3D - instalacion
cd /d "%~dp0"
echo.
echo  Levantamiento 3D - instalacion del programa de la PC
echo  ====================================================
echo.
where py >nul 2>nul
if errorlevel 1 (
  echo  No encontre Python. Instala Python 3.12 desde https://www.python.org/downloads/
  echo  y marca la casilla "Add python.exe to PATH". Luego vuelve a abrir este archivo.
  pause
  exit /b 1
)
py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo  Se necesita Python 3.12. Descargalo de https://www.python.org/downloads/release/python-3128/
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo  Creando entorno de Python...
  py -3.12 -m venv .venv || goto :error
)
echo  Instalando librerias (tarda unos minutos la primera vez)...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
echo.
echo  Listo. Abre "iniciar.bat" para recibir escaneos.
echo  La primera vez Windows preguntara si permites el acceso a la red:
echo  marca "Redes privadas" Y "Redes publicas" (los hotspots cuentan como publicas).
echo.
pause
exit /b 0
:error
echo.
echo  Algo fallo durante la instalacion. Copia el texto de arriba y mandaselo a Claude.
pause
exit /b 1
