@echo off
chcp 65001 >nul
title Levantamiento 3D - receptor
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo  Primero ejecuta instalar.bat
  pause
  exit /b 1
)
start "" http://localhost:8765
".venv\Scripts\python.exe" -I servidor_levanta.py %*
pause
