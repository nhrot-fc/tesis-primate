@echo off
rem Visor de espectrogramas. Abre un audio (Ctrl+O), un modelo (Ctrl+M) y detecta (Ctrl+R).
rem Corre sin consola; si algo falla, mira visor.log en esta carpeta.
setlocal
set "ROOT=%~dp0"
call "%ROOT%python\entorno.bat"
start "" "%ROOT%python\pythonw.exe" "%ROOT%src\main.py" %*
