@echo off
rem Pasa un modelo por archivos o carpetas de audio y deja una tabla de Raven
rem (<audio>.detections.txt) junto a cada grabacion.
rem
rem   - Arrastra una carpeta, o varios audios, sobre este archivo.
rem   - O desde una consola:  Detectar.bat --model models\frcnn D:\grabaciones
rem
rem Con mas de un modelo en models\ pregunta cual usar; --score cambia el umbral.
setlocal
set "ROOT=%~dp0"
call "%ROOT%python\entorno.bat"
if "%~1"=="" (
    echo Arrastra una carpeta o archivos de audio sobre Detectar.bat, o pasalos como argumento.
    echo.
    pause
    exit /b 1
)
"%ROOT%python\python.exe" "%ROOT%src\detect.py" %*
echo.
pause
