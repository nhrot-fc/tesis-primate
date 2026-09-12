@echo off
rem Agrega modelos al paquete: arrastra sobre este archivo el zip de cada modelo
rem (detector-*-model-*.zip). Descomprime models\<nombre>\ y, si el modelo lo
rem necesita, hf\ en esta misma carpeta. Usa el tar.exe que trae Windows 10 y 11.
setlocal
set "ROOT=%~dp0"
set "DEST=%ROOT:~0,-1%"
if "%~1"=="" (
    echo Arrastra el zip de un modelo sobre Agregar-modelo.bat.
    echo.
    pause
    exit /b 1
)
:next
if "%~1"=="" goto done
echo Agregando %~nx1 ...
tar -xf "%~1" -C "%DEST%"
if errorlevel 1 (echo   No se pudo descomprimir %~nx1) else (echo   Listo.)
shift
goto next
:done
echo.
echo Modelos en models\:
dir /b /ad "%ROOT%models"
echo.
pause
