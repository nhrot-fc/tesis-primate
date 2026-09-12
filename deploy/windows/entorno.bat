@echo off
rem Variables que mantienen el paquete autocontenido: nada sale a internet ni se escribe
rem fuera de esta carpeta. Lo llaman Visor.bat y Detectar.bat con ROOT ya definido.
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "HF_HOME=%ROOT%cache\hf"
set "YOLO_OFFLINE=1"
set "YOLO_CONFIG_DIR=%ROOT%cache\ultralytics"
set "MPLCONFIGDIR=%ROOT%cache\matplotlib"
set "PYTHONUTF8=1"
