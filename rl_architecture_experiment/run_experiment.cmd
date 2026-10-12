@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%;%PYTHONPATH%"
if defined PLATOON_EXPERIMENT_PYTHON (
    set "PYTHON_EXE=%PLATOON_EXPERIMENT_PYTHON%"
) else if exist "D:\sofat\anaconda3\envs\pytorch_gpu_env\python.exe" (
    set "PYTHON_EXE=D:\sofat\anaconda3\envs\pytorch_gpu_env\python.exe"
) else (
    set "PYTHON_EXE=python"
)
"%PYTHON_EXE%" -m platoon_control %*
exit /b %ERRORLEVEL%
