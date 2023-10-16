@echo off
set "python_exec=..\..\..\python_embeded\python.exe"

echo Installing ComfyUI MotionDiff..

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
    %python_exec% -s install.py
) else (
    echo Installing with system Python
    python install.py
)

pause