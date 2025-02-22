@echo off
REM Setup script for running the quickstart example

REM Add parent directory to Python path
set "SCRIPT_DIR=%~dp0"
set "PYTHONPATH=%SCRIPT_DIR%..;%PYTHONPATH%"

REM Create symbolic link to requirements if needed
if not exist "requirements.txt" (
    if exist "..\requirements.txt" (
        mklink "requirements.txt" "..\requirements.txt" 2>nul
        if errorlevel 1 (
            copy /Y "..\requirements.txt" "requirements.txt" >nul
        )
    )
)

REM Find Python executable
where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python3
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set PYTHON_CMD=python
    ) else (
        echo Error: Python not found. Please ensure Python is installed and in PATH
        exit /b 1
    )
)

echo Installing requirements...
"%PYTHON_CMD%" -m pip install -r requirements.txt
"%PYTHON_CMD%" -m pip install -e ..

echo Running quickstart example...
"%PYTHON_CMD%" model_quickstart.py

REM Keep window open if running by double-click
echo.
echo Press any key to exit...
pause >nul
