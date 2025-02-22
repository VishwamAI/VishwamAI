@echo off
REM Setup script for test environment on Windows

REM Enable command output
echo on

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Warning: Not running with administrator privileges.
    echo Some operations may fail.
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

REM Create symbolic link to requirements.txt if not exists
if not exist "requirements.txt" (
    if exist "..\requirements.txt" (
        mklink "requirements.txt" "..\requirements.txt" 2>nul
        if %ERRORLEVEL% NEQ 0 (
            copy /Y "..\requirements.txt" "requirements.txt" >nul
            if %ERRORLEVEL% NEQ 0 (
                echo Error: Failed to copy requirements.txt
                exit /b 1
            )
        )
    ) else (
        echo Error: requirements.txt not found in parent directory
        exit /b 1
    )
)

REM Create necessary directories
if not exist "test_reports" (
    mkdir "test_reports" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to create test_reports directory
        exit /b 1
    )
)

if not exist ".benchmarks" (
    mkdir ".benchmarks" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to create .benchmarks directory
        exit /b 1
    )
)

REM Install test dependencies
"%PYTHON_CMD%" -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install requirements
    exit /b 1
)

REM Install package in development mode
"%PYTHON_CMD%" -m pip install -e ..
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install package in development mode
    exit /b 1
)

REM Verify installation
"%PYTHON_CMD%" -c "import vishwamai; print(f'Vishwamai version: {vishwamai.__version__}')"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to verify installation
    exit /b 1
)

REM Make test scripts executable (Windows equivalent - not really needed but for consistency)
attrib -R run_precision_tests.bat >nul 2>&1
attrib -R run_precision_tests.sh >nul 2>&1

echo.
echo ✓ Test environment setup completed successfully
exit /b 0
