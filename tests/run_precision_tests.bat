@echo off
REM Script to run comprehensive precision testing suite on Windows

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

REM Create symlink to root requirements if it doesn't exist
if not exist "tests\requirements.txt" (
    if exist "requirements.txt" (
        mklink "tests\requirements.txt" "..\requirements.txt" 2>nul
        if %ERRORLEVEL% NEQ 0 (
            copy /Y "requirements.txt" "tests\requirements.txt" >nul
        )
    )
)

REM Set environment variables
set "PYTHONPATH=%CD%;%PYTHONPATH%"
set CUDA_VISIBLE_DEVICES=0
set TORCH_CUDA_ARCH_LIST=7.0+PTX
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo Installing test requirements...
"%PYTHON_CMD%" -m pip install -r tests\requirements.txt
"%PYTHON_CMD%" -m pip install -e .

REM Create test reports directory
if not exist test_reports mkdir test_reports

echo Starting precision test suite...
echo ================================================

REM Check CUDA availability
"%PYTHON_CMD%" -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

REM Create results directory for benchmarks
if not exist .benchmarks mkdir .benchmarks

REM Run tests for each precision mode
set FAILED=0

REM Function to run tests (implemented as a label)
:run_precision_tests
echo Running tests for %1 precision...
"%PYTHON_CMD%" -m pytest tests/test_precision.py tests/test_tree_planning.py tests/test_information_retrieval.py ^
    -v ^
    --precision-mode=%1 ^
    -m "precision" ^
    --benchmark-only ^
    --benchmark-autosave ^
    --html=test_reports/%1_report.html
if %ERRORLEVEL% NEQ 0 set FAILED=1
echo ------------------------------------------------
goto :eof

REM Run tests for each precision mode
call :run_precision_tests fp16
call :run_precision_tests fp32
call :run_precision_tests fp64
call :run_precision_tests bf16

REM Run mixed precision tests
echo Running mixed precision tests...
"%PYTHON_CMD%" -m pytest tests/test_precision.py::test_mixed_precision_training ^
    tests/test_tree_planning.py::test_tree_planning_numerical_stability ^
    tests/test_information_retrieval.py::test_retrieval_with_mixed_precision ^
    -v ^
    --html=test_reports/mixed_precision_report.html
if %ERRORLEVEL% NEQ 0 set FAILED=1

REM Run memory benchmarks
echo Running memory benchmarks...
"%PYTHON_CMD%" -m pytest tests/test_precision.py::test_precision_memory_usage ^
    tests/test_tree_planning.py::test_tree_planning_memory_usage ^
    tests/test_information_retrieval.py::test_retrieval_memory_efficiency ^
    -v ^
    --benchmark-only ^
    --html=test_reports/memory_benchmarks.html

REM Generate comparison report
"%PYTHON_CMD%" -m pytest-benchmark compare

echo ================================================

REM Show test report location
echo Test reports generated in test_reports/ directory
echo Benchmark results available in .benchmarks/ directory

REM Exit with appropriate status
if %FAILED%==0 (
    echo All precision tests completed successfully
    exit /b 0
) else (
    echo Some precision tests failed
    exit /b 1
)
