@echo off
REM Run the full benchmark suite
REM Usage: scripts\run_benchmarks.bat [latency|throughput|concurrency|quantization|all]

set SUITE=%1
if "%SUITE%"=="" set SUITE=all

set PYTHON_EXEC=python
if not "%CONDA_DEFAULT_ENV%"=="llm-engine" (
    if exist "%USERPROFILE%\miniconda3\envs\llm-engine\python.exe" (
        set PYTHON_EXEC="%USERPROFILE%\miniconda3\envs\llm-engine\python.exe"
    )
)

echo ========================================================
echo Running LLM Inference Benchmarks: %SUITE%
echo ========================================================

if "%SUITE%"=="latency" goto run_lat
if "%SUITE%"=="throughput" goto run_tp
if "%SUITE%"=="concurrency" goto run_conc
if "%SUITE%"=="quantization" goto run_quant
if "%SUITE%"=="all" goto run_all

:run_lat
%PYTHON_EXEC% benchmarks\benchmark_latency.py
goto end

:run_tp
%PYTHON_EXEC% benchmarks\benchmark_throughput.py
goto end

:run_conc
%PYTHON_EXEC% benchmarks\benchmark_concurrency.py
goto end

:run_quant
%PYTHON_EXEC% benchmarks\benchmark_quantization.py
goto end

:run_all
echo [1/4] Running Latency Benchmark...
%PYTHON_EXEC% benchmarks\benchmark_latency.py
echo [2/4] Running Throughput Benchmark...
%PYTHON_EXEC% benchmarks\benchmark_throughput.py
echo [3/4] Running Concurrency Benchmark...
%PYTHON_EXEC% benchmarks\benchmark_concurrency.py
echo [4/4] Running Quantization Benchmark...
%PYTHON_EXEC% benchmarks\benchmark_quantization.py
goto end

:end
echo Benchmark suite complete. Results saved in benchmarks\results\
