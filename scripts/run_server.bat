@echo off
REM Run the LLM Inference Engine server
REM Usage: scripts\run_server.bat [config_path]

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=configs\default.yaml

set PYTHON_EXEC=python
if not "%CONDA_DEFAULT_ENV%"=="llm-engine" (
    if exist "%USERPROFILE%\miniconda3\envs\llm-engine\python.exe" (
        set PYTHON_EXEC="%USERPROFILE%\miniconda3\envs\llm-engine\python.exe"
    )
)

set LLM_ENGINE_CONFIG=%CONFIG%
echo Starting LLM Inference Engine with config: %CONFIG%
%PYTHON_EXEC% -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
