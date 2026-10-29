@echo off
REM Run the test suite
REM Usage: scripts\run_tests.bat

set PYTHON_EXEC=python
if not "%CONDA_DEFAULT_ENV%"=="llm-engine" (
    if exist "%USERPROFILE%\miniconda3\envs\llm-engine\python.exe" (
        set PYTHON_EXEC="%USERPROFILE%\miniconda3\envs\llm-engine\python.exe"
    )
)

echo Running LLM Inference Engine tests...
%PYTHON_EXEC% -m pytest tests/ -v --tb=short %*
