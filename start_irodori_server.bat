@echo off
setlocal
cd /d "%~dp0"
set "UV_PYTHON_INSTALL_DIR=%~dp0.python"
set "UV_CACHE_DIR=%~dp0.uv-cache"
set "UV_PROJECT_ENVIRONMENT=%~dp0.venv"
set "HF_HOME=%~dp0.huggingface"
set "TORCH_HOME=%~dp0.torch-cache"
set "UV_EXE=%~dp0.tools\uv.exe"
if not exist "%UV_EXE%" (
    echo [setup] Downloading uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%~dp0.tools'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
)
if not exist "%UV_EXE%" (
    echo [setup] Failed to download uv. Install uv manually and retry.
    pause
    exit /b 1
)
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [setup] Building Python environment: uv sync --extra cu128
    "%UV_EXE%" sync --extra cu128
    if errorlevel 1 (
        echo [setup] uv sync failed.
        pause
        exit /b 1
    )
)
"%UV_EXE%" run --no-sync python -m irodori_openai_tts --host 127.0.0.1 --port 8088
if errorlevel 1 pause
endlocal
