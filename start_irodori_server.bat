@echo off
setlocal
cd /d "%~dp0"
set "UV_PYTHON_INSTALL_DIR=%~dp0.python"
set "UV_CACHE_DIR=%~dp0.uv-cache"
set "UV_PROJECT_ENVIRONMENT=%~dp0.venv"
set "HF_HOME=%~dp0.huggingface"
set "TORCH_HOME=%~dp0.torch-cache"
set "UV_EXE=%~dp0.tools\uv.exe"
set "LOCK_STAMP=%~dp0.venv\.alslime-lock-sha256"
if not exist "%UV_EXE%" (
    echo [setup] Downloading uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%~dp0.tools'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
)
if not exist "%UV_EXE%" (
    echo [setup] Failed to download uv. Install uv manually and retry.
    pause
    exit /b 1
)
rem uv.lock のハッシュを .venv 内の控えと比べ、環境が無いか依存が変わったときだけ sync する。
rem （本体を上書き更新したときに依存を追従させるため）
set "LOCK_HASH="
for /f "skip=1 tokens=* delims=" %%h in ('certutil -hashfile "%~dp0uv.lock" SHA256 ^| findstr /v /i "certutil"') do if not defined LOCK_HASH set "LOCK_HASH=%%h"
set "LOCK_HASH=%LOCK_HASH: =%"
set "STAMP_HASH="
if exist "%LOCK_STAMP%" set /p STAMP_HASH=<"%LOCK_STAMP%"
if not exist "%~dp0.venv\Scripts\python.exe" set "STAMP_HASH="
if /i not "%STAMP_HASH%"=="%LOCK_HASH%" (
    echo [setup] Building Python environment: uv sync --extra cu128
    "%UV_EXE%" sync --extra cu128
    if errorlevel 1 (
        echo [setup] uv sync failed.
        pause
        exit /b 1
    )
    >"%LOCK_STAMP%" echo %LOCK_HASH%
)
rem AlSlime が渡した接続先を使う。単独起動時の公開先はこの PC のみにする。
if not defined IRODORI_HOST set "IRODORI_HOST=127.0.0.1"
"%UV_EXE%" run --no-sync python -m irodori_openai_tts
if errorlevel 1 pause
endlocal
