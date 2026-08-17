@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .\.venv\Scripts\activate.bat
) else if exist ".\venv\Scripts\activate.bat" (
    call .\venv\Scripts\activate.bat
) else (
    echo ERROR: No virtual environment was found.
    echo Create one with: py -m venv venv
    echo Or use the existing project env: .\venv\Scripts\activate.bat
    exit /b 1
)

python app.py
