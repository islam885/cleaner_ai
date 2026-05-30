@echo off
echo ============================================================
echo Cleaner AI - Installation Script
echo ============================================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Running setup check...
python check_setup.py

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Run GUI: python run.py
echo 2. Run quick start: python quick_start.py
echo 3. Run demo: python demo.py
echo.
pause
