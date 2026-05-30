@echo off
echo Resetting Cleaner AI...
echo.

echo Removing old data...
rmdir /s /q data 2>nul
rmdir /s /q models 2>nul
rmdir /s /q logs 2>nul
rmdir /s /q __pycache__ 2>nul

echo.
echo Creating fresh directories...
mkdir data\inp
mkdir data\out
mkdir data\otv
mkdir models
mkdir logs

echo.
echo ✓ Reset complete!
echo.
echo Next: python run.py
echo.
pause
