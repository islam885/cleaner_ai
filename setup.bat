@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Running quick start setup...
python quick_start.py
echo.
echo Setup complete!
pause
