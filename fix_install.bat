@echo off
echo ============================================================
echo Cleaner AI - Fix Installation
echo ============================================================
echo.

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing core dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing image processing...
pip install opencv-python Pillow

echo.
echo Installing GUI...
pip install PyQt5 PyQt5-sip

echo.
echo Installing utilities...
pip install numpy scipy scikit-image tensorboard tqdm

echo.
echo Testing imports...
python test_imports.py

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Next: Run "python run.py" to launch GUI
echo.
pause
