#!/bin/bash

echo "============================================================"
echo "Cleaner AI - Installation Script"
echo "============================================================"
echo ""

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo ""
echo "Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Running setup check..."
python3 check_setup.py

echo ""
echo "============================================================"
echo "Installation complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Run GUI: python3 run.py"
echo "2. Run quick start: python3 quick_start.py"
echo "3. Run demo: python3 demo.py"
echo ""
