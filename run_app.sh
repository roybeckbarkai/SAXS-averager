#!/bin/bash

# Simple launcher for macOS and Linux

# 1. Determine the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "--- SAXS Averager Launcher ---"

# 2. Check if .venv exists, if not create it
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# 3. Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 4. Install/Update dependencies
echo "Checking dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the application
echo "Launching SAXS Averager..."
streamlit run SAXS_averager.py
