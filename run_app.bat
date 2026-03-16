@echo off
REM Simple launcher for Windows

echo --- SAXS Averager Launcher ---

REM 1. Check if .venv exists, if not create it
if not exist ".venv" (
    echo Creating virtual environment (.venv)...
    python -m venv .venv
)

REM 2. Activate the virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate

REM 3. Install/Update dependencies
echo Checking dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 4. Run the application
echo Launching SAXS Averager...
streamlit run SAXS_averager.py

pause
