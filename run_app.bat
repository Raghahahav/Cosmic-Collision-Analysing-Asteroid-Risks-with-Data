@echo off
echo Starting Cosmic Collision Analyzer Web App...
echo.

REM Activate virtual environment if it exists
if exist "..\.venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "..\.venv\Scripts\activate.bat"
) else (
    echo No virtual environment found. Make sure dependencies are installed.
)

REM Install streamlit dependencies if not already installed
echo Checking dependencies...
pip install streamlit plotly --quiet

echo.
echo 🌌 Launching Cosmic Collision Analyzer...
echo 🚀 Opening in your default browser...
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run streamlit_app.py

pause
