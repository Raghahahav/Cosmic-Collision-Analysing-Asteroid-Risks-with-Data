#!/bin/bash

echo "Starting Cosmic Collision Analyzer Web App..."
echo

# Activate virtual environment if it exists
if [ -f "../.venv/Scripts/activate" ]; then
    echo "Activating virtual environment..."
    source "../.venv/Scripts/activate"
elif [ -f "../.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "../.venv/bin/activate"
else
    echo "No virtual environment found. Make sure dependencies are installed."
fi

# Install streamlit dependencies if not already installed
echo "Checking dependencies..."
pip install streamlit plotly --quiet

echo
echo "🌌 Launching Cosmic Collision Analyzer..."
echo "🚀 Opening in your default browser..."
echo
echo "Press Ctrl+C to stop the server"
echo

streamlit run streamlit_app.py
