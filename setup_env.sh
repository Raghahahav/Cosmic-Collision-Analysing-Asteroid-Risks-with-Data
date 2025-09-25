#!/bin/bash

echo "Setting up Cosmic Collision Analysis Environment..."
echo

# Activate the virtual environment
echo "Activating virtual environment..."
source ../.venv/Scripts/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install all dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install Jupyter kernel
echo "Setting up Jupyter kernel for this environment..."
python -m ipykernel install --user --name cosmic-collision --display-name "Cosmic Collision Analysis"

echo
echo "Setup complete!"
echo
echo "To use this environment:"
echo "1. Run: source ../.venv/Scripts/activate"
echo "2. Then run: jupyter notebook"
echo "3. Select 'Cosmic Collision Analysis' kernel when running the notebook"
echo
