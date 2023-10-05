#!/bin/bash

# Name of the virtual environment
VENV_NAME="venv_optosim"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if the virtual environment already exists
if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment $VENV_NAME already exists. Skipping creation."
else
    # Create a virtual environment
    python3 -m venv $VENV_NAME
    echo "Virtual environment $VENV_NAME created."
fi

# Activate the virtual environment
source $VENV_NAME/bin/activate

# Install necessary packages with optosim
pip install -e ./

echo "Packages are installed in $VENV_NAME."
