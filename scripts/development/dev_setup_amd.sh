#!/bin/bash

echo "Creating a virtual environment"
python3 -m venv .venv

echo "Activating the virtual environment"
source .venv/bin/activate

echo "Installing dependencies"
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
pip install -r requirements/requirements_amd.txt

echo "Completed Installation!"

