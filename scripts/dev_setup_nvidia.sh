#!/bin/bash

echo "Creating a virtual environment"
python3 -m venv .venv

echo "Activating the virtual environment"
source .venv/bin/activate

echo "Installing dependencies"
pip install -r requirements/requirements_nvidia.txt

echo "Completed Installation!"

