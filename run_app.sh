#!/bin/bash
# run_app.sh — Launch the NephroTICK web interface
#
# Run this file once to start the app. It will open automatically in your browser.
# Press Ctrl+C in this terminal when you want to stop it.

cd "$(dirname "$0")"

# Install dependencies if not already installed
pip install -r requirements.txt --quiet

# Start the app
streamlit run app.py
