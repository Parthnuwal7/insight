#!/bin/bash
echo "Starting Sentiment Analysis Dashboard..."
echo
echo "Installing dependencies (if needed)..."
pip install -r requirements.txt
echo
echo "Starting Streamlit application..."
streamlit run app.py