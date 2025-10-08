#!/bin/bash
# Download model data if needed
python -c "
import os
import requests
from sentence_transformers import SentenceTransformer

# Ensure model is downloaded
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('Model loaded successfully')
except Exception as e:
    print(f'Error loading model: {e}')
"

# Start the application
gunicorn --bind 0.0.0.0:$PORT app:app