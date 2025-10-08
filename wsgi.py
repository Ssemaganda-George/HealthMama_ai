"""
WSGI Entry Point for HealthMama AI
Used by production servers like Gunicorn
"""
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.factory import create_app

# Create application instance
application = create_app()

# For Railway and other platforms that expect 'app'
app = application

if __name__ == "__main__":
    application.run()