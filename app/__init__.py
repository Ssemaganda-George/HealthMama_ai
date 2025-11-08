"""
HealthMama AI - Main Application Entry Point
Professional Flask application for health information chatbot
"""
import os
import sys

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .factory import create_app
import logging

logger = logging.getLogger(__name__)

# Create Flask app instance for WSGI import compatibility.
# This allows gunicorn to import "app:app" (package 'app' exposing attribute 'app')
try:
    app = create_app()
    application = app  # also export WSGI-standard name
except Exception as exc:
    # Avoid crashing at import time if environment not yet prepared.
    # Log the error so it's visible in startup logs.
    logger.warning("App creation at import-time failed: %s", exc)
    app = None
    application = None

def main():
    """Main application entry point"""
    try:
        # Run application
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG
        )
        
    except Exception as e:
        print(f"Failed to start HealthMama AI: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()