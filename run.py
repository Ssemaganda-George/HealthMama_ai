"""
HealthMama AI - Main Application Entry Point
Professional Flask application for health information chatbot
"""
import os
import sys

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.factory import create_app
from config.settings import get_config

def main():
    """Main application entry point"""
    try:
        # Create application
        app = create_app()
        
        # Get configuration
        config = get_config()
        
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