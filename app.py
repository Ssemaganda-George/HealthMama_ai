import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create the Flask application using the factory
from app.factory import create_app

app = create_app()
# also provide WSGI-standard name
application = app

if __name__ == "__main__":
    # Local debug run
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
