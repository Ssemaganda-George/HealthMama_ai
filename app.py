import os
import sys

# ...ensure project root is on path...
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create the Flask application using your factory
from app.factory import create_app

app = create_app()
# also export the WSGI-standard name used by some platforms
application = app

if __name__ == "__main__":
    # local debug fallback
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
