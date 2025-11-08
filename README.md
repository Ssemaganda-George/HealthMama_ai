# HealthMama AI 🏥

A professional Flask-based AI-powered health chatbot specialized in diabetes and preeclampsia information, with multilingual support including English and Luganda.

## 🌟 Features

- **AI-Powered Health Assistance**: Specialized chatbots for diabetes and preeclampsia
- **Multilingual Support**: English and Luganda language support with cultural context
- **Mobile-Responsive Design**: Professional UI optimized for all devices
- **Dark/Light Mode**: Toggle between themes for better user experience
- **Privacy-Focused**: Local chat history storage with user control
- **Professional Architecture**: Modular Flask application with best practices
- **Security Features**: Input validation and XSS protection
- **Production-Ready**: Configured for deployment on Railway, Heroku, and other platforms

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Ssemaganda-George/HealthMama_ai.git
cd HealthMama_ai
```

2. **Create and activate virtual environment**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env  # Create from example
```

Edit `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
FLASK_ENV=development
```

5. **Run the application**
```bash
python run.py
```

Visit `http://localhost:5000` in your browser.

## 📁 Project Structure

```
HealthMama_ai/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── factory.py                # Application factory
│   ├── routes/                   # Route blueprints
│   │   ├── __init__.py
│   │   ├── main.py              # Main routes (index, about)
│   │   ├── api.py               # API endpoints (chat, ask)
│   │   └── health.py            # Health checks and monitoring
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   ├── data_service.py      # Data loading and search
│   │   └── openai_service.py    # OpenAI API integration
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── validators.py        # Input validation
│       └── helpers.py           # Helper functions
├── config/                       # Configuration management
│   ├── __init__.py
│   └── settings.py              # Application settings
├── data_diabetes/               # Diabetes knowledge base
├── data_preelampsia/           # Preeclampsia knowledge base
├── static/                      # Static assets (CSS, JS, images)
├── templates/                   # Jinja2 templates
├── tests/                       # Unit tests
├── run.py                       # Development server entry point
├── wsgi.py                      # Production WSGI entry point
├── requirements.txt             # Python dependencies
├── Procfile                     # Heroku deployment
├── railway.toml                 # Railway deployment
└── README.md                    # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for chat functionality | Yes | - |
| `FLASK_SECRET_KEY` | Flask session secret key | Yes | - |
| `FLASK_ENV` | Environment (development/production) | No | development |
| `PORT` | Server port | No | 5000 |
| `LOG_LEVEL` | Logging level | No | INFO |

### Development vs Production

The application automatically detects the environment and configures itself accordingly:

- **Development**: Debug mode enabled, detailed logging
- **Production**: Debug mode disabled, optimized for performance

## 🔌 API Endpoints

### Health & Monitoring

- `GET /health` - Health check endpoint
- `GET /status` - Detailed application status
- `GET /test` - Debug and testing endpoint

### Chat API

- `POST /api/chat` - Main chat endpoint
- `POST /api/ask` - Legacy endpoint for backward compatibility
- `POST /api/clear_history` - Clear conversation history
- `GET /api/models` - Get available AI models

### Example API Usage

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is diabetes?", "model": "diabetes"}'
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=app
```

## 🚀 Deployment

### Railway

1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push to main branch

### Heroku

```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key_here
heroku config:set FLASK_SECRET_KEY=your_secret_here

# Deploy
git push heroku main
```

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key_here
export FLASK_SECRET_KEY=your_secret_here
export FLASK_ENV=production

# Run with Gunicorn
gunicorn --bind 0.0.0.0:8000 wsgi:application
```

## Deploying to Render (concrete steps)

1. Prepare repo (already done)
   - Ensure `requirements.txt` exists and lists dependencies.
   - Ensure `Procfile` is present: `web: gunicorn --bind 0.0.0.0:$PORT wsgi:application`
   - Ensure `wsgi.py` exposes `application` (done).

2. Create Web Service on Render (Dashboard)
   - Visit https://dashboard.render.com → New → Web Service.
   - Connect your Git provider and select this repository & branch (e.g., main).
   - Environment: Python.
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT wsgi:application`
   - Instance type / Plan: Starter (or your preferred plan).
   - Health Check Path: `/health` (optional but recommended).

3. Set Environment Variables (Render Dashboard → Environment)
   - Required (mark as secret):
     - OPENAI_API_KEY = <your_openai_api_key>
     - FLASK_SECRET_KEY = <random-secret>
   - Recommended:
     - FLASK_ENV = production
   - Note: Do NOT commit these values to your repository.

4. Deploy & Monitor
   - Click "Create Web Service" / "Manual Deploy".
   - Open the deploy logs in Render to watch build and startup.
   - If the process fails, check for missing environment variables or dependency errors.

5. Verify endpoints
   - GET https://<your-service>.onrender.com/health
   - GET https://<your-service>.onrender.com/status

6. Troubleshooting checklist
   - "OPENAI_API_KEY environment variable is required" on startup → add the secret in Render and redeploy.
   - Import errors or missing packages → confirm `requirements.txt` includes the package and correct versions.
   - Template/static path issues → factory.py sets template_folder and static_folder; verify relative paths if serving fails.

7. Optional: Render CLI / manifest
   - If you prefer infra-as-code, generate a manifest from Render's dashboard or use the Render CLI to export a valid blueprint. Do not store secrets in the manifest.

## Interactive Render Dashboard Walkthrough (step-by-step)

1. Log in
   - Open https://dashboard.render.com and sign in with your GitHub account.

2. Create a new Web Service
   - Click "New" → "Web Service".
   - Click "Connect a repository" → choose GitHub and authorize if needed.
   - Select repository: choose `HealthMama_ai` (or the repo that contains this project).
   - Branch: select `main` (or the branch you want to deploy).
   - Click "Next".

3. Configure service basics
   - Name: keep default (e.g., `healthmama-ai`) or enter a name.
   - Environment: choose "Python".
   - Region: choose nearest region.
   - Instance Type / Plan: select "Free Starter" or your preferred plan.
   - Click "Next".

4. Build & Start commands
   - Build Command: paste `pip install -r requirements.txt`
   - Start Command: paste `gunicorn --bind 0.0.0.0:$PORT wsgi:application`
   - Health Check Path (optional): `/health`
   - Click "Create Web Service" (or "Next" then "Create").

5. Add environment variables (required)
   - After service creation, go to the service → "Environment" tab.
   - Add the following (mark secrets where indicated):
     - Key: `OPENAI_API_KEY` — Value: your OpenAI API key (Secret)
     - Key: `FLASK_SECRET_KEY` — Value: a random secret string (Secret)
     - Key: `FLASK_ENV` — Value: `production`
   - Save changes.

6. Deploy
   - If deployment did not start automatically, click "Manual Deploy" → "Deploy latest commit".
   - Watch the "Deploys" logs in real time.

7. What to watch for in logs
   - Successful build: lines showing "Installing collected packages" and "Successfully installed".
   - Gunicorn start: logs showing "Booting worker" or "Listening at: http://0.0.0.0:<port>".
   - Common failure: `ValueError: OPENAI_API_KEY environment variable is required` → add the `OPENAI_API_KEY` secret and redeploy.

8. Verify service
   - Open the service URL shown by Render.
   - Check endpoints:
     - `GET https://<your-service>.onrender.com/health`
     - `GET https://<your-service>.onrender.com/status`
   - If endpoints return JSON with "status": "healthy" or "running", deployment succeeded.

9. Redeploy & rollbacks
   - Any git push to the connected branch triggers a new deploy.
   - Use "Manual Deploy" to deploy a specific branch/commit.
   - Use the "History" or "Deploys" tab to rollback or re-deploy a previous commit.

10. Troubleshooting quick tips
   - Missing dependency import errors: ensure package is declared in `requirements.txt`.
   - Template/static path errors: confirm `app.factory` uses the correct relative template/static paths.
   - If audio/image upload endpoints fail: check request size limits and logs for stack traces.
   - Paste failing log lines here and I will provide exact fixes.

## 🔒 Security Features

- **Input Validation**: Comprehensive validation for all user inputs
- **XSS Protection**: Automatic sanitization of user content
- **File Upload Security**: Restricted file types and size limits
- **Content Security Policy**: Browser-level security headers
- **Rate Limiting**: Protection against abuse (configurable)

## 🌍 Internationalization

The application supports:

- **English**: Full medical terminology and explanations
- **Luganda**: Culturally appropriate responses with local context
- **Auto-detection**: Automatically detects user language
- **Local Context**: References to Ugandan foods and healthcare practices

## 📊 Data Sources

- **Diabetes Knowledge Base**: Comprehensive diabetes management information
- **Preeclampsia Database**: Maternal health and preeclampsia guidance
- **Cultural Context**: Uganda-specific health practices and food recommendations

## 🛠️ Development

### Code Style

```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black app/ tests/

# Check style
flake8 app/ tests/

# Type checking
mypy app/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Add tests in `tests/`
3. Implement feature in appropriate modules
4. Update documentation
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Contact: [Your Contact Information]

## 🙏 Acknowledgments

- OpenAI for GPT API
- Flask community for excellent framework
- Contributors and testers

---

**HealthMama AI** - Empowering health decisions with AI 🏥✨


Project hosted here as a railway app:
https://railway.com/project/f15ffa48-defe-4766-a5be-9b538d248fce/service/d6a2ec4b-3ec1-4171-aa5c-80ea3ce17168/variables?environmentId=9abeadc0-ef3c-4dbf-812a-63b83ff6dcff