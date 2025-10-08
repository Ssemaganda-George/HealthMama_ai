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