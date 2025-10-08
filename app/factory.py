"""
Flask Application Factory Pattern
"""
from flask import Flask
from config.settings import get_config
import logging
import os


def create_app(config_name=None):
    """Application factory function"""
    
    # Get configuration
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    config_class = get_config()
    
    # Create Flask app
    app = Flask(__name__, 
                template_folder='../Templates',
                static_folder='../static')
    
    # Load configuration
    app.config.from_object(config_class)
    
    # Validate configuration
    try:
        config_class.validate_config()
    except ValueError as e:
        app.logger.error(f"Configuration error: {e}")
        raise
    
    # Setup logging
    setup_logging(app)
    
    # Initialize services
    from app.services.data_service import DataService
    data_service = DataService()
    data_service.initialize()
    app.data_service = data_service
    
    # Register blueprints
    register_blueprints(app)
    
    # Setup error handlers
    setup_error_handlers(app)
    
    app.logger.info("HealthMama AI application created successfully")
    
    return app


def setup_logging(app):
    """Setup application logging"""
    if not app.debug and not app.testing:
        # Production logging setup
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, app.config['LOG_LEVEL']))
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        )
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)
        app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL']))


def register_blueprints(app):
    """Register application blueprints"""
    from app.routes.main import main_bp
    from app.routes.api import api_bp
    from app.routes.health import health_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)  # Remove the /api prefix
    app.register_blueprint(health_bp)


def setup_error_handlers(app):
    """Setup error handlers"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'Server Error: {error}')
        return {'error': 'Internal server error'}, 500
    
    @app.errorhandler(400)
    def bad_request(error):
        return {'error': 'Bad request'}, 400