"""
Configuration settings for HealthMama AI Application
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = 'gpt-3.5-turbo'
    OPENAI_MAX_TOKENS = 500
    OPENAI_TEMPERATURE = 0.7
    
    # Application Configuration
    MAX_CONVERSATION_HISTORY = 10
    CONVERSATION_TRIM_SIZE = 8
    
    # Data Configuration
    DIABETES_DATA_PATH = 'data_diabetes/cleaned_diabetes.txt'
    PREECLAMPSIA_DATA_PATH = 'data_preelampsia/cleaned_preecampsia.txt'
    
    # Server Configuration
    HOST = '0.0.0.0'
    PORT = int(os.getenv('PORT', 5007))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = os.getenv('RATELIMIT_STORAGE_URL', 'memory://')
    
    @staticmethod
    def validate_config():
        """Validate critical configuration"""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Clean API key
        Config.OPENAI_API_KEY = Config.OPENAI_API_KEY.strip()
        
        return True


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    SECRET_KEY = 'test-secret-key'


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'default')
    return config.get(env, config['default'])