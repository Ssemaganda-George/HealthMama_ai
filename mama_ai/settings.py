# mama_ai/settings.py

import os
from pathlib import Path


# Basic settings
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'CdcMGPd25ayqWNGXG04lLnPzTi51Wt9zzxILKVYxhLV_LzQVKZ_V_prvVsCeWO-5sCc'

DEBUG = True
ALLOWED_HOSTS = []

# Installed apps (add your apps here)
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'users',  # Users app for authentication
    'ai',  # AI app for chatbot functionality
]

# Middleware (default provided by Django)
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


# Templates settings
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            BASE_DIR / 'templates',  # Check that this points to the correct templates folder
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI and ASGI settings (default settings provided)
WSGI_APPLICATION = 'mama_ai.wsgi.application'
ASGI_APPLICATION = 'mama_ai.asgi.application'

# Database settings
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password settings
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Static and media files settings
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / "static"]
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
ROOT_URLCONF = 'mama_ai.urls'  # Replace 'mama_ai' with the name of your project folder


# Other necessary settings
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# settings.py
# Enable CSRF protection
CSRF_COOKIE_SECURE = False
SESSION_COOKIE_SECURE = False
SECURE_SSL_REDIRECT = False  # Redirect to HTTPS
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.Argon2PasswordHasher',
]

# Configure authentication settings for secure login
# LOGIN_REDIRECT_URL = 'users/health_data/'
# LOGOUT_REDIRECT_URL = 'users/login/'

LOGIN_URL = 'login'  # Redirect to this view if unauthenticated
LOGIN_REDIRECT_URL = 'health_data'  # Redirect here after login
LOGOUT_REDIRECT_URL = 'login'  # Redirect here after logout
LOGIN_REDIRECT_URL = '/visualization/'
