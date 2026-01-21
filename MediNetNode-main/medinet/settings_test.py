"""
Test-specific Django settings for MediNet.

This settings file is used exclusively for running tests.
It provides hardcoded values to avoid dependency on .env files.
"""

# Set environment variables BEFORE importing settings
import os
os.environ.setdefault('SECRET_KEY', 'django-insecure-test-key-for-running-tests-only-do-not-use-in-production-12345')
os.environ.setdefault('DEBUG', 'True')
os.environ.setdefault('ALLOWED_HOSTS', 'localhost,127.0.0.1')

from .settings import *

# Override SECRET_KEY with a hardcoded test value
SECRET_KEY = 'django-insecure-test-key-for-running-tests-only-do-not-use-in-production-12345'

# Force DEBUG to True for tests
DEBUG = True

# Session timeout for tests (30 minutes = 1800 seconds)
SESSION_IDLE_TIMEOUT = 1800

# Use in-memory database for faster tests
# Each test gets a fresh database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
        'TEST': {
            'NAME': ':memory:',
        }
    },
    'datasets_db': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
        'TEST': {
            'NAME': ':memory:',
        }
    },
}

# Use in-memory cache for tests
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'test-cache',
    }
}

# Disable password validators for faster tests
AUTH_PASSWORD_VALIDATORS = []

# Speed up password hashing in tests
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.MD5PasswordHasher',
]

# Disable logging during tests to reduce noise
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'handlers': {
        'null': {
            'class': 'logging.NullHandler',
        },
    },
    'root': {
        'handlers': ['null'],
        'level': 'CRITICAL',
    },
}
