#!/bin/sh
# ==============================================================================
# MediNet-Node Docker Entrypoint Script
# ==============================================================================
# This script runs before the main application starts and handles:
# - Auto-generation of Django SECRET_KEY (first run only)
# - Persistence of secrets across container restarts
# - Database migrations and static files collection
# ==============================================================================

set -e

# Configuration directory inside the container
CONFIG_DIR="/usr/src/app/config"
SECRET_KEY_FILE="${CONFIG_DIR}/secret_key.txt"

echo "[MEDINET-NODE ENTRYPOINT] Starting initialization..."

# Create config directory if it doesn't exist
mkdir -p "${CONFIG_DIR}"

# ==============================================================================
# Generate Django SECRET_KEY if it doesn't exist
# ==============================================================================
if [ ! -f "${SECRET_KEY_FILE}" ]; then
    echo "[MEDINET-NODE ENTRYPOINT] Generating new Django SECRET_KEY..."

    # Generate a secure random SECRET_KEY (50 characters)
    # Using alphanumeric + special characters for maximum entropy
    python3 -c "
import secrets
import string

# Generate a 50-character random string with alphanumeric + special chars
alphabet = string.ascii_letters + string.digits + '!@#\$%^&*(-_=+)'
secret_key = ''.join(secrets.choice(alphabet) for i in range(50))
print(secret_key)
" > "${SECRET_KEY_FILE}"

    echo "[MEDINET-NODE ENTRYPOINT] SECRET_KEY generated and saved to ${SECRET_KEY_FILE}"
else
    echo "[MEDINET-NODE ENTRYPOINT] Using existing SECRET_KEY from ${SECRET_KEY_FILE}"
fi

# ==============================================================================
# Set file permissions (read-only for security)
# ==============================================================================
chmod 400 "${SECRET_KEY_FILE}"

echo "[MEDINET-NODE ENTRYPOINT] Configuration files secured (read-only)"

# ==============================================================================
# Export SECRET_KEY to environment for Django to use
# ==============================================================================
export SECRET_KEY=$(cat "${SECRET_KEY_FILE}")
echo "[MEDINET-NODE ENTRYPOINT] SECRET_KEY loaded into environment"

# ==============================================================================
# Run database migrations for default database FIRST
# IMPORTANT: Must run default database first because signals depend on users_role table
# ==============================================================================
echo "[MEDINET-NODE ENTRYPOINT] Running database migrations for default database..."
python manage.py migrate

# ==============================================================================
# Run database migrations for datasets_db (second)
# ==============================================================================
echo "[MEDINET-NODE ENTRYPOINT] Running database migrations for datasets_db..."
python manage.py migrate --database=datasets_db

# ==============================================================================
# Create cache table if using database cache in production
# ==============================================================================
if [ "${DEBUG}" = "False" ] || [ "${DEBUG}" = "false" ]; then
    echo "[MEDINET-NODE ENTRYPOINT] Creating cache table for production..."
    python manage.py createcachetable || echo "[MEDINET-NODE ENTRYPOINT] Cache table already exists or not needed"
fi

# ==============================================================================
# Collect static files (only in production)
# ==============================================================================
# TEMPORARILY DISABLED - collectstatic not critical for server startup
# if [ "${DEBUG}" = "False" ] || [ "${DEBUG}" = "false" ]; then
#     echo "[MEDINET-NODE ENTRYPOINT] Collecting static files..."
#     python manage.py collectstatic --noinput
# else
#     echo "[MEDINET-NODE ENTRYPOINT] Skipping collectstatic in development mode"
# fi
echo "[MEDINET-NODE ENTRYPOINT] Skipping collectstatic (not required for development)"

# ==============================================================================
# Start Django server
# ==============================================================================
# Check if we should use development or production server
if [ "${DEBUG}" = "False" ] || [ "${DEBUG}" = "false" ]; then
    echo "[MEDINET-NODE ENTRYPOINT] Starting production server with gunicorn on port 8000..."
    # For production, use gunicorn (make sure it's in requirements.txt)
    exec gunicorn medinet.wsgi:application \
        --bind 0.0.0.0:5001 \
        --workers 4 \
        --timeout 120 \
        --access-logfile - \
        --error-logfile -
else
    echo "[MEDINET-NODE ENTRYPOINT] Starting Django development server on port 5001..."
    exec python manage.py runserver 0.0.0.0:5001
fi
