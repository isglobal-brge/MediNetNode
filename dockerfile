FROM python:3.11-slim

# Establece variables de entorno para la configuración de la aplicación
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=medinet.settings
ENV PYTHONDONTWRITEBYTECODE=1

# Create application directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY MediNetNode-main/ .

# Create necessary directories
RUN mkdir -p /usr/src/app/config \
    /usr/src/app/logs \
    /usr/src/app/media \
    /usr/src/app/static \
    /usr/src/app/staticfiles \
    /usr/src/app/db

# Copy and set permissions for entrypoint script
COPY MediNetNode-main/entrypoint.sh /usr/src/app/entrypoint.sh
# Convert Windows line endings (CRLF) to Unix line endings (LF)
RUN sed -i 's/\r$//' /usr/src/app/entrypoint.sh && \
    chmod +x /usr/src/app/entrypoint.sh
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*
# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/').read()" || exit 1

# Use entrypoint script
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]