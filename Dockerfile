FROM python:3.10-slim

# Install system dependencies for OpenCV and other ML tasks
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DJANGO_DEBUG=False
ENV PYTHONPATH="/app:/app/django_app"

# Move to the django app directory
WORKDIR /app/django_app

# Run migrations and collect static files
# Note: we use sqlite by default, so we migrate on build or start
RUN python manage.py collectstatic --noinput

# Hugging Face Spaces typically use port 7860
EXPOSE 7860

# CMD to run the server using gunicorn
CMD ["sh", "-c", "python manage.py migrate && gunicorn --bind 0.0.0.0:7860 --timeout 600 interpreter_project.wsgi:application"]
