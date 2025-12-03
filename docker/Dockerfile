# Use Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy your code
COPY VoiceOperation/ ./VoiceOperation/

# Install dependencies
# Adjust this if you have a requirements.txt
COPY VoiceOperation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set entrypoint
CMD ["python", "VoiceOperation/main.py"]
