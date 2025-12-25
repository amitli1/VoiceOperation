FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# Install Python3 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    portaudio19-dev \
    && apt-get clean

# Upgrade pip first
RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir fastapi
RUN pip install --no-cache-dir uvicorn
RUN pip install --no-cache-dir python-multipart
RUN pip install --no-cache-dir faster-whisper


ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
RUN mkdir -p /root/.cache/huggingface/transformers

WORKDIR /app

COPY app_config ./app_config
COPY audio ./audio

CMD ["python3", "audio/services/whisper_service.py"]