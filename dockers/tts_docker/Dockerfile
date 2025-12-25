#FROM python:3.11-slim
#FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime
#FROM pytorch/pytorch:2.5.1-cuda12.2-cudnn9-runtime
#FROM nvidia/cuda:12.2.1-cudnn8-runtime-ubuntu22.04
#FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

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
RUN pip install --no-cache-dir pydantic
RUN pip install --no-cache-dir kokoro
RUN pip install --no-cache-dir sounddevice
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

WORKDIR /app

COPY app_config ./app_config
COPY audio ./audio

CMD ["python3", "audio/services/tts_service.py"]