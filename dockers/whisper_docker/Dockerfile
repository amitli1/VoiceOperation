FROM python:3.11-slim
#FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN pip install --no-cache-dir fastapi
RUN pip install --no-cache-dir uvicorn
RUN pip install --no-cache-dir python-multipart


ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
RUN mkdir -p /root/.cache/huggingface/transformers

WORKDIR /app

COPY AgenTeam/ ./AgenTeam/

WORKDIR /app/AgenTeam

CMD ["python3", "audio/services/whisper_service.py"]