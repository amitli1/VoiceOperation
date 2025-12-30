
#FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime7
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04





RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    portaudio19-dev \
    && apt-get clean


#RUN apt-get update && apt-get install -y libcudnn9-cuda-12

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    faster-whisper

WORKDIR /app
COPY app_config ./app_config
COPY audio ./audio

RUN pip freeze

CMD ["python3", "audio/services/whisper_service.py"]
