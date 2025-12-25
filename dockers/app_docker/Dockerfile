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
RUN pip install --no-cache-dir outlines==0.2.3
RUN pip install --no-cache-dir silero-vad
RUN pip install --no-cache-dir scipy
RUN pip install --no-cache-dir openwakeword
RUN pip install --no-cache-dir PyAudio==0.2.14
RUN pip install --no-cache-dir accelerate==1.12.0

#COPY models/faster-whisper-large-v3-turbo /.cache/huggingface/hub
COPY models/llm/Qwen/Qwen3-0.6B/ \
     /.cache/huggingface/hub/models--Qwen--Qwen3-0.6

WORKDIR /app

COPY app_config ./app_config
COPY nlp ./nlp
COPY main.py ./main.py

CMD ["python3", "main.py"]