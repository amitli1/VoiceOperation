FROM python:3.11-slim
# FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    && apt-get clean

RUN pip install --no-cache-dir fastapi
RUN pip install --no-cache-dir uvicorn
RUN pip install --no-cache-dir pydantic
RUN pip install --no-cache-dir kokoro
RUN pip install --no-cache-dir sounddevice
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

WORKDIR /app

COPY app_config ./app_config
COPY audio ./audio
COPY nlp ./nlp
COPY main.py ./main.py

CMD ["python3", "audio/services/tts_service.py"]