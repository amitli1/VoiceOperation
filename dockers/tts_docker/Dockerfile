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
RUN pip install -i https://pypi.fury.io/balacoon/ balacoon-tts

WORKDIR /app

COPY app_config ./app_config
COPY audio ./audio
#COPY dockers/tts_docker/en_us_cmuartic_jets_cpu.addon .
COPY dockers/tts_docker/en_us_cmartic_jets_gpu.addon .


# 🔥 Pre-download Kokoro model at build time
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HUGGINGFACE_HUB_CACHE=/models/hf


RUN python3 - <<'EOF'
from kokoro import KPipeline

print("Downloading Kokoro model...")
KPipeline(lang_code="en-us", device="cpu")
print("Model downloaded")
EOF

CMD ["python3", "audio/services/tts_service.py"]
#CMD ["python3", "audio/services/tts_service_balacoon.py"]
