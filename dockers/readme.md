## General:
    1. We use 2 dockers:
        'whisper_docker' - Responsible for voice transcription
        'app_docker'     - Management software (including the LLM)

    2. Make sure you have CUDA installed ('nvidia-smi' from terminal)


## First time build:
    1. Install huggingface-cli library
        1.1 Make sure you use python 3.10.12 (or higher version)
        1.2 Create venv ('python -m venv myVenv')
        1.3 Activate venv (myVenv/bin/activate) 
        1.4 pip install -U huggingface_hub

    2. Building whisper_docker:
        2.1 You need to download whisper model, so run the following command from terminal:
            2.1.1 huggingface-cli download mobiuslabsgmbh/faster-whisper-large-v3-turbo --cache-dir /home/amitli/repo/VoiceOperation/models/faster-whisper-large-v3-turbo
            2.2.2 you can change '/home/amitli/repo/VoiceOperation/models/faster-whisper-large-v3-turbo' to other path)
        2.2 update docker-compose.yml (under whisper_docker folder) with the model folder (under 'volumes' section)
        2.3 Go to whisper_docker folder:
             docker compose up --build

    3. Building app_docker:
        3.1 You need to download the LLM model, so run the following command from terminal:
            huggingface-cli download Qwen/Qwen3-0.6B --local-dir /home/amitli/repo/VoiceOperation/models/llm/Qwen/Qwen3-0.6B --local-dir-use-symlinks=False
        3.2 Go to app_docker folder:
             docker compose up --build

## Running:
    1. Go to whisper_docker folder and run 
        docker compose up
    2. Go to app_docker folder and run 
        docker compose up