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
        1.5 Note: I tested it with huggingface_hub==1.5.0

    2. Building whisper_docker:
        2.1 You need to download whisper model, so run the following command from terminal:
            2.1.1 Download:
               2.1.1.1 Linux: huggingface-cli download mobiuslabsgmbh/faster-whisper-large-v3-turbo --cache-dir /home/amitli/repo/VoiceOperation/models/faster-whisper-large-v3-turbo
               2.1.1.2 Windows: hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo --cache-dir /home/amitli/repo/VoiceOperation/models/faster-whisper-large-v3-turbo
            2.2.2 you can change '/home/amitli/repo/VoiceOperation/models/faster-whisper-large-v3-turbo' to other path)
        2.2 update docker-compose.yml (under whisper_docker folder) with the model folder (under 'volumes' section)
        2.3 Go to whisper_docker folder:
             docker compose up --build

    3. Building app_docker:
        3.1 You need to download the LLM model, so run the following command from terminal:
            huggingface-cli download Qwen/Qwen3-0.6B --local-dir /home/amitli/repo/VoiceOperation/models/llm/Qwen/Qwen3-0.6B --local-dir-use-symlinks=False
        3.2 Go to app_docker folder:
             docker compose up --build

    4. Building tts_docker:
        4.1 You need to download tts (Kokoro-82M) model, so run the following command from terminal:
            2.1.1 Download:
               2.1.1.1 Linux: huggingface-cli download hexgrad/Kokoro-82M --cache-dir /home/amitli/repo/VoiceOperation/models/kokoro_model
               2.1.1.2 Windows: hf download hexgrad/Kokoro-82M --cache-dir /home/amitli/repo/VoiceOperation/models/kokoro_model
            2.2.2 you can change '/home/amitli/repo/VoiceOperation/models/kokoro_model' to other path)
        2.2 update docker-compose.yml (under tts_docker folder) with the model folder (under 'volumes' section)
        2.3 Go to tts_docker folder:
             docker compose up --build
        2.4 Note: 
            2.4.1 If working without internet - it takes about 3 minutes for the model to load.
            2.4.2 Same situation when getting results for the first time

## Running:
    1. Go to whisper_docker folder and run 
        docker compose up
    2. Go to app_docker folder and run 
        docker compose up


# Test dockers with CURL commands:
   1. TTS:
      ```bash
         curl -X POST "http://127.0.0.1:8002/synthesize/"      -H "Content-Type: application/json"      -d '{"text": "Hello, how are you?", "voice": "en_us"}'
      ```
   2. Whisper:
      ```bash
        curl -F "file=@/home/amitli/repo/VoiceOperation/dockers/whisper_docker/common_voice_en_2925.wav" http://localhost:8013/transcribe/
      ```


## Known errors (&Fix):
1. Whisper:
   1.1 Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9
   1.2 Check torch.cuda.is_available()
   1.3 pip install --upgrade faster-whisper ctranslate2