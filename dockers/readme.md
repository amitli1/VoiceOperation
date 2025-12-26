## build:
     docker compose up --build 

## run:
    docker comppose up

## test:
    http://0.0.0.0:8002/docs#/

## test curl:
    curl -X POST http://localhost:8002/synthesize/   -H "Content-Type: application/json"   -d '{"text":"Hello, this is a Kokoro TTS test"}'

## Notes:
    sudo apt-get install -y nvidia-docker2