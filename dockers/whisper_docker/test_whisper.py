import requests

if __name__ == "__main__":
    URL = "http://localhost:8013/transcribe/"

    payload = {
        "audio_input": r"/home/amitli/repo/VoiceOperation/dockers/whisper_docker/common_voice_en_2925.wav"
    }

    response = requests.post(URL, json=payload)
    print(response.json())
