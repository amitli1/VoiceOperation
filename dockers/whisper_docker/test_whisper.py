import requests
from scipy.io import wavfile

if __name__ == "__main__":
    URL = "http://localhost:8013/transcribe/"

    sr, samples = wavfile.read("common_voice_en_2925.wav")

    response = requests.post(URL, json={"audio_input": samples.astype(int).tolist()})
    print(response.json())
