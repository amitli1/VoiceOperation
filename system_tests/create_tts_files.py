from pydub import AudioSegment
import numpy as np
import os
import requests

def create_wav(text, file_name):


    response            = requests.post(f"http://127.0.0.1:8002/synthesize/", json={"text": text})
    data                = response.json()
    samples = data['audio']
    samples = np.int16(samples / np.max(np.abs(samples)) * 32767)
    audio   = AudioSegment(samples.tobytes(),frame_rate=24000, channels=1, sample_width=2)
    audio   = audio.set_frame_rate(16000)
    audio.export(f"{os.getcwd()}/wav_commands/{file_name}", format="wav")

if __name__ == "__main__":

    create_wav("circle building number 1", "circle_building_1.wav")