import numpy             as np
import sounddevice       as sd
import soundfile         as sf
import requests
import logging
import os

def in_docker():
 return os.path.exists("/.dockerenv") or os.path.exists("/run/.dockerenv")

def get_running_ip():
    if in_docker():
        return "host.docker.internal"
    else:
        return "127.0.0.1"

def play_text(text_to_user):

    try:
        response     = requests.post(f"http://{get_running_ip()}:8002/synthesize/", json={"text": text_to_user})
        data         = response.json()
        sample_rate  = data["sample_rate"]
        audios       = [np.array(audio, dtype=np.float32) for audio in data["audio"]]
        full_audio   = np.concatenate(audios)

        sd.play(full_audio, samplerate=sample_rate, blocking=True)
    except Exception as e:
        logging.error('Cant connect to TTS service')

def play_wav_file(wav_file_name, output_device):
    logging.info(f'Play: {wav_file_name}')
    data, fs       = sf.read(wav_file_name, dtype='float32')
    data           = np.expand_dims(data, axis=1)
    sd.play(data, fs, device=output_device)
    sd.wait()