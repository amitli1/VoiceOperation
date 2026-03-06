import numpy             as np
import sounddevice       as sd
import soundfile         as sf
import requests
import logging
import os
import pyaudio

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


def get_support_sample_rate():
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:  # is input device
            logging.info(f"Device {i}: {dev['name']}")
            # Try common sample rates
            for rate in [8000, 16000, 22050, 44100, 48000, 96000]:
                try:
                    if p.is_format_supported(rate,
                                             input_device=dev['index'],
                                             input_channels=int(dev['maxInputChannels']),
                                             input_format=pyaudio.paInt16):
                        logging.info(f"  Supported rate: {rate} Hz")
                except ValueError:
                    pass

    p.terminate()

def get_input_device():
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:  # is input device
            logging.info(f"Device {i}: {dev['name']}")
            # Try common sample rates
            try:
                if p.is_format_supported(16000,
                                         input_device=dev['index'],
                                         input_channels=int(dev['maxInputChannels']),
                                         input_format=pyaudio.paInt16):
                    p.terminate()
                    return i
            except ValueError:
                pass

    p.terminate()



def get_output_device():
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxOutputChannels'] > 0:  # is input device
            logging.info(f"Device {i}: {dev['name']}")
            # Try common sample rates
            try:
                if p.is_format_supported(48000,
                                         input_device=dev['index'],
                                         input_channels=int(dev['maxInputChannels']),
                                         input_format=pyaudio.paInt16):
                    p.terminate()
                    return i
            except ValueError:
                pass

    p.terminate()