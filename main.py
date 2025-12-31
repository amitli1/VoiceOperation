from nlp.LLM_Handler     import LLM_Handler
from collections         import deque
from silero_vad          import load_silero_vad, get_speech_timestamps
from fastapi             import FastAPI, Request
from scipy.io.wavfile    import write
from datetime            import datetime
import numpy             as np
import sounddevice       as sd
import logging
import torch
import os
import openwakeword
import pyaudio
import time
import requests
import json
import uvicorn
import threading

def get_timestamp_string():
    return datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

CURRENT_DATE = get_timestamp_string()

def create_output_folder():
    folder_path = os.path.join("output", CURRENT_DATE)
    os.makedirs(folder_path, exist_ok=True)

def init_logger():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(   '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')

    # # File handler
    os.makedirs("logs", exist_ok=True)
    log_name     = f"logs/log_{CURRENT_DATE}.txt"
    file_handler = logging.FileHandler(log_name)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

init_logger()

def write_samples(fname, audio, samplerate=16000):
    start_time    = time.time()
    output_folder = f'output/{CURRENT_DATE}/{fname}.wav'
    write(output_folder, samplerate, audio)
    end_time = time.time()
    logging.info(f"\t[{(end_time-start_time):.2f} ms] Write audio (after wakeword) to: {output_folder}")


def in_docker():
 return os.path.exists("/.dockerenv") or os.path.exists("/run/.dockerenv")

def get_running_ip():
    if in_docker():
        #return "host.docker.internal"
        #return "172.17.0.1"
        return "whisper"
    else:
        return "127.0.0.1"

def capture_audio_after_wakeword(vad_model, last_audios, silence_threshold   = 1.0):

    recorded_audio      = []

    logging.info("Capturing speech...")
    start_time = time.time()


    while True:
        try:
            mic_audio         = np.frombuffer(mic_stream.read(CHUNK,
                                                              exception_on_overflow=False),
                                                              dtype=np.int16)

            recorded_audio.append(mic_audio)
            samples           = np.concatenate(recorded_audio, axis=0)
            if len(samples) < (silence_threshold * 16000):
                continue
            samples           = samples.astype(np.float32) / 32768.0
            tail_audio        = samples[-int(silence_threshold * 16000):]
            speech_timestamps = get_speech_timestamps(tail_audio, vad_model, sampling_rate=16000)
            is_silence        = len(speech_timestamps) == 0
            if is_silence:
                break
        except Exception as e:
            logging.error(f"\tError reading from audio stream. (\n{e}\n)")
            break

    elapsed_time   = time.time() - start_time
    recorded_audio = list(last_audios) + recorded_audio
    full_audio     = np.concatenate(recorded_audio).astype(np.float32) / 32768.0  # Normalize for Whisper
    audio_len      = len(full_audio) / 16000
    logging.info(f"[Timing] Audio capturing took {elapsed_time:.2f} seconds. [Audio len: {audio_len:.2F} sec]")
    return full_audio

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

def send_command(user_command):
    command  = f'http://localhost:8080/{user_command}'
    #response = requests.post(command, json={})
    #logging.info(f"Command status code: {response.status_code}")
    #logging.info(f"Response body      : {response.text}")

app = FastAPI()

@app.post("/message")
async def message_endpoint(data: dict):
    logging.info(f"Received: {data}")
    return {"status": "ok"}

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8053)


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

if __name__ == "__main__":

    logging.info('Start')
    #get_support_sample_rate()
    input_device = get_input_device()
    create_output_folder()
    #openwakeword.utils.download_models(['embedding_model', 'hey_jarvis_v0.1', 'melspectrogram', 'silero_vad'])
    logging.info(f'Cuda: {torch.cuda.is_available()}')
    llm_model = LLM_Handler()

    owwModel = openwakeword.Model(
        wakeword_models                = ["hey_jarvis"],
        inference_framework            = "onnx",
        enable_speex_noise_suppression = True
    )
    vad_model    = load_silero_vad()
    audio_buffer = deque(maxlen=10)
    CHUNK        = 4096
    FORMAT       = pyaudio.paInt16
    CHANNELS     = 1
    MIC_SR       = 16000
    audio        = pyaudio.PyAudio()
    mic_stream = audio.open(format             = FORMAT,
                            channels           = CHANNELS,
                            rate               = MIC_SR,
                            input              = True,
                            input_device_index = input_device,
                            frames_per_buffer  = CHUNK)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    logging.info('\n\n\nStart listen for wakeword')
    file_num = 0
    while True:
        mic_audio = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        audio_buffer.append(mic_audio)
        prediction = owwModel.predict(mic_audio)


        for mdl in prediction.keys():
            if prediction[mdl] >= 0.3:
                recorded_audio = capture_audio_after_wakeword(vad_model, audio_buffer)
                if (len(recorded_audio) / MIC_SR) <= 1.05:
                    audio_buffer.clear()
                    owwModel.reset()
                    logging.info('\n\n\nStart listen for wakeword')
                    break

                file_num = file_num + 1
                write_samples(f"out_{file_num}", recorded_audio, samplerate=16000)

                if isinstance(recorded_audio, np.ndarray):
                    recorded_audio = recorded_audio.tolist()
                logging.info(f'Call whisper service to transcribe: {len(recorded_audio)} samples')
                whisper_url = f"http://{get_running_ip()}:8013/transcribe/"
                response    = requests.post(whisper_url,json={"audio_input": recorded_audio})
                result      = response.json()
                text        = result['transcription']
                logging.info(f'Text: {text}')

                command = llm_model.run_llm(text)
                command = command['command']
                logging.info(f'Command: {command}')

                if command != "None":
                    # --- TTS
                    text_to_user = command.replace("_", " ")
                    play_text(text_to_user)

                    # Send
                    send_command(command)
                else:
                    play_text("Please say again")

                audio_buffer.clear()
                owwModel    .reset()
                logging.info('\n\n\nStart listen for wakeword')
                break




