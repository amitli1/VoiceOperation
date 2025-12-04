from nlp.LLM_Handler     import LLM_Handler
from collections         import deque
from silero_vad          import load_silero_vad, get_speech_timestamps
from app_config.settings import app_settings
import numpy             as np
import logging
import torch
import os
import openwakeword
import pyaudio
import time
import requests
import json
from datetime import datetime


def get_timestamp_string():
    return datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

def init_logger():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(   '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')

    # # File handler
    os.makedirs("logs", exist_ok=True)
    CURRENT_DATE = get_timestamp_string()
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

def in_docker():
 return os.path.exists("/.dockerenv") or os.path.exists("/run/.dockerenv")

def get_running_ip():
    if in_docker():
        return "host.docker.internal"
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

#
# def get_speech_status(vad_model, audio_chunk, sample_rate=app_settings.audio.vad.sample_rate):
#     audio_chunk = audio_chunk.astype(np.float32) / 32768.0
#     speech_prob = vad_model(torch.from_numpy(audio_chunk), sample_rate).item()
#     return speech_prob > app_settings.audio.vad.vad_threshold
#
# def source_capture_audio_after_wakeword(last_audios):
#         recorded_audio = []
#         for audio_history in last_audios:
#             recorded_audio.append(audio_history)
#         silence_duration = 0
#         silence_threshold = 0.4
#         grace_period = 0.8
#         grace_time_elapsed = 0.0
#         speech_detected = False
#
#         logging.info("[***] Capturing speech... (online process)")
#         start_time = time.time()
#
#         while True:
#             try:
#                 mic_audio = np.frombuffer(
#                     mic_stream.read(app_settings.audio.vad.vad_chunk, exception_on_overflow=False),
#                     dtype=np.int16
#                 )
#             except IOError:
#                 logging.error("Error reading from audio stream.")
#                 break
#             recorded_audio.append(mic_audio)
#
#             if not speech_detected:
#                 if get_speech_status(mic_audio):
#                     speech_detected = True
#                     logging.info(f"[pid = {os.getpid()}] Speech detected. Now monitoring for silence.")
#                 else:
#                     grace_time_elapsed += app_settings.audio.vad.vad_chunk / app_settings.audio.vad.sample_rate
#                     if grace_time_elapsed >= grace_period:
#                         logging.info("No speech detected during grace period. Stopping capture.")
#                         break
#             else:
#                 if not self.get_speech_status(mic_audio):
#                     silence_duration += app_settings.audio.vad.vad_chunk / app_settings.audio.vad.sample_rate
#                     if silence_duration >= silence_threshold:
#                         logging.info("Silence detected, stopping capture.")
#                         break
#                 else:
#                     silence_duration = 0
#
#         elapsed_time = time.time() - start_time
#         logging.info(f"[***] Audio capturing took {elapsed_time:.2f} seconds.")
#         full_audio = np.concatenate(recorded_audio).astype(np.float32)
#         return full_audio



def send_command(user_command):
    command  = f'http://localhost:8080/{user_command}'
    #response = requests.post(command, json={})
    #logging.info(f"Command status code: {response.status_code}")
    #logging.info(f"Response body      : {response.text}")

if __name__ == "__main__":
    logging.info('Start')
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
    mic_stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=MIC_SR,
                            input=True,
                            frames_per_buffer=CHUNK)

    logging.info('\n\n\nStart listen for wakeword')
    while True:
        mic_audio = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        audio_buffer.append(mic_audio)
        prediction = owwModel.predict(mic_audio)


        for mdl in prediction.keys():
            if prediction[mdl] >= 0.3:
                recorded_audio = capture_audio_after_wakeword(vad_model, audio_buffer)
                if (len(recorded_audio) / MIC_SR) <= 1.05:
                    audio_buffer.clear()
                    logging.info('\n\n\nStart listen for wakeword')
                    continue

                if isinstance(recorded_audio, np.ndarray):
                    recorded_audio = recorded_audio.tolist()
                response = requests.post(f"http://{get_running_ip()}:8013/transcribe/",json={"audio_input": recorded_audio})
                result   = response.json()
                text     = result['transcription']
                logging.info(f'Text: {text}')

                command = llm_model.run_llm(text)
                logging.info(f'Command: {command}')
                send_command(command)

                audio_buffer.clear()
                logging.info('\n\n\nStart listen for wakeword')




