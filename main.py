from StatusParser.power_status_manger import PowerStatusManger
from nlp.LLM_Handler     import LLM_Handler
from collections         import deque
from silero_vad          import load_silero_vad, get_speech_timestamps
from fastapi             import FastAPI, Request
from scipy.io.wavfile    import write
from datetime            import datetime
import numpy             as np
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
import librosa
from app_config.settings import app_settings
from system_tests.tester_manager import TesterManager
from utils import get_input_device, get_output_device, get_running_ip, play_wav_file
from pydub import AudioSegment


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


def warmup():
    try:
        logging.info(f'Start TTS warmup')
        response = requests.post(f"http://{get_running_ip()}:8002/synthesize/", json={"text": 'warmup TTS'})
        data = response.json()
        if response.status_code != 200:
            logging.error(f'TTS warmup failed, status code {response.status_code}')
        else:
            logging.info(f'End TTS warmup')
    except Exception as e:
        logging.error(f'TTS warmup failed: {e}')


    try:
        logging.info('Start WHISPER warmup')
        audio   = AudioSegment.from_wav(f"{os.getcwd()}/audio_files/Please_say_again.wav")
        samples = np.array(audio.get_array_of_samples())
        samples = samples / 32768.0
        audio_input = samples.tolist()
        response = requests.post(f"http://{get_running_ip()}:8013/transcribe/", json={"audio_input": audio_input})
        if response.status_code != 200:
            logging.error(f'Whisper warmup failed with status_code: {response.status_code != 200}')
        else:
            logging.info(f'End TTS warmup')
    except Exception as e:
        logging.error(f'Whisper warmup failed: {e}')



if __name__ == "__main__":

    logging.info('Start')

    if app_settings.test.run_in_test_mode :
        TesterManager()

    #get_support_sample_rate()
    input_device  = get_input_device()
    output_device = get_output_device()
    create_output_folder()
    #openwakeword.utils.download_models(['embedding_model', 'hey_jarvis_v0.1', 'melspectrogram', 'silero_vad'])
    logging.info(f'Cuda: {torch.cuda.is_available()}')
    llm_model           = LLM_Handler(model_name='Qwen/Qwen3-0.6B')
    power_status_manger = PowerStatusManger(*llm_model.get_llm_model()) # * -> unpack

    warmup()

    owwModel = openwakeword.Model(
        wakeword_models                = ["hey_jarvis"],
        inference_framework            = "onnx",
        enable_speex_noise_suppression = True
    )
    vad_model    = load_silero_vad()
    audio_buffer = deque(maxlen=10)
    CHUNK        = app_settings.audio.wakeword.chunk
    FORMAT       = pyaudio.paInt16
    CHANNELS     = app_settings.audio.wakeword.channels
    MIC_SR       = app_settings.audio.wakeword.sample_rate
    audio        = pyaudio.PyAudio()
    mic_stream = audio.open(format             = FORMAT,
                            channels           = CHANNELS,
                            rate               = MIC_SR,
                            input              = True,
                            input_device_index = input_device,
                            frames_per_buffer  = CHUNK)

    logging.warning('comment code - open restapi to get data from user')
    # server_thread = threading.Thread(target=run_server, daemon=True)
    # server_thread.start()

    logging.info('\n\n\nStart listen for wakeword')
    file_num = 0

    while True:

        wake_word_detected = False

        if app_settings.test.run_in_test_mode is True:
            wake_word_detected = True
            recorded_audio     = TesterManager().run_next_test_step()
            if recorded_audio is None:
                break
        else:
            mic_audio = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            audio_buffer.append(mic_audio)
            prediction = owwModel.predict(mic_audio)

            for mdl in prediction.keys():
                if prediction[mdl] >= 0.3:
                    recorded_audio = capture_audio_after_wakeword(vad_model, audio_buffer)
                    if (len(recorded_audio) / MIC_SR) <= 1.05:
                        audio_buffer.clear()
                        owwModel.reset()
                        logging.info(f'Wake word detected with: {prediction[mdl]}% but audio is too short: {(len(recorded_audio) / MIC_SR)} seconds')
                        break
                    wake_word_detected = True
                    logging.info(f'Wake word detected with: {prediction[mdl]}%')

        if wake_word_detected:
            file_num = file_num + 1
            write_samples(f"out_{file_num}", recorded_audio, samplerate=16000)

            if isinstance(recorded_audio, np.ndarray):
                recorded_audio = recorded_audio.tolist()
            whisper_url = f"http://{get_running_ip()}:8013/transcribe/"
            response    = requests.post(whisper_url,json={"audio_input": recorded_audio})
            result      = response.json()
            text        = result['transcription']
            logging.info(f'Text: {text}')

            command = llm_model.run_llm(text)
            command = command['command']
            logging.info(f'Command: {command}')

            if command != "None":
                # # --- TTS
                # text_to_user = command.replace("_", " ")
                # play_text(text_to_user)
                #
                # # Send
                # send_command(command)
                play_wav_file(f"audio_files/{command}.wav", output_device)

                if command == "show_power_screen":
                    power_status_manger.handle_power_status(text)

            else:
                #play_text("Please say again")
                play_wav_file("audio_files/Please_say_again.wav", output_device)

            if app_settings.test.run_in_test_mode is True:
                TesterManager().check_last_results(command)
            else:
                audio_buffer.clear()
                owwModel    .reset()
                logging.info('\n\n\nStart listen for wakeword')





