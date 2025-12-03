from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import torch
import os
import openwakeword
import numpy as np
from collections           import deque
from silero_vad            import load_silero_vad, get_speech_timestamps
import pyaudio
import time
import requests
import json

def init_logger():


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(   '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')

    # # File handler
    # os.makedirs("/mnt/nvme/outputs", exist_ok=True)
    # CURRENT_DATE = get_timestamp_string()
    # log_name     = f"/mnt/nvme/outputs/log_{CURRENT_DATE}.txt"
    # file_handler = logging.FileHandler(log_name)
    # file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

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


def load_llm():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )

    with open("nlp/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    return tokenizer, model, system_prompt

def run_llm(tokenizer, model, system_prompt, user_prompt):

    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    output_tokens = model.generate(
        **inputs,
        max_new_tokens = 256,
        do_sample      = False,
    )

    response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    start         = response_text.rfind("{")
    end           = response_text.rfind("}") + 1
    json_block    = response_text[start:end]

    try:
        result = json.loads(json_block)
    except Exception as e:
        logging.error(f'Error while load json: {e}')
        result = None

    return result

def send_command(command):
    response = requests.post(command, json={})
    logging.info(f"Command status code: {response.status_code}")
    logging.info(f"Response body      : {response.text}")

if __name__ == "__main__":
    logging.info('Start')
    #openwakeword.utils.download_models(['embedding_model', 'hey_jarvis_v0.1', 'melspectrogram', 'silero_vad'])
    logging.info(f'Cuda: {torch.cuda.is_available()}')

    llm_tokenizer, llm_model, system_prompt = load_llm()

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
                logging.info('--- wakeword ---')
                recorded_audio = capture_audio_after_wakeword(vad_model, audio_buffer)

                if isinstance(recorded_audio, np.ndarray):
                    # Convert numpy array to list for JSON serialization
                    recorded_audio = recorded_audio.tolist()
                response = requests.post(f"http://{get_running_ip()}:8013/transcribe/",json={"audio_input": recorded_audio})
                result   = response.json()
                text     = result['transcription']
                logging.info(f'Text: {text}')

                command = run_llm(llm_tokenizer, llm_model, system_prompt, text)
                logging.info(f'Command: {command}')
                send_command(command)




