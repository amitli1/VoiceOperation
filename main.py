from app_config.settings import app_settings
import logging
import torch
import os
import openwakeword
import numpy as np
from collections           import deque
import pyaudio

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


if __name__ == "__main__":
    logging.info('Start')
    openwakeword.utils.download_models(['embedding_model', 'hey_jarvis_v0.1', 'melspectrogram', 'silero_vad'])
    logging.info(f'Cuda: {torch.cuda.is_available()}')

    owwModel = openwakeword.Model(
        wakeword_models                = ["hey_jarvis"],
        inference_framework            = "onnx",
        enable_speex_noise_suppression = True
    )

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
    logging.info('Start listen for wakeword')
    while True:
        mic_audio = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        audio_buffer.append(mic_audio)
        prediction = owwModel.predict(mic_audio)


        for mdl in prediction.keys():
            if prediction[mdl] >= 0.3:
                logging.info('--- wakeword ---')

