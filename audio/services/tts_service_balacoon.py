from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import numpy as np
from balacoon_tts import TTS
import uvicorn
import os
import torch
import logging
import glob

from torch.xpu import device


def in_docker():
 return os.path.exists("/.dockerenv") or os.path.exists("/run/.dockerenv")

app    = FastAPI()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.info("CUDA: {}".format(torch.cuda.is_available()))

def load_tts(addon_path=None):
    tts                = TTS(addon_path)
    supported_speakers = tts.get_speakers()
    speaker            = supported_speakers[-1]
    return tts, speaker

# Load the TTS model once
if in_docker():
    logging.info(f"Start loading balacoon_tts model (in docker)")
    ##tts, speaker = load_tts('/app/en_us_cmartic_jets_gpu.addon')
    tts, speaker = load_tts('/app/en_us_cmuartic_jets_cpu.addon')
    logging.info(f"\tloaded")
else:
    logging.info('Start loading balacoon_tts model (not docker)')
    cwd          = os.getcwd()
    repo_root    = os.path.abspath(os.path.join(cwd, "../../"))
    tts_dir      = os.path.join(repo_root, "dockers/tts_docker/uk_ltm_jets_cpu.addon")
    tts, speaker = load_tts(tts_dir)


SAMPLE_RATE = 24000

class TTSRequest(BaseModel):
    text: str

@app.post("/synthesize/")
async def synthesize_tts(request: TTSRequest):

    text = request.text
    try:
        samples = tts.synthesize(text, speaker)
        return {
            "audio": samples.tolist(),
            "sample_rate": SAMPLE_RATE
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    logging.info("CUDA: {}".format(torch.cuda.is_available()))
    uvicorn.run(app, host="0.0.0.0", port=8002)
