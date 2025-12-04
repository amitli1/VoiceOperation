from typing import Dict
from pydantic import BaseModel
import yaml

class STTConfig(BaseModel):
    model_size: str
    num_beams: int
    confidence_threshold: float
    compression_ratio_threshold: float

class VadConfig(BaseModel):
    vad_threshold: float
    vad_chunk: int
    sample_rate: int

class WakewordConfig(BaseModel):
    channels: int
    chunk: int
    sample_rate: int


class AudioConfig(BaseModel):
    vad_threshold: float
    language: str
    stt: STTConfig
    vad: VadConfig
    wakeword: WakewordConfig

class TestConfig(BaseModel):
    run_in_test_mode: bool
    use_case: str


class Settings(BaseModel):
    audio: AudioConfig
    test: TestConfig

# amitli
def load_config(path: str = "app_config/conf.yaml") -> Settings:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Settings(**data)

app_settings = load_config()
