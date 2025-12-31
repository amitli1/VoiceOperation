from kokoro  import KPipeline
from pathlib import Path
import numpy as np
import os
import soundfile as sf
import librosa

def create_wav(tts, text, file_path):
    generator  = tts(text, voice='am_adam')
    audios     = [audio.tolist() for _, _, audio in generator]
    full_audio = np.concatenate(audios, axis=0)

    data_resampled = librosa.resample(full_audio, orig_sr=24000, target_sr=48000)
    sf.write(file_path, data_resampled, 48000)

if __name__ == "__main__":

    pipeline  = KPipeline(lang_code='a', device='cuda')
    full_path = Path(os.getcwd()).parent.parent
    create_wav(pipeline, "show overview"    , f"{full_path}/audio_files/show_overview.wav")
    create_wav(pipeline, "show power screen", f"{full_path}/audio_files/show_power_screen.wav")
    create_wav(pipeline, "show navigation"  , f"{full_path}/audio_files/show_navigation.wav")
    create_wav(pipeline, "show inventory"   , f"{full_path}/audio_files/show_inventory.wav")
    create_wav(pipeline, "Please say again" , f"{full_path}/audio_files/Please_say_again.wav")
