import sounddevice as sd
import numpy       as np
import requests
import time

if __name__ == "__main__":
    url      = "http://127.0.0.1:8002/synthesize/"
    payload  = {"text": "Hello world, this is a test."}
    start    = time.time()
    response = requests.post(url, json=payload)
    end      = time.time()
    data     = response.json()

    print(f"Finished in {(end - start):.2f} seconds")
    if "audio" in data:
        audio       = np.array(data["audio"][0], dtype=np.float32)  # assuming one audio
        sample_rate = data["sample_rate"]

        sd.play(audio, samplerate=sample_rate)
        sd.wait()  # Wait until audio finishes