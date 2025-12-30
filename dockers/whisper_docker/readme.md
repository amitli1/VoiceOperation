## Notes:
- faster-whisper → ctranslate2 is NOT using PyTorch’s cuDNN.
- It dynamically loads system cuDNN shared libraries, specifically:
- ctranslate2 wheel was built against cuDNN ≥ 9.1