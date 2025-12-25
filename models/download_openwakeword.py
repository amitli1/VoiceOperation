import openwakeword

if __name__ == '__main__':

    openwakeword.utils.download_models(['embedding_model', 'hey_jarvis_v0.1', 'melspectrogram', 'silero_vad'],
                                       target_directory = "/home/amitli/repo/VoiceOperation/models/openWakeWord/")