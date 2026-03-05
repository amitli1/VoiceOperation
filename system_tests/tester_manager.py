import logging
import multiprocessing
import time
from queue import Queue
import os
from pydub import AudioSegment
import numpy as np
from app_config.settings import app_settings

class TesterManager:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance           = super(TesterManager, cls).__new__(cls)
            cls._instance.testQueue = multiprocessing.Queue()
            cls._instance.test_step = multiprocessing.Value('i', 0)
        return cls._instance


    def get_test_queue(self):
        return self.testQueue

    def load_file(self, file_name):
        audio = AudioSegment.from_wav(file_name)
        samples = np.array(audio.get_array_of_samples())
        samples = samples / 32768.0
        return samples


    def log_step(self, text):

        text        = f'[Step: {self.test_step.value}] {text}'
        width       = 70
        border      = '*' * width
        inner_width = width - 2  # space for side asterisks
        centered_text = f'* {text.center(inner_width - 2)} *'
        logging.info(border)
        logging.info(centered_text)
        logging.info(border)

    def log_fail(self, text):

        text        = f'{text}'
        width       = 70
        border      = '*' * width
        inner_width = width - 2  # space for side asterisks
        centered_text = f'* {text.center(inner_width - 2)} *'
        logging.error(border)
        logging.error(centered_text)
        logging.error(border)

    def run_next_test_step(self, curr_step_results=None):

        if app_settings.test.use_case == 'general':
            #p = multiprocessing.Process(target=self._run_general_test, args=(curr_step_results,))
            #p.start()
            return self._run_general_test(None)

    def _run_general_test(self, curr_step_results=None):
        time.sleep(1)  # wait for other tasks to finish
        self.test_step.value = self.test_step.value + 1
        if self.test_step.value == 1:
            self.log_step(f'Load: Go_back_to_home_screen.wav')
            samples = self.load_file(rf'./system_tests/wav_commands/Go_back_to_home_screen.wav')
            #self.testQueue.put(samples)
            return samples
        elif self.test_step.value == 2:
            self.log_step(f'Load: Missiles_status.wav')
            samples = self.load_file(rf'./system_tests/wav_commands/Missiles_status.wav')
            #self.testQueue.put(samples)
            return samples
        elif self.test_step.value == 3:
            self.log_step(f'Load: Open_power_screen.wav')
            samples = self.load_file(rf'./system_tests/wav_commands/Open_power_screen.wav')
            #self.testQueue.put(samples)
            return samples
        elif self.test_step.value == 4:
            self.log_step(f'Load: Where_am_I_located.wav')
            samples = self.load_file(rf'./system_tests/wav_commands/Where_am_I_located.wav')
            #self.testQueue.put(samples)
            return samples
        else:
            self.log_step(f'Load: None')
            #self.testQueue.put(None)
            return None

    def check_last_results(self, command):

        icon = ""
        if self.test_step.value == 1:
            if command == "show_overview":
                icon = "✅"
            else:
                icon = "🛑"
        elif self.test_step.value == 2:
            if command == "show_inventory":
                icon = "✅"
            else:
                icon = "🛑"
        elif self.test_step.value == 3:
            if command == "show_power_screen":
                icon = "✅"
            else:
                icon = "🛑"
        elif self.test_step.value == 4:
            if command == "show_navigation":
                icon = "✅"
            else:
                icon = "🛑"

        logging.info(f'\tCurrent test results: {icon}')