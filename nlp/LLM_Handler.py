from transformers import AutoTokenizer, AutoModelForCausalLM
import outlines
import torch
import logging
import os


class LLM_Handler:

    def __init__(self, model_name='Qwen/Qwen3-0.6B'):
        #model_name      = "Qwen/Qwen2.5-0.5B"
        #model_name     = "Qwen/Qwen3-0.6B"
        #model_name = "Qwen/Qwen3-1.7B"
        if self.in_docker():
            model_name = "/models/Qwen/Qwen3-0.6B"
        logging.info(f'Load: {model_name}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_model  = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype      = "auto",
            device_map = "auto"
        )
        self.hf_model.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.hf_model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.hf_model.generation_config.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )
        self.model = outlines.models.Transformers(self.hf_model, self.tokenizer)

        #with open("nlp/system_prompt.txt", "r", encoding="utf-8") as f:
        with open("nlp/system_no_examples.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        with open("nlp/json_schema.txt", "r", encoding="utf-8") as f:
            self.json_schema = f.read()

        self.json_gen = outlines.generate.json(self.model, self.json_schema)

    def in_docker(self):
        return os.path.exists("/.dockerenv") or os.path.exists("/run/.dockerenv")

    def run_llm(self, user_text):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text}
        ]

        # Build prompt using Qwen chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize              = False, # return chat template (string)
            add_generation_prompt = True,
            enable_thinking       = False,
            temperature           = 0.0

        )

        result = self.json_gen(prompt)
        return result