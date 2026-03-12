from transformers     import AutoTokenizer, AutoModelForCausalLM
from pathlib          import Path
import numpy          as np
import sounddevice    as sd
import logging
import json
import os
import torch
import jsonref
import requests
from utils import play_text
import time

class PowerStatusManger:
    def __init__(self, tokenizer, model):
        self.schema     = self._get_schema()
        self.tokenizer  = tokenizer
        self.model      = model

    def handle_power_status(self, user_q):
        status_msg = self.load_status_msg()
        result     = self.ask_llm(self.tokenizer, self.model, user_q, status_msg, self.schema)
        logging.info(f"Q: {user_q}")
        logging.info(f"A: {result}")
        play_text(result)

    def ask_llm(self, tokenizer, model, question, status_data, schema_data):

        examples = """
        Example 1:
        Data: [{"batteryStatus": {"charging": 1}}]
        Question: Is the battery in charging mode ?
        Answer: Yes, the battery is charging.

        Example 2:
        Data: [{"batteryStatus": {"charging": 0}}]
        Question: Is the battery in charging mode ?
        Answer: No, the battery is not charging.

       Example 3:
        Data: [{"batteryStatus": {"voltageLevel": 75}}]
        Question: what is the battery voltage level ?
        Answer: The battery voltage level is 75.

        ---
        """

        prompt = f"""Task: Answer the question using ONLY the data provided. 
        Constraint:
        - Provide a direct answer.
        - The answer must be a full sentence that includes the question topic.
        - Do not explain how you derived the answer.
        - If the answer cannot be found in the data, reply exactly: "The requested information cannot be determined from the provided data."

        Data: {json.dumps(status_data, indent=2)}
        Schema: {json.dumps(schema_data, indent=2)}

        Question: {question}
        Answer:"""

        prompt    = examples + prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        logging.info(f"Number of prompt tokens: {input_ids.shape[1]}" )
        start_time = time.time()
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=False,
            tokenizer=tokenizer,
            stop_strings=["\n", "Answer:", "Question:", "Explanation:"],
            pad_token_id=tokenizer.pad_token_id
        )

        generated_tokens    = outputs[0][input_ids.shape[-1]:]  # get answer (after prompt)
        full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        end_time            = time.time()
        logging.info(f'Total run time: {(end_time-start_time):.2f} seconds')
        final_answer = full_generated_text.strip().split('\n')[0].strip()
        return final_answer

    def _get_schema(self):
        schema = self._load_full_schema()
        schema = self._get_relevant_schema_part(schema)
        schema = self._simplify_schema(json.loads(schema))
        return schema

    def _get_relevant_schema_part(self, full_schema):
        try:
            target = \
            full_schema['paths']['/power-source-data']['get']['responses']['200']['content']['application/json'][
                'schema']
            return json.dumps(target, indent=2)
        except KeyError:
            return json.dumps(full_schema.get('components', {}), indent=2)

    def _proxy_to_dict(self, obj):
        # convert jsonref to standart dict
        if isinstance(obj, jsonref.JsonRef):
            return self._proxy_to_dict(obj.__subject__)
        elif isinstance(obj, dict):
            return {k: self._proxy_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._proxy_to_dict(item) for item in obj]
        return obj

    def _simplify_schema(self, schema):
        """
        Extracts a simplified version of the JSON Schema, retaining only 'type' and 'description'.
        This reduces token noise by removing metadata like 'x-stoplight', 'xml', and 'default'
        which often cause small LLMs (like Qwen 1.7B) to hallucinate or repeat text.

        Args:
            schema (dict): The full JSON Schema object.

        Returns:
            dict: A cleaned dictionary containing field types and their descriptions.
        """
        simplified = {}

        # Handle schemas that define an array of items (common in status messages)
        root_items = schema.get("items", schema)
        properties = root_items.get("properties", {})

        for prop_name, details in properties.items():
            # Handle nested objects (e.g., batteryStatus)
            if details.get("type") == "object" and "properties" in details:
                nested_props = {}
                for sub_name, sub_details in details["properties"].items():
                    nested_props[sub_name] = {
                        "type": sub_details.get("type"),
                        "description": sub_details.get("description", "N/A").strip()
                    }
                simplified[prop_name] = {
                    "type": "object",
                    "properties": nested_props
                }
            else:
                # Handle standard top-level fields
                simplified[prop_name] = {
                    "type": details.get("type"),
                    "description": details.get("description", "N/A").strip()
                }

        return simplified

    def _load_full_schema(self):

        main_json_path = rf'{os.getcwd()}/StatusParser/LMU.PowerSource.Service.OpenAPI.Spec/lmu-power-source-api.json'

        try:
            with open(main_json_path, 'r') as f:
                data = json.load(f)

            base_uri = f"file://{os.path.abspath(main_json_path)}"
            resolved = jsonref.replace_refs(data, base_uri=base_uri)

            # replace  $ref with the relevant json
            res = self._proxy_to_dict(resolved)

        except Exception as e:
            logging.error(e)
            res = ""

        return res

    def load_status_msg(self):
        power_source_status_msg = [
            {
                "acPower": 0,
                "vehiclePower": 0,
                "version": "2.0",
                "defaultTopicName": "PowerSourceData",
                "batteryStatus": {
                    "charging": 0,
                    "voltageLevel": 10
                },
                "insPower": 0,
                "gfePower": 0,
                "lcuPower": 9,
                "mcuPower": 2
            }
        ]

        return power_source_status_msg