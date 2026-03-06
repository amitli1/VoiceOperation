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


def get_relevant_schema_part(full_schema):
    try:
        target = full_schema['paths']['/power-source-data']['get']['responses']['200']['content']['application/json']['schema']
        return json.dumps(target, indent=2)
    except KeyError:
        return json.dumps(full_schema.get('components', {}), indent=2)


def proxy_to_dict(obj):
    # convert jsonref to standart dict
    if isinstance(obj, jsonref.JsonRef):
        return proxy_to_dict(obj.__subject__)
    elif isinstance(obj, dict):
        return {k: proxy_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [proxy_to_dict(item) for item in obj]
    return obj


def simplify_schema(schema):
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

def load_full_schema():


    main_json_path = rf'{os.getcwd()}/LMU.PowerSource.Service.OpenAPI.Spec/lmu-power-source-api.json'
    base_uri       = Path(main_json_path).parent.as_uri() + '/'

    try:
        with open(main_json_path, 'r') as f:
            data = json.load(f)

        base_uri = f"file://{os.path.abspath(main_json_path)}"
        resolved = jsonref.replace_refs(data, base_uri=base_uri)

        # replace  $ref with the relevant json
        res = proxy_to_dict(resolved)

    except Exception as e:
        print(e)
        res = ""

    return res


def load_status_msg():
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


def create_llm_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map        = "auto",
        dtype             = torch.float16,
        trust_remote_code = True
    )
    return tokenizer, model

def ask_llm(tokenizer, model, question, status_data, schema_data):

    examples = """
    Example 1:
    Data: [{"batteryStatus": {"charging": 1}}]
    Question: Is the battery in charging mode ?
    Answer: Yes

    Example 2:
    Data: [{"batteryStatus": {"charging": 0}}]
    Question: Is the battery in charging mode ?
    Answer: No
    
   Example 3:
    Data: [{"batteryStatus": {"voltageLevel": 75}}]
    Question: what is the battery voltage level ?
    Answer: 75

    ---
    """

    prompt = f"""Task: Answer the question using ONLY the data provided. 
    Constraint: Provide a direct answer. No explanations, no intro, no context.

    Data: {json.dumps(status_data, indent=2)}
    Schema: {json.dumps(schema_data, indent=2)}
    
    Question: {question}
    Answer:"""


    prompt    = examples + prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs  = model.generate(
        input_ids,
        max_new_tokens = 100,
        do_sample      = False,
        tokenizer      = tokenizer,
        stop_strings   = ["\n", "Answer:", "Question:", "Explanation:"],
        pad_token_id   = tokenizer.pad_token_id
    )


    generated_tokens    = outputs[0][input_ids.shape[-1]:] # get answer (after prompt)
    full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    final_answer        = full_generated_text.strip().split('\n')[0].strip()
    return final_answer


def get_schema():
    schema     = load_full_schema()
    schema     = get_relevant_schema_part(schema)
    schema     = simplify_schema(json.loads(schema))
    return schema



def main():
    # MODEL_NAME = "Qwen/Qwen3-0.6B"
    MODEL_NAME       = "Qwen/Qwen3-1.7B"
    tokenizer, model = create_llm_model(MODEL_NAME)

    schema     = get_schema()
    status_msg = load_status_msg()

    user_q = "What is the current voltage level of the battery ?"
    user_q = "Is the battery in charging mode ?"
    user_q = "What is the LCU and MCU battery power ?"
    #user_q = "What is LMU power version ?"

    result = ask_llm(tokenizer, model, user_q, status_msg, schema)
    print(f"Q: {user_q}")
    print(f"A: {result}")

    play_text(user_q)
    play_text(result)

def run_unit_tests():
    from StatusParser.status_tester_manager import StatusTesterManager

    # MODEL_NAME = "Qwen/Qwen3-0.6B"
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    statusTesterManager = StatusTesterManager(MODEL_NAME)
    statusTesterManager.run_tests()

if __name__ == '__main__':

    print(f"CUDA: {torch.cuda.is_available()}")

    main()
    #run_unit_tests()
