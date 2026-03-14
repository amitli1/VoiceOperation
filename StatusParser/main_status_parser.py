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
import logging

from StatusParser.power_status_manger import PowerStatusManger
from utils import play_text

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_NAME = "Qwen/Qwen3-1.7B"
#MODEL_NAME = "Qwen/Qwen3-4B-FP8"
#MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

def try_me(full_schema):
    get_schemas = {}
    paths = full_schema.get('paths', {})

    for path_name, methods in paths.items():
        # Check if 'get' exists for this specific path
        if 'get' in methods:
            try:
                # Navigate the nested structure safely
                responses = methods['get'].get('responses', {})
                # Look for 200 OK, then content, then the JSON schema
                success_response = responses.get('200', {})
                schema = success_response.get('content', {}).get('application/json', {}).get('schema')

                if schema:
                    get_schemas[path_name] = schema
            except (KeyError, TypeError):
                continue

    # If we found nothing in paths, fallback to components
    if not get_schemas:
        return full_schema.get('components', {}).get('schemas', {})

    return get_schemas

def get_relevant_schema_part(full_schema):

    if True:
        return try_me(full_schema)

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
    if not isinstance(schema, dict):
        return schema

        # Define the "DNA" of the schema we want to keep
    allowed_keys = {'type', 'description', 'enum', 'x-enumNames'}

    # Create a new dict with only the allowed keys
    simplified = {k: simplify_schema(v) for k, v in schema.items() if k in allowed_keys}

    # Special handling for 'properties' to ensure we recurse into nested objects
    if 'properties' in schema:
        simplified['properties'] = {
            k: simplify_schema(v) for k, v in schema['properties'].items()
        }

    # Special handling for 'items' in arrays
    if 'items' in schema:
        simplified['items'] = simplify_schema(schema['items'])

    return simplified

def load_full_schema(main_json_path):

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


def load_power_status_msg():
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



def get_schema(main_json_path):
    schema     = load_full_schema(main_json_path)
    schema     = get_relevant_schema_part(schema)
    schema     = simplify_schema(schema) #json.loads(schema)
    return schema



def main():

    tokenizer, model  = create_llm_model(MODEL_NAME)
    powerStatusManger = PowerStatusManger(tokenizer, model)

    status_msg = load_power_status_msg()

    user_q = "What is the current voltage level of the battery ?"
    user_q = "Is the battery in charging mode ?"
    user_q = "What is the LCU and MCU battery power ?"
    #user_q = "What is LMU power version ?"
    #user_q = "What is the car battery power ?"

    result = powerStatusManger.ask_llm(tokenizer, model, user_q, status_msg)
    print(f"Q: {user_q}")
    print(f"A: {result}")

    play_text(user_q)
    play_text(result)

def run_unit_tests():
    from StatusParser.status_tester_manager import StatusTesterManager

    statusTesterManager = StatusTesterManager(MODEL_NAME)
    statusTesterManager.run_tests()

def test_load_schema():

    shcema_path  = rf'{os.getcwd()}/StatusParser/LMU.PowerSource.Service.OpenAPI.Spec/lmu-power-source-api.json'
    #shcema_path = rf'{os.getcwd()}/StatusParser/LMU.MCU_LRADS.BFF.OpenAPI.Spec/lmu-mcu-lrads-bff-api.json'
    #shcema_path = rf'{os.getcwd()}/StatusParser/LMU.Navigation.BFF.OpenAPI.Spec/lmu-navigation-bff-api.json'
    #shcema_path = rf'{os.getcwd()}/StatusParser/LMU.TimeManagement.BFF.OpenAPI.Spec/lmu-time-management-bff-api.json'
    power_schema = get_schema(shcema_path)
    #print(json.dumps(power_schema, indent=4))
    print(power_schema)

if __name__ == '__main__':

    print(f"CUDA: {torch.cuda.is_available()}")
    #test_load_schema()

    #main()
    run_unit_tests()

