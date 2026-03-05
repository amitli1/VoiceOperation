from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import json
import os
import torch
import jsonref
from pathlib import Path


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
            "lcuPower": 0,
            "mcuPower": 0
        }
    ]

    return power_source_status_msg


def ask_question(question, status_data, schema_data):

    MODEL_NAME = "Qwen/Qwen3-1.7B"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model      = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map        = "auto",
        dtype             = torch.float16,
        trust_remote_code = True
    )
    prompt = f"""You are a system assistant. 
                 Use the following JSON data and Schema to answer the user's question.
                
                Data:
                {json.dumps(status_data, indent=2)}
                
                Schema Context:
                {json.dumps(schema_data, indent=2)}  
                
                Question: {question}
                Answer:"""

    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens        = 1000,
        do_sample             = False,
        pad_token_id          = tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()


if __name__ == '__main__':

    print(f"CUDA: {torch.cuda.is_available()}")

    schema     = load_full_schema()
    schema     = get_relevant_schema_part(schema)
    schema     = simplify_schema(json.loads(schema))

    status_msg = load_status_msg()

    user_q     = "What is the current voltage level of the battery?"
    result     = ask_question(user_q, status_msg, schema)
    print(f"Q: {user_q}")
    print(f"A: {result}")