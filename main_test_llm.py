from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os

if __name__ == "__main__":

    print(f"CUDA: {torch.cuda.is_available()}")
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype       = "auto",
        device_map  = "auto"
    )

    # load system prompt
    with open("nlp/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # prepare the model input
    user_prompt = "how much rockets do we have ?"
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    output_tokens = model.generate(
        **inputs,
        max_new_tokens = 256,
        do_sample      = False,
    )

    response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    start         = response_text.rfind("{")
    end           = response_text.rfind("}") + 1
    json_block    = response_text[start:end]

    try:
        result = json.loads(json_block)
        print("\nParsed result:", result)
    except Exception as e:
        print("Failed to parse JSON:", e)
