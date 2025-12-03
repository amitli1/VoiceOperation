from main import load_llm, run_llm
import logging

llm_tokenizer, llm_model, system_prompt = load_llm()

def run_test(user_text, gt_command):
    command = run_llm(llm_tokenizer, llm_model, system_prompt, text)
    if command != gt_command:
        logging.info(f'Failed: {user_text}')
    else:
        logging.info(f'PASS: {user_text}')

if __name__ == "__main__":


    #text = "open general"
    text = "how much battery do I have?"
    command = run_llm(llm_tokenizer, llm_model, system_prompt, text)
    logging.info(f'Command: {command}')
