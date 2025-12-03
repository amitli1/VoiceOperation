from main import load_llm, run_llm
import logging

llm_tokenizer, llm_model, system_prompt = load_llm()

def run_test(user_text, gt_command):
    user_command = run_llm(llm_tokenizer, llm_model, system_prompt, user_text)
    user_command = user_command['command']
    if user_command != gt_command:
        logging.info(f'Failed: {user_text}')
    else:
        logging.info(f'PASS: {user_text}')

if __name__ == "__main__":

    run_test(user_text="open general", gt_command="http://localhost:8080/show_overview")
    run_test(user_text="how much battery do I have?", gt_command="http://localhost:8080/show_power_screen")
    run_test(user_text="show me the map", gt_command="http://localhost:8080/show_navigation")
    run_test(user_text="how many waepons do we have ?", gt_command="http://localhost:8080/show_inventory")
