from nlp.LLM_Handler import LLM_Handler
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

llm_handler = LLM_Handler()

def run_test(user_text, gt_command):
    user_command = llm_handler.run_llm(user_text)
    user_command = user_command['command']
    if user_command != gt_command:
        logging.info(f'Failed: {user_text}')
    else:
        logging.info(f'PASS: {user_text}')

if __name__ == "__main__":

    run_test(user_text="open general"                 , gt_command="show_overview")
    run_test(user_text="how much battery do I have?"  , gt_command="show_power_screen")
    run_test(user_text="show me the map"              , gt_command="show_navigation")
    run_test(user_text="how many weapons do we have ?", gt_command="show_inventory")
