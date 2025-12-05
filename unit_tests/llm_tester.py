from nlp.LLM_Handler import LLM_Handler
#from nlp.Big_LLM import LLM_Handler

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

llm_handler = LLM_Handler()

def run_test(user_text, gt_command):
    user_command = llm_handler.run_llm(user_text)
    #print(user_command)
    user_command = user_command['command']
    if user_command != gt_command:
        logging.info(f'Failed: {user_text}, Model: {user_command}')
    else:
        logging.info(f'PASS: {user_text}')

if __name__ == "__main__":

    run_test(user_text="open general"                 , gt_command="show_overview")
    run_test(user_text="how much battery do I have?"  , gt_command="show_power_screen")
    run_test(user_text="show me the map"              , gt_command="show_navigation")
    run_test(user_text="how many weapons do we have ?", gt_command="show_inventory")

    run_test(user_text="Open power screen"       , gt_command="show_power_screen")
    run_test(user_text="Go back to home screen"  , gt_command="show_overview")
    run_test(user_text="What is my power source" , gt_command="show_power_screen")


    run_test(user_text="What are launcher's angles?", gt_command="show_navigation")
    run_test(user_text="show Crate status"          , gt_command="show_navigation")
    run_test(user_text="Interceptors status"        , gt_command="show_inventory")
