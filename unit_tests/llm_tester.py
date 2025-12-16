from nlp.LLM_Handler import LLM_Handler
#from nlp.Big_LLM import LLM_Handler

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# TODO:
# 1. not deteministic
# 2. MLX - MLX is a machine learning framework developed by Apple (for apple)

#model_name      = "Qwen/Qwen2.5-0.5B"        # 2-3/10
#model_name     = "Qwen/Qwen3-0.6B"          #7-8/10
model_name     = "Qwen/Qwen3-0.6B-FP8"          #7-8/10  (1360)
llm_handler = LLM_Handler(model_name)

def run_test(user_text, gt_command):
    user_command = llm_handler.run_llm(user_text)
    #print(user_command)
    user_command = user_command['command']
    result       = 0
    if user_command != gt_command:
        logging.info(f'Failed: {user_text}, Model: {user_command}')
    else:
        logging.info(f'PASS: {user_text}')
        result = 1
    return result

if __name__ == "__main__":

    total = 0
    total = total + run_test(user_text="open general"                 , gt_command="show_overview")
    total = total + run_test(user_text="how much battery do I have?"  , gt_command="show_power_screen")
    total = total + run_test(user_text="show me the map"              , gt_command="show_navigation")
    total = total + run_test(user_text="how many weapons do we have ?", gt_command="show_inventory")

    total = total + run_test(user_text="Open power screen"       , gt_command="show_power_screen")
    total = total + run_test(user_text="Go back to home screen"  , gt_command="show_overview")
    total = total + run_test(user_text="What is my power source" , gt_command="show_power_screen")


    total = total + run_test(user_text="What are launcher's angles?", gt_command="show_navigation")
    total = total + run_test(user_text="show Crate status"          , gt_command="show_navigation")
    total = total + run_test(user_text="Interceptors status"        , gt_command="show_inventory")

    print(f"Model: {model_name}, Total pass: {total}")
