from nlp.LLM_Handler import LLM_Handler
import torch


if __name__ == "__main__":

    print(f"CUDA: {torch.cuda.is_available()}")
    llm_model = LLM_Handler()

    print(llm_model.run_llm("Where am I located?"))
    print(llm_model.run_llm("Thank you"))
    print(llm_model.run_llm("I'll show you"))
    print(llm_model.run_llm("Good moorning"))
    print(llm_model.run_llm("nothing"))
