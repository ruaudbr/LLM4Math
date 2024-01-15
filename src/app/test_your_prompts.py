import sys
from tqdm import tqdm

import pandas as pd
import os

# caution: path[0] is reserved for script path (or '' in REPL)
from utils.constants import MODELS_ID, IS_CHAT
from utils.utils import *

def test_a_file(file_name, model_name, precision):

    ###### Precision ######
    if precision not in ["4", "8", "16", "32"]:
        raise Exception("Invalide model name :(\nValid precisions are : 4, 8, 16 or 32")

    ###### Loading the model ######
    print(f"Loading model : {model_name}")

    global model, tokenizer
    load_hf_model(model_name, precision)


    ###### Fetching the prompts ######
    PROMPTS_FOLDER = "./src/generation/prompts/"
    try:
        prompts_df = pd.read_csv(file_name)
    except FileNotFoundError:
        raise Exception(
            f"The prompts file {file_name} was not found in the folder {PROMPTS_FOLDER} :("
        )
    prompt_col_name = prompts_df.columns[0]
    prompts = prompts_df[prompt_col_name].to_list()

    ###### Generating ######
    answers = []
    print(f"Generating answers")
    for prompt in tqdm(prompts, desc="Prompt"):
        prompt = prompt.strip()

        the_answer = list(predict(prompt, [], model_name, no_log= True))[-1]
        #if IS_CHAT[model_name]:
        #    messages = [{"role": "user", "content": prompt}]
        #    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
        #        model.device
        #    )
        #else:
        #    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        #    inputs = inputs["input_ids"], inputs["attention_mask"]
#
        #outputs = model.generate(
        #    inputs,
        #    # temperature=1.1,
        #    do_sample=True,
        #    top_k=5,
        #    top_p=20,
        #    num_return_sequences=1,
        #    repetition_penalty=1.5,
        #    eos_token_id=tokenizer.eos_token_id,
        #    pad_token_id=tokenizer.eos_token_id,
        #    max_length=2048,
        #)
        #answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answers.append(the_answer)


    ###### Saving prompts:answers into a .csv ######
    print(f"Saving the answers")
    assert len(prompts) == len(answers), "The number of prompts and answers is not the same"
    prompts_df["answer"] = answers
    ANSWERS_FOLDER = os.path.dirname(file_name)
    answers_file_name = f"answers_{model_name}_{precision}_{os.path.basename(file_name)}"
    prompts_df.to_csv(ANSWERS_FOLDER + "/" + answers_file_name, index=False)
    return ANSWERS_FOLDER + "/" + answers_file_name
