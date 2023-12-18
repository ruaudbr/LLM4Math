import sys
from tqdm import tqdm

import pandas as pd

from constants import MODELS_ID
from utils import load_model

# Parsing aguments
arg = sys.argv
if len(arg) != 4:
    raise Exception(
        f"Invalid syntax calling this script. Please use the following :\n python {arg[0]} model_name precision prompt_file_name"
    )
model_name, precision = arg[1], arg[2]

###### Model name #####
if model_name not in MODELS_ID:
    model_names = "\n".join(MODELS_ID.keys())
    raise Exception(f"Invalide model name :(\nValid model names are : \n{model_names}")
model_id = MODELS_ID[model_name]

###### Precision ######
if precision not in ["4", "8", "16", "32"]:
    raise Exception("Invalide model name :(\nValid precisions are : 4, 8, 16 or 32")

###### Loading the model ######
print(f"Loading model : {model_id}")
tokenizer, model = load_model(model_id, precision)


###### Fetching the prompts ######
PROMPTS_FOLDER = "./src/generation/prompts/"
file_name = arg[3]
try:
    prompts_df = pd.read_csv(PROMPTS_FOLDER + file_name)
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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        # temperature=1.1,
        do_sample=True,
        top_k=5,
        top_p=20,
        num_return_sequences=1,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_length=2048,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answers.append(answer)


###### Saving prompts:answers into a .csv ######
print(f"Saving the answers")
assert len(prompts) == len(answers), "The number of prompts and answers is not the same"
prompts_df["answer"] = answers
ANSWERS_FOLDER = "./src/generation/generated_answers/"
answers_file_name = f"answers_{model_name}_{precision}_{file_name}"
prompts_df.to_csv(ANSWERS_FOLDER + answers_file_name, index=False)
