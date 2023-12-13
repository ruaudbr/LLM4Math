from tqdm import tqdm
import sys

from constants import MODELS_ID, PROMPTS
from utils import load_model

# Parsing aguments
arg = sys.argv
if len(arg) != 4:
    raise Exception(
        f"Invalid syntax calling this script. Please use the following :\n python {arg[0]} model_name precision input_file"
    )
model_name, precision, input_filename = arg[1], arg[2], arg[3]

###### Model name ######
if model_name not in MODELS_ID:
    raise Exception(
        f"Invalide model name :(\nValid model names are : \n {MODELS_ID.keys()}"
    )
model_id = MODELS_ID[model_name]

###### Precision ######
if precision not in ["4", "8", "16", "32"]:
    raise Exception("Invalide model name :(\nValid precisions are : 4, 8, 16 or 32")


###### Loading the model ######
print(f"Loading model : {model_id}")
tokenizer, model = load_model(model_id, precision)


###### Generating ######
output_filname = "answers" + "_" + model_name + "_" + precision + ".txt"
file = open(output_filname, "w")

for prompt_idx, prompt in tqdm(PROMPTS.items(), desc="Prompt"):
    prompt = prompt.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        # temperature=1.1,
        do_sample=False,
        top_k=5,
        top_p=10,
        num_return_sequences=1,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    file.write(answer)
