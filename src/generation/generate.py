from tqdm import tqdm
import sys

from constants import MODELS_ID
from utils import load_model


arg = sys.argv
if len(arg) < 4:
    raise Exception(
        f"Not enough arguments provided : python {arg[0]} model_name precision input_file"
    )

prompts = []
# Open the file in read mode
with open(arg[3], "r") as file:
    # Read all lines into a list
    prompts = file.readlines()


###### Model name ######
if not (arg[1] in MODELS_ID):
    raise Exception(
        f"Invalide model name :(\nValid model names are : \n {MODELS_ID.keys()}"
    )
model_name = arg[1]
model_id = MODELS_ID[model_name]

###### Precision ######

precision = arg[2]
if precision not in ["4", "8", "16", "32"]:
    raise Exception("Invalide model name :(\nValid precisions are : 4, 8, 16 or 32")



###### Loading the model ######
print(f"Loading model : {model_id}")
tokenizer, model = load_model(model_id, precision)


output_filname = "answers" + "_" + model_name + "_" + precision + ".txt"
file = open(output_filname, "w")

print("Ready to generate")
for prompt in tqdm(prompts, desc="Generation of the prompts"):
    prompt = prompt.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        #temperature=1.1,
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
