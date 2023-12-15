# Hugging-Face models ids 
from constants import MODELS_ID
from utils import load_model


###### Choose your model with its name ######
model_not_chosen = True
print(f"The available models are : \n {MODELS_ID.keys()}")
while model_not_chosen:
    model_name = input("Please choose a model name from the list above") 
    if model_name in MODELS_ID:
        model_not_chosen = False
    else:
        print("Unkown model :(. Please choose a model name from the list above.")
model_id = MODELS_ID[model_name]


###### Choose your quantization config ######
precision_not_chosen = True
while precision_not_chosen:
    precision = input("Select a precision : 4, 8, 16 or 32 ")
    
    if precision in ["4", "8", "16", "32"]:
        precision_not_chosen = False
    else:
        print("Please choose a valid precision : 4, 8, 16 or 32")




print(f"Loading model : {model_id}")
tokenizer, model = load_model(model_id, precision)


# Instruction : if the model is a chat model, specify context, persona, personality, skills, ...
print("model ready")
still_generating = True
while still_generating:
    
    prompt = input("input the prompt (!help) : \n")
    if prompt == "!help":
        print("!exit to close the model")
    elif prompt == "!exit":
        still_generating = False
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            # temperature=1.1, # >1 augmente la diversité/surprise sur la génération (applatie la distribution sur le next token), <1 diminue la diversité de la génération (rend la distribution + spiky)
            do_sample=False,
            top_k=5,
            top_p=10,  # le token suivant est tiré du top 'top_p' de la distribution uniquement
            num_return_sequences=1,
            repetition_penalty=1.5,  # pour éviter les répétitions, je suis pas au clair avec commment il marche celui-là mais important à priori
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


###
# chat = True
####
#
# instruction = "You are a passionate elementary school teacher. " + \
# "You are teaching a class of 20 pupils. " + \
# "You love to explain things to childrens with images they understand at their age and relevant examples."
#
## Prompt : your question, task, ...
# prompt = "Write a math exercice around a football with a couple of multiplications."
# prompt_no_chat = "Here is a small 3-examples math word problem for children aged 8 years old on basic multiplication with a football theme/story to hook them : "
#
# text_input = instruction + prompt if chat else prompt_no_chat
# inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
#
#
# outputs = model.generate(
#    **inputs,
#    #temperature=1.1, # >1 augmente la diversité/surprise sur la génération (applatie la distribution sur le next token), <1 diminue la diversité de la génération (rend la distribution + spiky)
#    do_sample=False,
#    top_k=5,
#    top_p=10, # le token suivant est tiré du top 'top_p' de la distribution uniquement
#    num_return_sequences=1,
#    repetition_penalty=1.5, #pour éviter les répétitions, je suis pas au clair avec commment il marche celui-là mais important à priori
#    eos_token_id=tokenizer.eos_token_id,
#    max_length=1024,
#    )
#
## display the generated answer
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
