# Hugging-Face model ids
models_id = {
    ### Mistral-based ###
    "mistral7b_instruct" : "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral7b_orca" : "Open-Orca/Mistral-7B-OpenOrca",
    "zephyr7b" : "HuggingFaceH4/zephyr-7b-beta",
    "vigostral7b" : "bofenghuang/vigostral-7b-chat",

#    "mistral7b_original" : "mistralai/Mistral-7B-v0.1",

    ### Llama-based ###
    "llama2-chat7b" : "meta-llama/Llama-2-7b-chat-hf",
    "llama2-chat13b" : "meta-llama/Llama-2-13b-chat-hf",

    "vigogne7b" : "bofenghuang/vigogne-2-7b-chat",
    "vigogne7b_instruct" : "bofenghuang/vigogne-2-7b-instruct", #ok pour les licenses

    "wizard7b_math" : "WizardLM/WizardMath-7B-V1.0",
    "wizard13b_math" : "WizardLM/WizardMath-13B-V1.0",

    "wizard15b_coder" : "WizardLM/WizardCoder-15B-V1.0",
    "wizard34b_coder" : "WizardLM/WizardCoder-Python-34B-V1.0",

    ###bigscience bloom (7b)
    "bloom7b" : "bigscience/bloom-7b1",
    ## GPT-neo
    #"gptNeo_original" : "EleutherAI/gpt-neo-2.7B",
    ## GPT-J
    #"gptJ_original" : "EleutherAI/gpt-j-6B",

}

###### Choose your model with its name ######
model_name = "wizard7b_math" #@param ["mistral7b_instruct", "mistral7b_orca", "zephyr7b", "vigostral7b","bloom7b", "vigogne7b_instruct", "llama2-chat7b", "llama2-chat13b", "vigogne7b","wizard7b_math", "wizard13b_math", "wizard15b_coder", "wizard34b_coder"]

###### Choose your quantization config ######
quant_config = "32bits" # @param ["4bits", "8bits", "16bits", "32bits"]

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = models_id[model_name]

# quantization to int4 (don't want to mess with "device" here, to be studied)
#4bit, 4 bits = 1/2 byte --> #paramsInB * 1/2 = RAM needed to load full model
if quant_config == "4bits":
    print("Loading model in 4bits")
    #quant config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        #bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        load_in_8bit=False
    )
    #Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                #device_map="auto",
            )
    tokenizer = AutoTokenizer.from_pretrained(model_id)


# quantization to int8  (don't want to mess with "device" here, to be studied)
#8bit, 8 bits = 1 byte --> #paramsInB * 1 = RAM needed to load full model
elif  quant_config == "8bits":
    print("Loading model in 8bits")
    #load quantized model
    model = AutoModelForCausalLM.from_pretrained(
                model_id,
                #device_map="auto",
                load_in_8bit=True, # 8bits here
          )
    tokenizer = AutoTokenizer.from_pretrained(model_id)



#half-precision, 16 bits = 2 bytes --> #paramsInB * 2 = RAM needed to load full model
elif  quant_config == "16bits":
    print("Loading model in half-precision")
    #device-agnostic code
    device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )
    #load model
    model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16, #half-precision here
                device_map="auto",
            )#.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)#.to(device)



#full-precision, 32bits = 4 bytes --> #paramsInB * 4 = RAM needed to load full model
elif  quant_config == "32bits":
    print("Loading model in full-precision")
    #device-agnostic code
    device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )
    #load model
    model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32, #full-precision here
            ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)#.to(device)

# Instruction : if the model is a chat model, specify context, persona, personality, skills, ...

###
chat = True
###

instruction = "You are a passionate elementary school teacher. " + \
"You are teaching a class of 20 pupils. " + \
"You love to explain things to childrens with images they understand at their age and relevant examples."

# Prompt : your question, task, ...
prompt = "Write a math exercice around a football with a couple of multiplications."
prompt_no_chat = "Here is a small 3-examples math word problem for children aged 8 years old on basic multiplication with a football theme/story to hook them : "

text_input = instruction + prompt if chat else prompt_no_chat
inputs = tokenizer(text_input, return_tensors="pt").to(model.device)


outputs = model.generate(
    **inputs,
    #temperature=1.1, # >1 augmente la diversité/surprise sur la génération (applatie la distribution sur le next token), <1 diminue la diversité de la génération (rend la distribution + spiky)
    do_sample=False,
    top_k=5,
    top_p=10, # le token suivant est tiré du top 'top_p' de la distribution uniquement
    num_return_sequences=1,
    repetition_penalty=1.5, #pour éviter les répétitions, je suis pas au clair avec commment il marche celui-là mais important à priori
    eos_token_id=tokenizer.eos_token_id,
    max_length=1024,
    )

# display the generated answer
print(tokenizer.decode(outputs[0], skip_special_tokens=True))