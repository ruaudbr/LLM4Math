import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_model(model_id, precision):
    # quantization to int4 (don't want to mess with "device" here, to be studied)
    # 4bit, 4 bits = 1/2 byte --> #paramsInB * 1/2 = RAM needed to load full model
    if precision == "4":
        print("Loading model in 4 bits")
        # quant config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            load_in_8bit=False,
        )
        # Load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            # device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)


    # quantization to int8  (don't want to mess with "device" here, to be studied)
    # 8bit, 8 bits = 1 byte --> #paramsInB * 1 = RAM needed to load full model
    elif precision == "8":
        print("Loading model in 8 bits")
        # load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # device_map="auto",
            load_in_8bit=True,  # 8bits here
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)


    # half-precision, 16 bits = 2 bytes --> #paramsInB * 2 = RAM needed to load full model
    elif precision == "16":
        print("Loading model in half-precision")
        # device-agnostic code
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # half-precision here
            device_map="auto",
        )  # .to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)  # .to(device)


    # full-precision, 32bits = 4 bytes --> #paramsInB * 4 = RAM needed to load full model
    elif precision == "32":
        print("Loading model in full-precision")
        # device-agnostic code
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # full-precision here
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)  # .to(device)
        
    return tokenizer, model