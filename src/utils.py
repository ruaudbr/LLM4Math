import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from constants import DEFAULT_CACHE


def load_model(model_id, precision, cache_dir=DEFAULT_CACHE):
    # quantization to int4 (don't want to mess with "device" here, to be studied)
    # 4bit, 4 bits = 1/2 byte --> #paramsInB * 1/2 = RAM needed to load full model
    if precision == "4":
        print("Loading model in 4 bits")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            # use_flash_attn=True,
            # use_flash_attention_2=True,
            # attn_implementation="flash_attention_2",
            device_map="auto",  # accelerate dispatches layers to ram, vram or disk
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # quantization to int8  (don't want to mess with "device" here, to be studied)
    # 8bit, 8 bits = 1 byte --> #paramsInB * 1 = RAM needed to load full model
    elif precision == "8":
        print("Loading model in 8 bits")
        # load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=True,
            use_flash_attention=True,
            device_map="auto",  # accelerate dispatches layers to ram, vram or disk
            cache_dir=cache_dir,
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
            use_flash_attention=True,
            device_map="auto",  # accelerate dispatches layers to ram, vram or disk
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # full-precision, 32bits = 4 bytes --> #paramsInB * 4 = RAM needed to load full model
    elif precision == "32":
        print("Loading model in full-precision")
        # device-agnostic code
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # full-precision here
            use_flash_attention=True,
            device_map="auto",  # accelerate dispatches layers to ram, vram or disk
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    return tokenizer, model
