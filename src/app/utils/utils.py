import logging

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

from ctransformers import AutoModelForCausalLM as c_AutoModelForCausalLM


from utils.constants import (
    MODELS_ID,
    DEFAULT_MODEL,
    DEFAULT_PRECISION,
    DEFAULT_HF_CACHE,
    DEFAULT_GGUF_CACHE,
    GENERATION_CONFIG,
    ORIGINAL_MODEL,
)


# -------------------------------------------
# logger
logger = logging.getLogger("app")


# -------------------------------------------
# loading models functions
def load_model(
    model_name: str,
    precision_chosen: str,
    gpu_layer: int = 0,
):
    """
    Root function for loading models.
    Dispatches the loading to the right function based on the chosen model.

    Parameters:
        model_name: name of the model to load.
        precision_chosen: precision to load the model in (quantization). Default value or chosen by the user.
        gpu_layer: number of layers to off-load on GPU. Default value or chosen by the user.

    """
    if "gguf" in model_name:
        load_gguf_model(model_name, gpu_layer)
    else:
        load_hf_model(model_name, precision_chosen)


def load_gguf_model(
    model_name: str,
    gpu_layer: int = DEFAULT_MODEL,
    cache_dir: str = DEFAULT_GGUF_CACHE,
):
    """
    Load a GGUF model through the ctransformers library.

    Parameters:
        model_name: name of the model file to load.
        gpu_layer: number of layers to off-load on GPU. Default value or chosen by the user.
        cache_dir: path to the cache directory. Default value.
    """

    global model, tokenizer
    model_path = cache_dir + model_name
    model_type = "llama" if "llama" in model_path else "mistral"

    logger.info(f"Loading {model_type}-type model from : \n {model_path}")

    try:
        model = c_AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type=model_type,
            model_file=model_name,
            gpu_layers=gpu_layer,
        )
        if (model_name in ORIGINAL_MODEL):
            model_id = MODELS_ID[ORIGINAL_MODEL[model_name]]
        else:
            logger.info("Impossible de retrouvé le model original, informe qqu de l'equipe de programation!")
            model_id = "meta-llama/Llama-2-7b-chat-hf" if "llama" in model_path else "mistralai/Mistral-7B-Instruct-v0.1"
            logger.info(f"utilisation de {model_id} par defaut pour le tokeniser")
    
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )

        logger.info("Model loaded successfully :)")

    except Exception as e:
        logger.info(f"Error loading model through ctransformers: \n{e} ")
        model, tokenizer = None, None


def load_hf_model(
    model_name: str,
    precision: str = DEFAULT_PRECISION,
    cache_dir: str = DEFAULT_HF_CACHE,
):
    """
    Loads a model and its tokenizer from the hugging-face hub through the transformers library.

    Parameters:
        model_name: name of the model to load.
        precision: precision to load the model in (quantization). Default value or chosen by the user.
        cache_dir: path to the cache directory. Default value.
    """

    global model, tokenizer
    model_id = MODELS_ID[model_name]

    logger.info(f"Loading {model_name} at {model_id}")
    if precision == "4":
        logger.info("Loading model in 4 bits")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=nf4_config,
            device_map="auto",  # accelerate dispatches layers to ram, vram or disk
            cache_dir=cache_dir,
        )
    elif precision == "8":
        logger.info("Loading model in 8 bits")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        int8_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=int8_config,
            device_map="auto",
            cache_dir=cache_dir,
        )
    elif precision == "16":
        logger.info("Loading model in 16 bits")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # f16_config = BitsAndBytesConfig(
        #    load_in_16bit=True,
        #    bnb_16bit_compute_dtype=torch.bfloat16,
        # )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # quantization_config=f16_config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
    else:
        logger.info("Loading model in 32 bits")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            cache_dir=cache_dir,
        )

    logger.info("Model loaded successfully :)")


# -------------------------------------------
# generation functions
def generate_answer(
    model_name: str,
    message: str,
    history: list[list[str, str]],
):
    """
    Root function for generating answers.
    Dispatches the generation to the right function based on the chosen model.

    Parameters:
        model_name_chosen: name of the model to use to generate the response. Default value or chosen by the user.
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
    """

    global model, tokenizer

    if model is None:
        yield "No LLM has been loaded yet :( Please load a model first."
    elif "gguf" in model_name:
        yield from generate_gguf(message, history, model)
    else:
        yield from generate_hf(message, history, model, tokenizer)


def generate_gguf(
    message: str,
    history: list[list[str, str]],
    model: c_AutoModelForCausalLM,
):
    """
    Generates an answer using GGUF's ctransformers.
    Responds to the user's message based on the dialogue history.

    Parameters:
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        model: model to use to generate the response.
    """

    if model is None:
        return "No LLM has been loaded yet ..."

    else:
        messages = []
        for item in history:
            messages += [
                {"role": "user", "content": item[0]},
                {"role": "assistant", "content": item[1]},
            ]
        messages.append({"role": "user", "content": message})
        messages = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        print(messages)


        logger.info("Started generating text ...")
        partial_message = ""
        for new_token in model(messages, stream=True):
            # looking for the end of the answer and the beginning of the next one
            if "human>:" in partial_message or "bot>:" in partial_message:
                break
            else:
                partial_message += new_token
                yield partial_message
        logger.info("Answer generated :)")


def generate_hf(
    message,
    history,
    model,
    tokenizer,
):
    """
    Generates an answer using Hugging-Face's transformers.
    Responds to the user's message based on the dialogue history.

    Parameters:
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        model: model to use to generate the response.
        tokenizer: tokenizer to use to generate the response.
    """
    messages = []
    for item in history:
        messages += [
            {"role": "user", "content": item[0]},
            {"role": "assistant", "content": item[1]},
        ]
    messages.append({"role": "user", "content": message})
    print(messages)
    input_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    input_tokens = tokenizer(input_tokens, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=None,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generate_kwargs = dict(
        input_tokens,
        streamer=streamer,
        **GENERATION_CONFIG,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)

    logger.info("Started generating text ...")
    t.start()
    partial_message = ""
    for new_token in streamer:
        if new_token != "<":
            partial_message += new_token
            yield partial_message
    logger.info("Answer generated :)")
