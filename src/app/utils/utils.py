import os
import logging

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from ctransformers import AutoModelForCausalLM as c_AutoModelForCausalLM

from threading import Thread

from utils.constants import (
    MODELS_ID,
    DEFAULT_MODEL,
    DEFAULT_PRECISION,
    DEFAULT_CACHE,
    GENERATION_CONFIG,
)


# -------------------------------------------
# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# -------------------------------------------
# loading models functions
def load_available_models_paths(path: str):
    """
    Load the locally available models' paths

    Parameters:
        path: path to the directory containing all models already available locally.
    """
    models_paths = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            models_paths[file] = file_path
    return models_paths


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
    global model, tokenizer

    if "gguf" in model_name:
        load_gguf_model(model_name, gpu_layer)
    else:
        load_hf_model(model_name, precision_chosen)


def load_gguf_model(
    model_path: str,
    gpu_layer: int = DEFAULT_MODEL,
):
    """
    Load a GGUF model through the ctransformers library.

    Parameters:
        model_path: path to the model to load.
        gpu_layer: number of layers to off-load on GPU. Default value or chosen by the user.
    """

    global model, tokenizer

    if "llama" in model_path:
        model_type = "llama2"
    else:
        model_type = "mistral"

    logger.info(f"Loading {model_type}-type model from : \n {model_path}")

    try:
        llm = c_AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type=model_type,
            gpu_layers=gpu_layer,
        )
        logger.info("Model loaded successfully :)")

    except Exception as e:
        logger.info(f"Error loading model through ctransformers: \n{e} ")
        model, tokenizer = None, None


def load_hf_model(
    model_name: str,
    precision: str = DEFAULT_PRECISION,
    cache_dir: str = DEFAULT_CACHE,
):
    """
    Loads a model and its tokenizer from the hugging-face hub through the transformers library.

    Parameters:
        model_name: name of the model to load.
        precision: precision to load the model in (quantization). Default value or chosen by the user.
        cache_dir: path to the cache directory. Default value.
    """

    global tokenizer, model
    model_id = MODELS_ID[model_name]

    logger.info(f"Loading {model_name} at {model_id}")
    if precision == "4":
        logger.info("Loading model in 4 bits")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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
        f16_config = BitsAndBytesConfig(
            load_in_16bit=True,
            bnb_16bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=f16_config,
            device_map="auto",
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
        dialogue_history_to_format = history + [[message, ""]]
        messages = "".join(
            [
                "".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])
                for item in dialogue_history_to_format
            ]
        )

        logger.info("Started generating text ...")
        partial_message = ""
        for new_token in model(messages, stream=True):
            if new_token != "<":
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

    dialogue_history_to_format = history + [[message, ""]]
    messages = "".join(
        [
            "".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])
            for item in dialogue_history_to_format
        ]
    )
    input_tokens = tokenizer(messages, return_tensors="pt").input_ids.cuda()
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
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
