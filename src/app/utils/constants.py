import os

MODELS_ID = {
    "Yi34b_chat": "01-ai/Yi-34b-Chat",
    ### Mistral-based ###
    "mixtral7b_instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral7b_instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral7b_orca": "Open-Orca/Mistral-7B-OpenOrca",
    "zephyr7b": "HuggingFaceH4/zephyr-7b-beta",
    "vigostral7b": "bofenghuang/vigostral-7b-chat",
    "mistral7b_original": "mistralai/Mistral-7B-v0.1",
    ### Llama-based ###
    "llama2-chat7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-chat13b": "meta-llama/Llama-2-13b-chat-hf",
    "vigogne7b": "bofenghuang/vigogne-2-7b-chat",
    "vigogne7b_instruct": "bofenghuang/vigogne-2-7b-instruct",  # ok pour les licenses
    "wizard7b_math": "WizardLM/WizardMath-7B-V1.0",
    "wizard13b_math": "WizardLM/WizardMath-13B-V1.0",
    "wizard15b_coder": "WizardLM/WizardCoder-15B-V1.0",
    "wizard34b_coder": "WizardLM/WizardCoder-Python-34B-V1.0",
    # bigscience bloom (7b)
    "bloom7b": "bigscience/bloom-7b1",
    # GPT-neo
    "gptNeo_original": "EleutherAI/gpt-neo-2.7B",
    # GPT-J
    "gptJ_original": "EleutherAI/gpt-j-6B",
}


# locally stored models
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


# locally stored models + interesting models stored on the hub
MODELS_PATH = "/home/pie2023/dataSSD/models/"
available_models_paths = load_available_models_paths(MODELS_PATH)
MODEL_NAMES = list(available_models_paths.keys()) + list(MODELS_ID.keys())
DEFAULT_MODEL = "llama2-chat7b"

PRECISIONS = ["4", "8", "16", "32"]
DEFAULT_PRECISION = "4"

DEFAULT_CACHE = "/home/pie2023/dataSSD/models"


# hf generation config
GENERATION_CONFIG = dict(
    max_new_tokens=1024,
    do_sample=True,
    top_p=0.95,
    top_k=1000,
    temperature=1.0,
    num_beams=1,
)
