import gradio as gr
from ctransformers import AutoModelForCausalLM
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def load_available_models_paths(path):
    models_paths = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".gguf"):
                file_path = os.path.join(root, file)
                models_paths[file] = file_path
    return models_paths


def load_model(model_name, gpu_layer):
    MODEL_PATH = available_models_paths[model_name]

    global llm, llm_is_loaded

    if "llama" in MODEL_PATH:
        model_type = "llama2"
    else:
        model_type = "mistral"

    logger.info(f"Loading {model_type}-type model from : \n {MODEL_PATH}")

    try:
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type=model_type,
            gpu_layers=gpu_layer,
        )
        logger.info("Model loaded successfully :)")
        llm_is_loaded = True
        return llm

    except Exception as e:
        logger.info(f"Error loading model : {e}")
        return None


def predict(message, history):
    if llm is None or not llm_is_loaded:
        return "No LLM has been loaded yet ..."

    dialogue_history_to_format = history + [[message, ""]]
    messages = "".join(
        [
            "".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])
            for item in dialogue_history_to_format
        ]
    )
    logger.info("Started generating text ...")
    # print("Started generating text ...")
    partial_message = ""
    for new_token in llm(messages, stream=True):
        if new_token != "<":
            partial_message += new_token
            yield partial_message
    logger.info("Done generating text :)")


def gradio_app(models_path):
    # Load the model and its tokenizer
    global available_models_paths
    available_models_paths = load_available_models_paths(models_path)

    # Initialize the llm. Will be chosen by the user
    global llm_is_loaded, llm
    llm, llm_is_loaded = None, False

    # Create a Gradio Chatbat Interface
    with gr.Blocks() as iface:
        with gr.Tab("Teacher Assistant"):
            gr.ChatInterface(predict)

        with gr.Tab("Model choice and Options"):
            model_name_chosen = gr.Dropdown(available_models_paths.keys())
            gpu_layers_chosen = gr.Slider(
                0,
                5000,
                step=1,
                info="#layers to off-load on GPU",
            )
            b1 = gr.Button("Load model")
            b1.click(
                load_model,
                inputs=[model_name_chosen, gpu_layers_chosen],
                outputs=llm,
            )

    # Launch Gradio Interface
    logger.info("Launching Gradio Interface...")
    iface.launch()


### How to use this script : ###
# python src/app/app_GGUF.py
# python src/app/app_GGUF.py "llama2" "llama-2-7b-chat.Q4_K_M.gguf"
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        path = sys.argv[2]
    else:
        path = "/home/pie2023/dataSSD/gguf_models"

    gradio_app(path)
