import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import sys
import os


def load_available_models_paths(path):
    models_paths = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".gguf"):
                file_path = os.path.join(root, file)
                models_paths[file] = file_path
    return models_paths


def load_tokenizer_and_model(model_name):
    MODEL_PATH = available_models_paths[model_name]

    global tokenizer, llm, llm_is_loaded

    if "llama" in MODEL_PATH:
        model_type = "llama2"
    else:
        model_type = "mistral"
    print(f"Loading {model_type}-type model from : \n {MODEL_PATH}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("Model loaded successfully :)")
        llm_is_loaded = True
        return tokenizer, model

    except Exception as e:
        print(f"Error loading model : {e}")
        return None, None


def predict(message, history):
    if tokenizer is None or llm is None or not llm_is_loaded:
        return "No LLM has been loaded yet ..."

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
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
    )

    t = Thread(target=llm.generate, kwargs=generate_kwargs)
    print("Started generating text ...")
    t.start()

    partial_message = ""
    for new_token in streamer:
        if new_token != "<":
            partial_message += new_token
            yield partial_message
    print("Done generating text :)")


def gradio_app(models_path):
    # Load the model and its tokenizer
    global available_models_paths
    available_models_paths = load_available_models_paths(models_path)

    # Initialize the llm. Will be chosen by the user
    global tokenizer, llm, llm_is_loaded
    tokenizer, llm, llm_is_loaded = None, None, False

    # Create a Gradio Chatbat Interface
    with gr.Blocks() as iface:
        with gr.Tab("Teacher Assistant"):
            gr.ChatInterface(predict)

        with gr.Tab("Model choice and Options"):
            model_name_chosen = gr.Dropdown(available_models_paths.keys())
            b1 = gr.Button("Load model")
            b1.click(
                load_tokenizer_and_model,
                inputs=[model_name_chosen],
                outputs=(tokenizer, llm),
            )

    # Launch Gradio Interface
    print("Launching Gradio Interface...")
    iface.launch()


# python app.py model_name
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <model_name>")
        sys.exit(1)

    # Retrieve the model name from the command line
    model_name_arg = sys.argv[1]

    # Launch the Gradio interface
    gradio_app(model_name_arg)
