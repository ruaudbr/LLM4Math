import gradio as gr
from ctransformers import AutoModelForCausalLM
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

def generate_text(prompt):
    global llm
    global ready
    if llm is None:
        return "Error, no model selected"
    if not ready:
        return "the llm is not ready yet"
    print("Start generating text...")
    generated_text = llm(prompt, stream=False)
    print("Done generating text :)")
    
    return generated_text

def load_model(model_name, gpu_layer):
    MODEL_PATH = available_models_paths[model_name]
    global llm
    global ready
    
    if "llama" in MODEL_PATH:
        model_type = "llama2"
    else:
        model_type = "mistral"
    print(f"Loading model from {MODEL_PATH}, type {model_type}")
    ready = False
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type=model_type,
            gpu_layers=gpu_layer
        )
        print("done loading model")
        ready = True
        return llm
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def gradio_app(models_path):

    # Load the model and its tokenizer
    global available_models_paths
    available_models_paths = load_available_models_paths(models_path)
    
    # Initialize the llm. Will be chosen by the user
    global llm, ready
    llm, ready = None, None
    #llm = load_model(model_name, model_type, gpu_layers)

    # Create a Gradio Chatbat Interface
    with gr.Blocks() as iface:
        with gr.Tab("Teacher Assistant"):
            gr.Interface(
                fn=generate_text,
                inputs="text",
                outputs="text",
                live=False,
                title="Teacher Assistant",
                description="Ask your Teacher Assistant to generate educational content",
            )
        with gr.Tab("option"):
            model_Dd = gr.Dropdown(available_models_paths.keys())
            sl1 = gr.Slider(0, 5000, step=1, info="nombre de layer sur le GPU")
            b1 = gr.Button("Load model")

            b1.click(load_model, inputs=[model_Dd, sl1], outputs=llm)
    
    # Launch Gradio Interface
    iface.launch()


# python src/app/app_GGUF.py "llama2" "llama-2-7b-chat.Q4_K_M.gguf"
if __name__ == "__main__":
    #if len(sys.argv) < 3:
    if len(sys.argv) >= 3:
        path = sys.argv[2]
    else:
        path = "/home/pie2023/dataSSD/models"
    # Launch the Gradio interface
    gradio_app(path)
