import gradio as gr
from ctransformers import AutoModelForCausalLM
import sys

def generate_text(prompt):
    
    print("Start generating text...")
    generated_text = llm(prompt, stream=False)
    print("Done generating text :)")
    
    return generated_text


def load_model(
    model_name, 
    model_type="mistral", 
    gpu_layers=0, 
    model_folder="./models/"
):
    
    MODEL_PATH = model_folder + model_type + "/" + model_name
    
    
    print(f"Loading model from {MODEL_PATH}")
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type=model_type,
            gpu_layers=gpu_layers
        )
        print("Model loaded successfully :)")
        return llm
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def gradio_app(model_name, model_type, gpu_layers):

    # Load the model and its tokenizer
    global llm
    llm = load_model(model_name, model_type, gpu_layers)

    # Create a Gradio Chatbat Interface
    iface = gr.Interface(
        fn=generate_text,
        inputs="text",
        outputs="text",
        live=False,
        title="Teacher Assistant",
        description="Ask your Teacher Assistant to generate educational content",
    )
    
    # Launch Gradio Interface
    iface.launch()


# python src/app/app_GGUF.py "llama2" "llama-2-7b-chat.Q4_K_M.gguf"
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python app.py <model_name>")
        sys.exit(1)
    
    # Retrieve the model name from the command line
    model_type = sys.argv[1]
    model_name = sys.argv[2]
    gpu_layers = 0 if len(sys.argv) == 3 else int(sys.argv[3])
    # Launch the Gradio interface
    gradio_app(model_name, model_type, gpu_layers)
