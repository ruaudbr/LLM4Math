import gradio as gr
from ctransformers import AutoModelForCausalLM
import sys

def generate_text(prompt):
    
    generated_text = llm(prompt, stream=False)
    
    return generated_text


def load_model(model_type="mistral", model_folder="../../models/"):
    
    MODEL_PATH = model_folder + model_type
    
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type=model_type,
            gpu_layers=0
        )
        return llm
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def gradio_app(model_name):

    # Load the model and its tokenizer
    global llm
    llm = load_model(model_name)

    # Create Gradio Interface
    iface = gr.Interface(
        fn=generate_text,
        inputs="text",
        outputs="text",
        live=True,
        title="Teacher Assistant",
        description="Ask your Teacher Assistant to generate educational content",
    )

    # Launch Gradio Interface
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
