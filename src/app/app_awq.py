import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys



#TODO:  add generation_config option and streamer option
def generate_text(prompt):
    
    # Convert prompt to input tokens
    input_tokens = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids.cuda()
    
    # generate output tokens
    outputs = model.generate(input_tokens)

    # convert output tokens to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def load_model(model_name, model_folder="../../models/"):
    
    MODEL_PATH = model_folder + model_name 
    
    try:
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            low_cpu_mem_usage=True,
            device_map="cuda:0"
        )
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def gradio_app(model_name):

    # Load the model and its tokenizer
    global model, tokenizer
    model, tokenizer = load_model(model_name)

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
