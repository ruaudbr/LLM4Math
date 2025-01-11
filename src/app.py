import argparse
import logging
import os

import gradio as gr
from utils import load_model, predict, predict_image, save_option, test_a_file, initialize_pipeline
from utils.constants import (
    IMAGE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_PRECISION,
    MODELS_ID,
    OLLAMA_MODEL,
    PRECISIONS,
    RAG_DATABASE,
    OLLAMA_default,
    available_models_paths,
)

# Set Gradio temporary directory
os.environ["GRADIO_TEMP_DIR"] = "/home/pie2023/data/gradio_temp"

# ---------------------------------------------------------------------------
# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def changedropdownvalue(mode: str):
    if mode == "hugging face":
        model_name = gr.Dropdown(choices=list(MODELS_ID.keys()), value=DEFAULT_MODEL)
        return model_name
    elif mode == "gguf":
        model_name = gr.Dropdown(choices=list(available_models_paths.keys()))

        return model_name
    else:
        model_name = gr.Dropdown(choices=OLLAMA_MODEL, value=OLLAMA_default)
        return model_name


# ---------------------------------------------------------------------------
# Gradio app
def gradio_app():
    global model, tokenizer, pipe
    model, tokenizer, pipe = None, None, None
    with gr.Blocks("soft") as iface:
        # create an option menu to choose the model to load
        

        # Create a Gradio Chatbot Interface
        with gr.Tab("Text Generation"):
            with gr.Accordion("Model choice and options"):
                with gr.Tab("model option"):
                    Mode_radio = gr.Radio(
                        choices=["hugging face", "gguf", "RAG"],
                        value="hugging face",
                        label="What type of model to use",
                    )
                    model_name_chosen = gr.Dropdown(
                        choices=MODELS_ID.keys(),
                        value=DEFAULT_MODEL,
                        label="Choose a model",
                    )
                    precision = gr.Dropdown(
                        choices=PRECISIONS,
                        value=DEFAULT_PRECISION,
                        label="Choose a quantization precision",
                        info="Hugging face-models only",
                    )
                    gpu_layer = gr.Slider(
                        minimum=0.0,
                        maximum=50.0,
                        value=10.0,
                        step=1,
                        label="Choose the #layers to off-load on GPU",
                        info="GGUF-models only",
                    )
                    RAG_database = gr.Dropdown(
                        choices=list(RAG_DATABASE.keys()),
                        label="what database to use",
                        info="RAG-models only",
                    )
                    Mode_radio.change(
                        changedropdownvalue, Mode_radio, outputs=model_name_chosen
                    )
                    b1 = gr.Button("Load model")
                    b1.click(
                        load_model,
                        inputs=[
                            model_name_chosen,
                            precision,
                            gpu_layer,
                            RAG_database,
                            Mode_radio,
                        ],
                    )
                with gr.Tab("generation option"):
                    max_length = gr.Slider(
                        minimum=0,
                        maximum=2048,
                        value=500,
                        step=1,
                        label="How many token can the model generate max (not working yet)",
                    )
                    prefix_text = gr.Text(
                        label="what does the answer should start with (can force an AI to follow a patern)",
                        placeholder="###CONTEXT###",
                    )
                    end_text = gr.Text(
                        placeholder="</s>",
                        label="what should be consider the end of the bot generation (can cause the AI to stop generating sonner)",
                    )
                    b2 = gr.Button("Save options")
                    b2.click(save_option, [max_length, prefix_text, end_text])
                
            with gr.Accordion("Teacher Assistant"):
                gr.ChatInterface(
                    predict,
                    additional_inputs=[model_name_chosen],
                )

            with gr.Accordion("Test a file"):
                gr.Interface(
                    test_a_file,
                    [
                        gr.File(
                            file_count="single",
                            file_types=[".csv"],
                            label="Fichier a tester au format csv",
                        )
                    ],
                    "file",
                )

        with gr.Tab("Image Generation"):
            with gr.Tab("SDXL"):
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=list(IMAGE_MODELS.keys()), label="Choose a model"
                    )
                    initialize_bt = gr.Button("Initialize Pipeline", scale=1)
                    def initialize_and_set_pipeline(model_name):
                        global pipe
                        model= IMAGE_MODELS[model_name]
                        pipe = initialize_pipeline(model_name)

                    initialize_bt.click(
                        fn=initialize_and_set_pipeline, inputs=[model_name], outputs=None, show_progress=False
                    )
                with gr.Row():
                    prompt = gr.Textbox(
                        placeholder="Insert your prompt here:", scale=5, container=False
                    )
                    generate_bt = gr.Button("Generate", scale=1)

                image = gr.Image(type="filepath")
                with gr.Accordion("Advanced options", open=False):
                    seed = gr.Slider(
                        randomize=True, minimum=0, maximum=12013012031030, label="Seed", step=1
                    )
                    num_inference_steps = gr.Slider(
                        minimum=1, maximum=100, value=50, step=1, label="Number of Inference Steps"
                    )
                    guidance_scale = gr.Slider(
                        minimum=0.0, maximum=20.0, value=7.5, step=0.1, label="Guidance Scale"
                    )
                    width = gr.Slider(
                        minimum=64, maximum=1024, value=512, step=64, label="Width"
                    )
                    height = gr.Slider(
                        minimum=64, maximum=1024, value=512, step=64, label="Height"
                    )
                inputs = [prompt] #seed, num_inference_steps, guidance_scale, width, height]
                outputs = [image]

                generate_bt.click(
                    fn=lambda *args: predict_image(pipe, *args), inputs=inputs, outputs=outputs, show_progress=False
                )
            with gr.Tab("Gradio"):
                with gr.Row():
                    prompt = gr.Textbox(
                        placeholder="Insert your prompt here:", scale=5, container=False
                    )
                    generate_bt = gr.Button("Generate", scale=1)

                image = gr.Image(type="filepath")
                with gr.Accordion("Advanced options", open=False):
                    seed = gr.Slider(
                        randomize=True, minimum=0, maximum=12013012031030, label="Seed", step=1
                    )
                    num_inference_steps = gr.Slider(
                        minimum=1, maximum=100, value=2, step=1, label="Number of Inference Steps"
                    )
                    guidance_scale = gr.Slider(
                        minimum=0.0, maximum=20.0, value=0, step=0.1, label="Guidance Scale"
                    )
                    width = gr.Slider(
                        minimum=64, maximum=1024, value=512, step=64, label="Width"
                    )
                    height = gr.Slider(
                        minimum=64, maximum=1024, value=512, step=64, label="Height"
                    )
                inputs = [prompt, seed, num_inference_steps, guidance_scale, width, height]
                outputs = [image]
                generate_bt.click(
                    fn=lambda *args: predict_image(pipe, *args), inputs=inputs, outputs=outputs, show_progress=False
                )

    # Launch Gradio Interface
    logger.info("Launching Gradio Interface...")
    iface.launch()


# ---------------------------------------------------------------------------
# CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web-app to interact with a LLM")
    save_option(500, "", "</s>")
    gradio_app()
