import os
import time
import tempfile
from pathlib import Path
import torch
from safetensors.torch import load_file
from diffusers import (
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    AutoencoderTiny,
)
from huggingface_hub import hf_hub_download

def initialize_pipeline(base_model, checkpoint, repo,use_taesd=False, taesd_model=None,torch_dtype=torch.float16, device="cuda"):
    # Configure device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load UNet configuration and model
    unet_config = UNet2DConditionModel.load_config(base_model, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to(device, torch_dtype)
    unet.load_state_dict(load_file(hf_hub_download(repo, checkpoint), device=device))

    # Initialize the pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(base_model, unet=unet, torch_dtype=torch_dtype, variant="fp16", safety_checker=False).to(device)

    # Optionally use Tiny Autoencoder
    if use_taesd and taesd_model:
        pipe.vae = AutoencoderTiny.from_pretrained(taesd_model, torch_dtype=torch_dtype, use_safetensors=True).to(device)

    # Set custom scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    return pipe



# Example Usage
if __name__ == "__main__":
    # Parameters for initialization
    BASE_MODEL = "CompVis/stable-diffusion-v1-4"
    CHECKPOINT = "path/to/checkpoint.pth"
    REPO = "your-huggingface-repo"
    TAESD_MODEL = "path/to/taesd-model"

    USE_TAESD = os.environ.get("USE_TAESD", "0") == "1"

    # Initialize the pipeline
    pipeline = initialize_pipeline(
        base_model=BASE_MODEL,
        checkpoint=CHECKPOINT,
        repo=REPO,
        use_taesd=USE_TAESD,
        taesd_model=TAESD_MODEL,
        torch_dtype=torch.float16,
        device="cuda"
    )

    # Generate an image
    prompt = "A futuristic cityscape at sunset, vibrant colors, highly detailed"
    generated_image_path = predict_image(
        pipe=pipeline,
        prompt=prompt,
        seed=42,
        num_inference_steps=100,
        guidance_scale=8.5,
        width=768,
        height=768
    )

    print(f"Generated image saved at: {generated_image_path}")