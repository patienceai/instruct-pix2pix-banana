# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
import os

def download_model():
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16)

if __name__ == "__main__":
    download_model()
