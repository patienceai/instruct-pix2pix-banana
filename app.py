import torch
from torch import autocast
from diffusers import StableDiffusionInstructPix2PixPipeline
import base64
from io import BytesIO
import os
import PIL
import requests

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16).to("cuda")

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', None)
    image_url = model_inputs.get('image_url', None)
    #height = model_inputs.get('height', 512)
    #width = model_inputs.get('width', 512)
    num_inference_steps = model_inputs.get('num_inference_steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    image_guidance_scale = model_inputs.get('image_guidance_scale', 1.5)
    input_seed = model_inputs.get("seed",None)
    
    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    if image_url == None:
        return {'message': "No image URL provided"}
    
    init_img = download_image(image_url)

    model.enable_xformers_memory_efficient_attention()

    # Run the model
    with autocast("cuda"):
        image = model(prompt, negative_prompt=negative_prompt, image=init_img, num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale, guidance_scale=guidance_scale, generator=generator).images[0]
    
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
