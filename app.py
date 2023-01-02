import torch
from torch import autocast
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
import base64
from io import BytesIO
import os

#Disable NSFW
def dummy(images, **kwargs):
    return images, False
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    
    repo_id = "prompthero/openjourney"

    model = DiffusionPipeline.from_pretrained(repo_id).to("cuda")

    model.safety_checker = dummy
    
# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed",None)
    num_outputs = model_inputs.get("num_outputs", 1)
    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    with autocast("cuda"):
        images = model([prompt]*num_outputs,height=height,width=width,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,generator=generator).images
    
    images_base64 = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        images_base64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    # Return the results as a dictionary
    return {'image_base64': images_base64}