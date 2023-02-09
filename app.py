import torch
from torch import autocast
from diffusers import DiffusionPipeline
import concurrent.futures
from upload import upload_image

#Disable NSFW
def dummy(images, **kwargs):
    return images, False
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    
    repo_id = "dreamlike-art/dreamlike-photoreal-2.0"

    model = DiffusionPipeline.from_pretrained(repo_id,  safety_checker = None).to("cuda")
    
# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    modifiers = model_inputs.get('modifiers', [])
    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 512)
    negative_prompt = model_inputs.get('negative_prompt', None)
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
    
    if negative_prompt != None:
        negative_prompt = [negative_prompt] * num_outputs

    # Run the model
    with autocast("cuda"):
        images = model([prompt]*num_outputs,negative_prompt=negative_prompt,height=height,width=width,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,generator=generator).images
    
    images_url = []
    with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = [executor.submit(upload_image, image, prompt, modifiers) for image in images]
        concurrent.futures.wait(futures)
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            images_url.append(result)

    # Return the results as a dictionary
    return {'images_url': images_url}