import torch
from torch import autocast
from diffusers import DiffusionPipeline
import concurrent.futures
from firebase_admin import storage
from firebase_admin import credentials
import firebase_admin
import uuid
from io import BytesIO

cred = credentials.Certificate('/ai-studio-credential.json')
app = firebase_admin.initialize_app(cred, {
  "apiKey": "AIzaSyDHzXueVeh_I4m5kemyCVGmYtu_uDfQ1Mw",
  "authDomain": "ai-art-app-32bd3.firebaseapp.com",
  "projectId": "ai-art-app-32bd3",
  "storageBucket": "ai-art-app-32bd3.appspot.com",
  "messagingSenderId": "968016255589",
  "appId": "1:968016255589:web:443264566a633de98f448c",
  "measurementId": "G-Z1JTTFC6WV",
})

bucket = storage.bucket(app=app)

def upload_image(image, prompt, modifiers):

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_id = str(uuid.uuid4())
    ref = "aistudio/"+image_id+".png"
    buffered.seek(0)
    blob = bucket.blob(ref)
    blob.metadata={
        "prompt":prompt,
        "modifiers": modifiers
    }
    blob.upload_from_file(buffered, content_type='image/png')
    blob.make_public()
    return blob.public_url


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    repo_id = "./model/dreamlike-photoreal-2.0"

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