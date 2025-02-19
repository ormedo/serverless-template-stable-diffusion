# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import DiffusionPipeline
import torch

import os

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    #Set auth token which is required to download stable diffusion model weights
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    repo_id = "./dreamlike-photoreal-2.0"

    model = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16,use_auth_token=HF_AUTH_TOKEN,  safety_checker = None)



if __name__ == "__main__":
    download_model()