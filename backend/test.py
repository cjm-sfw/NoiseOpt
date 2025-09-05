import torch
from diffusers import DiffusionPipeline
import os
from huggingface_hub import login
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
hf_endpoint = os.getenv("HF_ENDPOINT")

if hf_token:
    login(token=hf_token)

# 如果配置了镜像，则设置环境变量
if hf_endpoint:
    os.environ['HF_ENDPOINT'] = hf_endpoint
    
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
os.makedirs(cache_dir, exist_ok=True)


model = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            use_auth_token=hf_token if hf_token else None,
            cache_dir=cache_dir,
            mirror=hf_endpoint
        )