"""
应用配置模块
"""
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 模型配置
MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
HPS_MODEL_PATH = os.getenv("HPS_MODEL_PATH", "./models/hps_v2_compressed.pt")

# Redis配置
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 生成默认配置
DEFAULT_GENERATION_CONFIG = {
    "model_name": MODEL_ID,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "sampler_name": "Euler",
    "lora_config": []  # 默认不加载LoRA
}

# 目录配置
STATIC_DIR = "static"
DATA_DIR = "data"
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
LATENTS_DIR = os.path.join(STATIC_DIR, "latents")
LORA_MODELS_DIR = os.path.join(STATIC_DIR, "lora_models")
DRAWBENCH_DIR = os.path.join(DATA_DIR, "drawbench")

# 创建必要目录
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LATENTS_DIR, exist_ok=True)
os.makedirs(LORA_MODELS_DIR, exist_ok=True)
os.makedirs(DRAWBENCH_DIR, exist_ok=True)

def get_lora_path(lora_name):
    """
    获取LoRA文件的完整路径
    """
    return os.path.join(LORA_MODELS_DIR, lora_name)

def ensure_dir(file_path):
    """
    确保文件所在目录存在
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
