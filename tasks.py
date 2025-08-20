"""
Celery任务定义
"""
from celery import Celery
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, FluxPipeline
import numpy as np
import json
import os
import fasteners  # 用于文件锁
from datetime import datetime

# 模型缓存
MODEL_CACHE = {}

# 初始化Celery app
app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task(acks_late=True, bind=True)
def generate_single_image(self, prompt_id: int, prompt_text: str, seed: int, generation_config: dict):
    """
    核心任务：生成单张图像并保存结果。
    参数:
        prompt_id: 提示词ID
        prompt_text: 提示词文本
        seed: 随机种子
        generation_config: 包含所有生成参数的字典
    """
    model_id = generation_config.get("model_name", "runwayml/stable-diffusion-v1-5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 1. 准备生成器与初始噪声
        generator = torch.Generator(device=device).manual_seed(seed)
        if "FLUX" in model_id.upper():
            latent_shape = (1, 16, 64, 64)  # FLUX
        else:
            latent_shape = (1, 4, 64, 64)  # SD v1.5
        init_latents = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float16)
        
        # 2. 加载模型管道（使用缓存机制）
        global MODEL_CACHE
        if "FLUX" in model_id.upper():
            if model_id not in MODEL_CACHE:
                MODEL_CACHE[model_id] = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = MODEL_CACHE[model_id]
        else:
            if model_id not in MODEL_CACHE:
                MODEL_CACHE[model_id] = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = MODEL_CACHE[model_id]
        
        # 确保使用正确的采样器
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        
        # 3. 动态加载LoRA
        lora_configs = generation_config.get("lora_config", [])
        for lora_settings in lora_configs:
            lora_path = f"static/lora_models/{lora_settings['lora_name']}"
            lora_scale = lora_settings['lora_scale']
            pipe.load_lora_weights(lora_path)  # Diffusers官方方法
            # 注意: 某些版本可能需要 pipe.fuse_lora() 以获得性能提升
            
        # 4. 生成图像
        with torch.no_grad():
            image = pipe(
                prompt=prompt_text,
                num_inference_steps=generation_config.get("num_inference_steps", 20),
                guidance_scale=generation_config.get("guidance_scale", 7.5),
                width=generation_config.get("width", 1024),
                height=generation_config.get("height", 1024),
                generator=generator,
                latents=init_latents  # 传入初始噪声
            ).images[0]
        
        # 5. 保存输出
        image_filename = f"static/images/{prompt_id}/{seed}.jpg"
        latent_filename = f"static/latents/{prompt_id}/{seed}.npy"
        os.makedirs(os.path.dirname(image_filename), exist_ok=True)
        os.makedirs(os.path.dirname(latent_filename), exist_ok=True)
        
        image.save(image_filename)
        np.save(latent_filename, init_latents.cpu().numpy())
        
        # 6. 计算评估分数
        clip_score = calculate_clip_score(image, prompt_text)
        hps_score = calculate_hpsv2_score(image, prompt_text)
        
        # 7. 原子性更新元数据文件（使用文件锁）
        data_file_path = f"data/{prompt_id}.json"
        lock = fasteners.InterProcessLock(data_file_path + ".lock")
        with lock:
            with open(data_file_path, 'r+') as f:
                data = json.load(f)
                data['generations'][str(seed)] = {
                    "seed": seed,
                    "image_path": image_filename,
                    "latent_path": latent_filename,
                    "clip_score": clip_score,
                    "hpsv2_score": hps_score,
                    "status": "completed",
                    "created_at": datetime.now().isoformat()
                }
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
        
        return {"status": "success", "seed": seed}
    
    except torch.cuda.OutOfMemoryError as e:
        # 显存不足错误处理
        cleanup_partial_results(image_filename, latent_filename)
        log_error(f"CUDA OOM error for prompt {prompt_id}: {str(e)}")
        raise self.retry(exc=e, countdown=300, max_retries=1)  # 显存错误延长重试间隔
        
    except (OSError, IOError) as e:
        # 文件系统错误处理
        cleanup_partial_results(image_filename, latent_filename)
        log_error(f"Filesystem error for prompt {prompt_id}: {str(e)}")
        update_failure_status(prompt_id, seed, "filesystem_error")
        raise self.retry(exc=e, countdown=120, max_retries=2)
        
    except json.JSONDecodeError as e:
        # JSON解析错误处理
        cleanup_partial_results(image_filename, latent_filename)
        log_error(f"JSON error for prompt {prompt_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3)
        
    except Exception as e:
        # 通用错误处理
        cleanup_partial_results(image_filename, latent_filename)
        log_error(f"Unexpected error for prompt {prompt_id}: {str(e)}")
        update_failure_status(prompt_id, seed, "failed")
        raise self.retry(exc=e, countdown=60, max_retries=3)  # 失败重试3次

def cleanup_partial_results(image_path, latent_path):
    """清理生成失败时的部分结果文件"""
    for path in [image_path, latent_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                log_error(f"Failed to cleanup {path}: {str(e)}")

def update_failure_status(prompt_id, seed, error_type):
    """原子性更新失败状态"""
    data_file_path = f"data/{prompt_id}.json"
    lock = fasteners.InterProcessLock(data_file_path + ".lock")
    with lock:
        try:
            with open(data_file_path, 'r+') as f:
                data = json.load(f)
                data['generations'][str(seed)] = {
                    "status": "failed",
                    "error_type": error_type,
                    "failed_at": datetime.now().isoformat()
                }
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
        except Exception as e:
            log_error(f"Failed to update failure status: {str(e)}")

def log_error(message):
    """记录错误日志"""
    with open("logs/error.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} - {message}\n")

from torchmetrics.functional.multimodal import clip_score
import hpsv2
from PIL import Image
import torch
import tempfile
import os

def calculate_clip_score(image, prompt_text):
    """
    使用torchmetrics的CLIP score计算图像和文本的相似度
    Args:
        image: PIL图像对象或图像路径
        prompt_text: 文本提示
    
    Returns:
        CLIP score (0-1)
    """
    # 确保输入是PIL图像
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise ValueError("image参数必须是PIL图像对象或图像路径")
    
    # 将图像转换为tensor
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()  # HWC -> CHW并添加batch维度
    
    # 计算CLIP score
    score = clip_score(image_tensor, prompt_text)
    
    # 转换为0-1范围 (CLIP score返回的是-1到1之间的值)
    normalized_score = (score + 1) / 2
    
    return float(normalized_score)

def calculate_hpsv2_score(image, prompt_text):
    """
    计算HPSv2分数
    Args:
        image: PIL图像对象或图像路径
        prompt_text: 文本提示
    
    Returns:
        HPSv2分数 (0-100)
    """
    # HPSv2评估需要图像路径和文本
    if isinstance(image, Image.Image):
        # 如果是PIL图像对象，保存到临时文件
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmpfile:
            image.save(tmpfile.name)
            image_path = tmpfile.name
            # HPSv2评估
            score = hpsv2.score(image_path, prompt_text)
    elif isinstance(image, str):
        image_path = image
        score = hpsv2.score(image_path, prompt_text)
    else:
        raise ValueError("image参数必须是PIL图像对象或图像路径")
    
    return float(score[0]) if isinstance(score, list) else float(score)
