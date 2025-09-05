from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import io
import base64
import json
import os
import uuid
from typing import List, Dict, Any
import numpy as np
import logging
import csv
from dotenv import load_dotenv
from huggingface_hub import login
from utils import _clip_score_update, _get_clip_model_and_processor

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保日志消息立即刷新到文件
logging.getLogger().handlers[0].flush = lambda: logging.getLogger().handlers[0].stream.flush()

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型
model = None
# clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
clip_model, clip_processor = _get_clip_model_and_processor("openai/clip-vit-base-patch16")
logger.info("开始加载CLIP模型")

# 数据存储路径
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
DATABASE_FILE = os.path.join(DATA_DIR, "database.json")
PROMPT_FILE = os.path.join(DATA_DIR, "prompts.json")

# 初始化数据库文件
if not os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'w') as f:
        json.dump([], f)

# 初始化Prompt文件
if not os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, 'w') as f:
        json.dump([], f)

@app.get("/")
async def root():
    return {"message": "GoldenNoise Data Collection Platform"}

@app.post("/init_model")
async def init_model():
    """
    初始化模型
    """
    global model
    
    try:
        # 获取Hugging Face token和镜像配置
        hf_token = os.getenv("HF_TOKEN")
        hf_endpoint = os.getenv("HF_ENDPOINT")
        
        if hf_token:
            logger.info("使用Hugging Face token进行认证")
            login(token=hf_token)
        
        # 如果配置了镜像，则设置环境变量
        if hf_endpoint:
            logger.info(f"使用Hugging Face镜像: {hf_endpoint}")
            os.environ['HF_ENDPOINT'] = hf_endpoint
        
        logger.info("开始加载FLUX.1-schnell模型")
        # 设置缓存目录
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载FLUX.1-schnell模型
        model = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            use_auth_token=hf_token if hf_token else None,
            cache_dir=cache_dir,
            mirror=hf_endpoint
        )
        model.enable_model_cpu_offload()
        logger.info("FLUX.1-schnell模型加载完成")
        
        return {"status": "success", "message": "Models loaded successfully"}
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/generate_batch")
async def generate_batch(
    prompt: str = Form(...),
    num_images: int = Form(4),
    steps: int = Form(20),
    cfg_scale: float = Form(7.0)
):
    """
    生成一批图像
    """
    global model
    
    if model is None:
        logger.error("模型未初始化")
        return {"status": "error", "message": "Model not initialized"}
    
    try:
        logger.info(f"开始生成图像，prompt: {prompt}, 数量: {num_images}")
        # 生成图像
        images = []
        seeds = []
        
        for i in range(num_images):
            # 生成随机种子
            seed = np.random.randint(0, 1000000)
            generator = torch.Generator("cpu").manual_seed(seed)
            
            # 生成图像
            result = model(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                output_type="np"
            )
            
            # 将图像转换为numpy数组（0-1范围）
            image_np = result.images[0]
            images.append(image_np)
            seeds.append(seed)
        
        logger.info("图像生成完成，开始计算CLIP分数")
        # 计算CLIP分数
        clip_scores = calculate_clip_score(images, prompt)
        
        # 保存图像并返回base64编码
        image_data = []
        image_paths = []
        
        for i, (img, seed) in enumerate(zip(images, seeds)):
            # 将numpy数组转换为PIL图像以便保存
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            
            # 保存图像
            image_filename = f"{uuid.uuid4()}.png"
            image_path = os.path.join(IMAGES_DIR, image_filename)
            img_pil.save(image_path)
            image_paths.append(image_path)
            
            # 转换为base64
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            image_data.append({
                "image": img_str,
                "seed": seed,
                "clip_score": clip_scores[i],
                "image_path": image_path
            })
        
        logger.info(f"图像处理完成，返回 {num_images} 张图像")
        return {
            "status": "success",
            "prompt": prompt,
            "images": image_data
        }
    except Exception as e:
        logger.error(f"图像生成失败: {str(e)}")
        return {"status": "error", "message": str(e)}

def calculate_clip_score(images, prompts):
    """
    计算图像的CLIP分数
    """
    global clip_score_fn
    
    
    # 如果prompts是单个字符串，则复制为列表
    if isinstance(prompts, str):
        prompts = [prompts] * len(images)
    
    # 将numpy数组堆叠成批次
    images_np = np.stack(images)
    logger.info(f"images shape is {images_np.shape}, the max of the images is {images_np.max()}")
    
    try:
        # 计算CLIP分数
        images_int = (images_np * 255).astype("uint8")
        clip_score, _ = _clip_score_update(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts, clip_model, clip_processor)
        clip_score = clip_score.detach()
        
        # clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return [round(float(score), 4) for score in clip_score]
    except Exception as e:
        import pdb;pdb.set_trace()
        logger.error(f"Error calculating CLIP scores: {e}")
        # 返回默认分数
        return [0.5] * len(images)

@app.post("/upload_prompts")
async def upload_prompts(file: UploadFile = File(...)):
    """
    上传并处理Prompt文件
    """
    try:
        # 读取文件内容
        content = await file.read()
        filename = file.filename
        
        # 解析文件内容
        if filename.endswith('.csv'):
            # 解析CSV文件
            content_str = content.decode('utf-8')
            lines = content_str.split('\n')
            prompts = []
            
            # 跳过标题行，从第二行开始处理
            for i in range(1, len(lines)):
                line = lines[i].strip()
                if line:
                    # 简单的CSV解析，按逗号分割
                    fields = line.split(',')
                    # 假设caption在第二列（索引1）
                    if len(fields) > 1:
                        prompts.append(fields[1])
        elif filename.endswith('.json'):
            # 解析JSON文件
            content_str = content.decode('utf-8')
            data = json.loads(content_str)
            prompts = []
            
            # 根据JSON结构提取prompt
            if isinstance(data, list):
                # 如果是数组，假设每个元素都有caption字段
                prompts = [item.get('caption', item.get('prompt', '')) for item in data]
            elif 'prompts' in data:
                # 如果有prompts字段
                prompts = data['prompts']
            else:
                # 尝试其他可能的结构
                for key in data:
                    if isinstance(data[key], str):
                        prompts.append(data[key])
        else:
            return {"status": "error", "message": "不支持的文件格式"}
        
        # 保存到Prompt文件
        with open(PROMPT_FILE, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        logger.info(f"成功上传并处理 {len(prompts)} 个Prompt")
        return {"status": "success", "message": f"成功上传 {len(prompts)} 个Prompt", "count": len(prompts)}
    except Exception as e:
        logger.error(f"上传Prompt文件失败: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/submit_selection")
async def submit_selection(selection_data: Dict[str, Any]):
    """
    提交用户选择
    """
    try:
        logger.info(f"收到用户选择提交请求，prompt: {selection_data.get('prompt', 'N/A')}")
        # 读取现有数据
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # 添加新数据
        data.append(selection_data)
        
        # 保存到文件
        with open(DATABASE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info("用户选择已保存到数据库")
        return {"status": "success", "message": "Selection saved successfully"}
    except Exception as e:
        logger.error(f"保存用户选择失败: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/get_prompts")
async def get_prompts():
    """
    获取Prompt列表
    """
    try:
        # 读取Prompt文件
        if os.path.exists(PROMPT_FILE):
            with open(PROMPT_FILE, 'r') as f:
                prompts = json.load(f)
        else:
            prompts = []
        
        return {"status": "success", "prompts": prompts}
    except Exception as e:
        logger.error(f"获取Prompt列表失败: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取主机和端口
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    
    # 配置uvicorn日志
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )
