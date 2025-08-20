"""
Golden Noise 评估系统 - 增强日志版
"""
import os
import json
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

# --- 数据模型定义 ---
class SystemSettings(BaseModel):
    model_name: str = "runwayml/stable-diffusion-v1-5"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5

class PromptItem(BaseModel):
    id: str
    text: str
    created_at: str
    num_judgments: int = 0

class ImagePair(BaseModel):
    id: str
    prompt_id: str
    image_a: str
    image_b: str 
    model_name: str
    seed: int

class Judgment(BaseModel):
    prompt_id: str
    winner_seed: int
    loser_seed: int
    user_id: str
    judged_at: str
    notes: Optional[str] = None

class LoRAModel(BaseModel):
    id: str
    name: str
    path: str
    scale: float = 1.0
    enabled: bool = True

# --- 应用初始化 ---
app = FastAPI(
    title="Golden Noise 评估系统",
    version="2.1",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 辅助函数 ---
def load_initial_data():
    """加载初始数据"""
    try:
        global settings, prompts_db, judgments_db, lora_db
        
        logger.info("开始加载系统初始数据")
        
        # 加载设置
        if os.path.exists("data/settings.json"):
            with open("data/settings.json", "r") as f:
                settings = SystemSettings(**json.load(f))
            logger.info(f"加载系统设置: {settings}")
        
        # 加载提示词
        if os.path.exists("data/prompts.json"):
            with open("data/prompts.json", "r") as f:
                prompts_db = [PromptItem(**item) for item in json.load(f)]
            logger.info(f"加载 {len(prompts_db)} 条提示词")
        
        # 加载评估记录
        if os.path.exists("data/judgments.json"):
            with open("data/judgments.json", "r") as f:
                judgments_db = [Judgment(**item) for item in json.load(f)]
            logger.info(f"加载 {len(judgments_db)} 条评估记录")
        
        # 加载LoRA模型
        if os.path.exists("data/lora_models.json"):
            with open("data/lora_models.json", "r") as f:
                lora_db = [LoRAModel(**item) for item in json.load(f)]
            logger.info(f"加载 {len(lora_db)} 个LoRA模型")
        
        logger.info("系统数据加载完成")
    except Exception as e:
        logger.error(f"加载初始数据失败: {str(e)}")
        raise

def save_settings():
    """保存系统设置"""
    try:
        logger.info(f"保存系统设置: {settings}")
        with open("data/settings.json", "w") as f:
            json.dump(settings.dict(), f, indent=2)
    except Exception as e:
        logger.error(f"保存设置失败: {str(e)}")
        raise

def save_prompts():
    """保存提示词数据"""
    try:
        logger.info(f"保存 {len(prompts_db)} 条提示词")
        with open("data/prompts.json", "w") as f:
            json.dump([item.dict() for item in prompts_db], f, indent=2)
    except Exception as e:
        logger.error(f"保存提示词失败: {str(e)}")
        raise

def save_judgments():
    """保存评估记录"""
    try:
        logger.info(f"保存 {len(judgments_db)} 条评估记录")
        with open("data/judgments.json", "w") as f:
            json.dump([item.dict() for item in judgments_db], f, indent=2)
    except Exception as e:
        logger.error(f"保存评估记录失败: {str(e)}")
        raise

def save_lora_models():
    """保存LoRA模型数据"""
    try:
        logger.info(f"保存 {len(lora_db)} 个LoRA模型")
        with open("data/lora_models.json", "w") as f:
            json.dump([item.dict() for item in lora_db], f, indent=2)
    except Exception as e:
        logger.error(f"保存LoRA模型失败: {str(e)}")
        raise

def generate_judge_queue():
    """从train_60000.json生成评估队列"""
    global current_judge_queue
    try:
        logger.info("开始生成评估队列")
        train_data_path = "/workspace/GoldenNoise/Golden-Noise-for-Diffusion-Models/data/train_60000.json"
        
        if not os.path.exists(train_data_path):
            logger.warning(f"训练数据文件不存在: {train_data_path}")
            current_judge_queue = []
            return

        with open(train_data_path, "r") as f:
            train_data = json.load(f)
        
        current_judge_queue = []
        for idx, item in enumerate(train_data[:100]):  # 限制前100条作为示例
            pair = ImagePair(
                id=str(idx+1),
                prompt_id=item.get("id", str(idx+1)),
                image_a=item.get("image_a", f"/static/images/{idx+1}/a.jpg"),
                image_b=item.get("image_b", f"/static/images/{idx+1}/b.jpg"),
                model_name=settings.model_name,
                seed=item.get("seed", hash(item["id"]) % 1000000)
            )
            current_judge_queue.append(pair)
        
        logger.info(f"生成 {len(current_judge_queue)} 个评估对")
    except Exception as e:
        logger.error(f"生成评估队列失败: {str(e)}")
        raise

# --- 全局状态 ---
settings = SystemSettings()
prompts_db: List[PromptItem] = []
judgments_db: List[Judgment] = []
lora_db: List[LoRAModel] = []
current_judge_queue: List[ImagePair] = []

# 创建必要目录
os.makedirs("data/prompts", exist_ok=True)
os.makedirs("data/judgments", exist_ok=True)
os.makedirs("static/lora_models", exist_ok=True)
os.makedirs("static/images", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 初始化数据
try:
    load_initial_data()
    generate_judge_queue()
except Exception as e:
    logger.critical(f"系统初始化失败: {str(e)}")
    raise

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- API路由 ---
@app.post("/api/initialize")
async def initialize_system(
    prompt_file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """系统初始化"""
    try:
        logger.info(f"开始系统初始化，模型: {model_name}")
        
        # 保存提示词文件
        prompt_data = json.loads(await prompt_file.read())
        prompt_file_path = f"data/prompts/{prompt_file.filename}"
        
        with open(prompt_file_path, "w") as f:
            json.dump(prompt_data, f, indent=2)
        logger.info(f"提示词文件保存至: {prompt_file_path}")
        
        # 更新系统设置
        settings.model_name = model_name
        save_settings()
        
        # 生成评估队列
        generate_judge_queue()
        
        logger.info("系统初始化成功")
        return {"success": True, "message": "系统初始化成功"}
    
    except Exception as e:
        logger.error(f"系统初始化失败: {str(e)}")
        raise HTTPException(500, detail=f"初始化失败: {str(e)}")

@app.get("/api/settings")
async def get_system_settings():
    """获取当前系统设置"""
    logger.info("获取系统设置")
    return settings

@app.post("/api/settings/save")
async def save_system_settings(new_settings: SystemSettings):
    """保存系统设置"""
    try:
        logger.info(f"保存新系统设置: {new_settings}")
        global settings
        settings = new_settings
        save_settings()
        return {"success": True, "message": "设置已保存"}
    except Exception as e:
        logger.error(f"保存系统设置失败: {str(e)}")
        raise HTTPException(500, detail=f"保存失败: {str(e)}")

@app.get("/api/prompts/{prompt_id}/next_comparison")
async def get_next_comparison(prompt_id: str):
    """获取下一组待比较的图像对 (新API)"""
    try:
        if not current_judge_queue:
            generate_judge_queue()
            if not current_judge_queue:
                raise HTTPException(404, detail="No images available for judging")
        
        next_pair = current_judge_queue[0]
        logger.info(f"获取下一组比较图像: {next_pair.id}")
        
        return {
            "prompt_id": next_pair.prompt_id,
            "prompt_text": f"Prompt {next_pair.prompt_id}",  # TODO: 从prompts_db获取实际文本
            "choice_a": {
                "seed": next_pair.seed,
                "image_url": next_pair.image_a
            },
            "choice_b": {
                "seed": next_pair.seed + 1,  # 示例种子
                "image_url": next_pair.image_b
            }
        }
    except Exception as e:
        logger.error(f"获取比较图像失败: {str(e)}")
        raise HTTPException(500, detail=f"获取失败: {str(e)}")

@app.post("/api/judgments")
async def submit_judgment(judgment: Judgment):
    """提交评估结果 (新API)"""
    try:
        judgment.judged_at = datetime.now().isoformat()
        judgments_db.append(judgment)
        save_judgments()
        
        logger.info(f"提交评估结果: {judgment}")
        
        # 更新提示词的评估计数
        for prompt in prompts_db:
            if prompt.id == judgment.prompt_id:
                prompt.num_judgments += 1
                save_prompts()
                logger.info(f"更新提示词 {prompt.id} 评估计数")
                break
        
        return {"success": True, "message": "评估已提交"}
    except Exception as e:
        logger.error(f"提交评估失败: {str(e)}")
        raise HTTPException(500, detail=f"提交失败: {str(e)}")

# 保留旧API暂时兼容
@app.get("/api/judge/queue")
async def get_judge_queue():
    """获取当前评估队列 (旧API)"""
    logger.info(f"获取评估队列，共 {len(current_judge_queue)} 项")
    return {"items": current_judge_queue}

@app.post("/api/judge/submit")
async def submit_judgment_old(judgment_data: dict):
    """提交评估结果 (旧API转换层)"""
    try:
        # 转换旧格式数据到新格式
        new_judgment = Judgment(
            prompt_id=judgment_data["image_pair_id"].split("_")[0],
            winner_seed=judgment_data.get("winner_seed", 0),
            loser_seed=judgment_data.get("loser_seed", 0),
            user_id="legacy_user",
            judged_at=datetime.now().isoformat()
        )
        return await submit_judgment(new_judgment)
    except Exception as e:
        logger.error(f"旧API请求转换失败: {str(e)}")
        raise HTTPException(400, detail="无法处理旧格式请求")

@app.get("/api/prompts")
async def get_all_prompts():
    """获取所有提示词"""
    logger.info("获取所有提示词")
    return {"prompts": prompts_db}

@app.get("/api/lora/list")
async def list_lora_models():
    """获取所有LoRA模型"""
    logger.info("获取LoRA模型列表")
    return {"models": lora_db}

@app.post("/api/lora/upload")
async def upload_lora_model(file: UploadFile, name: str = Form(...)):
    """上传LoRA模型"""
    try:
        logger.info(f"开始上传LoRA模型: {name}")
        file_path = f"static/lora_models/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        lora_id = str(len(lora_db) + 1)
        new_lora = LoRAModel(
            id=lora_id,
            name=name,
            path=file_path
        )
        lora_db.append(new_lora)
        save_lora_models()
        
        logger.info(f"LoRA模型上传成功: {new_lora}")
        return {"success": True, "model": new_lora}
    except Exception as e:
        logger.error(f"LoRA模型上传失败: {str(e)}")
        raise HTTPException(500, detail=f"上传失败: {str(e)}")

@app.delete("/api/lora/delete/{lora_id}")
async def delete_lora_model(lora_id: str):
    """删除LoRA模型"""
    try:
        logger.info(f"删除LoRA模型 ID: {lora_id}")
        global lora_db
        lora_db = [m for m in lora_db if m.id != lora_id]
        save_lora_models()
        return {"success": True, "message": "LoRA已删除"}
    except Exception as e:
        logger.error(f"删除LoRA模型失败: {str(e)}")
        raise HTTPException(500, detail=f"删除失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("启动FastAPI服务")
    uvicorn.run(app, host="0.0.0.0", port=8000)
