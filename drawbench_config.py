"""
DrawBench数据集配置
"""
import os
from config import DRAWBENCH_DIR

# DrawBench数据集配置
DRAWBENCH_CONFIG = {
    "version": "1.0",
    "base_url": "https://huggingface.co/datasets/laion/DrawBench/resolve/main",
    
    # 基准测试名称列表
    "benchmark_names": [
        "simple_objects",
        "complex_scenes",
        "animals",
        "vehicles",
        "fashion",
        "architecture",
        "art_styles",
        "abstract_concepts",
        "daily_life",
        "fantasy_creatures"
    ],
    
    # 每个基准测试的配置
    "benchmarks": {
        "simple_objects": {
            "description": "简单物体生成基准测试",
            "prompts": [
                "a red apple on a white background",
                "a yellow banana on a blue background",
                "a green cube on a gray surface",
                "a glass sphere on a wooden table",
                "a metal cylinder on a black background"
            ],
            "download_url": f"{DRAWBENCH_DIR}/simple_objects",
            "num_samples": 100
        },
        "complex_scenes": {
            "description": "复杂场景生成基准测试",
            "prompts": [
                "a bustling city street at night with neon lights",
                "a peaceful forest with sunlight filtering through trees",
                "a mountain village with snow-covered peaks",
                "an underwater scene with colorful coral and fish",
                "a futuristic city with flying cars and holograms"
            ],
            "download_url": f"{DRAWBENCH_DIR}/complex_scenes",
            "num_samples": 100
        },
        "animals": {
            "description": "动物生成基准测试",
            "prompts": [
                "a golden retriever puppy playing in a park",
                "a tiger in a jungle environment",
                "a flock of birds flying in the sky",
                "a cat sitting on a windowsill",
                "a school of fish swimming in an aquarium"
            ],
            "download_url": f"{DRAWBENCH_DIR}/animals",
            "num_samples": 100
        },
        "vehicles": {
            "description": "交通工具生成基准测试",
            "prompts": [
                "a classic 1969 Chevrolet Camaro",
                "a modern electric car on a highway",
                "a futuristic flying vehicle in a city",
                "a vintage airplane in a hangar",
                "a high-speed train in a mountainous region"
            ],
            "download_url": f"{DRAWBENCH_DIR}/vehicles",
            "num_samples": 100
        },
        "fashion": {
            "description": "时尚服装生成基准测试",
            "prompts": [
                "a woman wearing a red evening gown",
                "a man in a business suit and tie",
                "a model wearing futuristic clothing",
                "a traditional Japanese kimono",
                "a sports outfit for running"
            ],
            "download_url": f"{DRAWBENCH_DIR}/fashion",
            "num_samples": 100
        },
        "architecture": {
            "description": "建筑生成基准测试",
            "prompts": [
                "a modern minimalist house",
                "a medieval castle with towers",
                "a skyscraper in a city skyline",
                "a traditional Chinese temple",
                "a futuristic building with glass facade"
            ],
            "download_url": f"{DRAWBENCH_DIR}/architecture",
            "num_samples": 100
        },
        "art_styles": {
            "description": "艺术风格生成基准测试",
            "prompts": [
                "impressionist painting of a garden",
                "cubist portrait of a woman",
                "surrealist scene with melting clocks",
                "abstract expressionist canvas",
                "realistic portrait in oil painting style"
            ],
            "download_url": f"{DRAWBENCH_DIR}/art_styles",
            "num_samples": 100
        },
        "abstract_concepts": {
            "description": "抽象概念生成基准测试",
            "prompts": [
                "the concept of freedom represented visually",
                "a visualization of happiness",
                "the idea of time passing",
                "a representation of love",
                "an abstract depiction of hope"
            ],
            "download_url": f"{DRAWBENCH_DIR}/abstract_concepts",
            "num_samples": 100
        },
        "daily_life": {
            "description": "日常生活场景生成基准测试",
            "prompts": [
                "a family having breakfast in the morning",
                "people commuting to work on a rainy day",
                "children playing in a park",
                "a chef cooking in a kitchen",
                "a student studying in a library"
            ],
            "download_url": f"{DRAWBENCH_DIR}/daily_life",
            "num_samples": 100
        },
        "fantasy_creatures": {
            "description": "幻想生物生成基准测试",
            "prompts": [
                "a dragon flying over a castle",
                "a unicorn in a magical forest",
                "a griffin perched on a mountain",
                "a mermaid swimming underwater",
                "a fantasy elf character"
            ],
            "download_url": f"{DRAWBENCH_DIR}/fantasy_creatures",
            "num_samples": 100
        }
    }
}

def get_drawbench_dir():
    """获取DrawBench数据集存储目录"""
    return DRAWBENCH_DIR

def get_drawbench_benchmark_path(benchmark_name):
    """获取特定基准测试的存储路径"""
    if benchmark_name not in DRAWBENCH_CONFIG["benchmarks"]:
        raise ValueError(f"未知的基准测试名称: {benchmark_name}")
    
    return os.path.join(DRAWBENCH_DIR, benchmark_name)
