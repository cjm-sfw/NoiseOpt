# GoldenNoise 数据采集平台

## 项目简介

GoldenNoise 数据采集平台是一个用于收集文生图模型中"Golden Noise"（优质噪声）的数据集采集工具。该平台支持人工挑选高质量图像，并形成可分析的数据集，为后续的 Noise Optimization 研究提供基础数据和接口。

## 技术栈

- 后端：FastAPI + Diffusers + Transformers
- 前端：HTML + CSS + JavaScript
- 模型：FLux1.dev-schnell
- 辅助评分：CLIP 模型
- 数据存储：JSON + 图像文件

## 功能特性

1. 支持 FLux1.dev-schnell 模型加载和图像生成
2. 集成 CLIP 模型自动评分功能
3. 双标签页界面：参数设置和工作台
4. 图像批量生成和展示
5. Golden Image 选择和提交
6. JSON 格式数据存储

## 项目结构

```
NoiseOpt/
├── backend/
│   └── main.py          # FastAPI 后端服务
├── frontend/
│   └── index.html       # 前端界面
├── data/
│   ├── images/          # 生成的图像文件
│   └── database.json    # 数据库存储文件
├── Dev.md               # 开发指导文档
└── README.md            # 项目说明文档
```

## 安装和运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动平台：
```bash
./start.sh
```

3. 停止平台：
```bash
./stop.sh
```

4. 清理所有进程（如果stop.sh无法终止所有进程）：
```bash
./cleanup.sh
```

4. 访问平台：
- 前端界面：http://localhost:3000
- 后端API：http://localhost:8000

## 使用说明

1. 在"参数设置"标签页中配置模型参数
2. 点击"初始化模型"加载 FLux1.dev-schnell 和 CLIP 模型
3. 切换到"工作台"标签页
4. 点击"生成图像"按钮生成一批图像
5. 根据图像质量和 CLIP 分数选择 Golden Image
6. 点击"提交选择"保存数据到数据库
7. 点击"下一组"继续处理下一个 prompt

## 数据格式

数据存储在 `data/database.json` 文件中，每条记录包含：
- prompt: 生成图像的文本提示
- golden_images: 用户选择的优质图像信息
- all_images: 所有生成的图像信息

每张图像信息包括：
- seed: 生成图像的随机种子
- clip_score: CLIP 模型评分
- image_path: 图像文件路径

## 开发计划

详细的开发计划和进度请参考 [Dev.md](Dev.md) 文件。
