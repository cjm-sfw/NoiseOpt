# GoldenNoise 采集平台 — 辅助开发指导文档
## 1. 项目目标
- 搭建一个最小可运行 demo（MVP），用于采集文生图模型的 Golden Noise。
- 支持人工挑选高质量图像，并形成可分析的数据集。
- 为后续 Noise Optimization 研究 提供基础数据和接口。
---
## 2. 技术栈
- 后端：FastAPI
- 前端：HTML / CSS / JavaScript（单页 + 两 Tab）
- 模型加载：Diffusers (FLux1.dev-schnell)
- 辅助评分：CLIP 模型
- 数据存储：JSON
- 可视化 / 分析 (暂不完成，后续补充)：Python (Matplotlib / Seaborn / t-SNE / UMAP)
---
## 3. 系统架构
### 3.1 前端（单页两 Tab）
1. Tab1：参数设置
    - 模型选择
    - 参数配置：steps、CFG scale、scheduler、image size
    - Prompt 文件上传（CSV/JSON）
    - 初始化按钮 → 调用 /init_model
    - 可选 CLIP 模型开关
2. Tab2：工作台
    - Prompt 展示
    - 批量图像生成对比（grid 或两两对比）
    - 选择 Golden 图像
    - 提交按钮 → /submit_selection
    - “下一组”按钮 → /generate_batch 获取下一批图像
    - 可显示辅助指标（CLIP-score、推荐 seed）
---
### 3.2 后端（FastAPI）
1. 模型管理模块
    - 加载 / 切换 Diffusers 模型
    - 可选加载 CLIP 模型
2. 图像生成模块
    - 输入：prompt + 参数 + seed
    - 输出：batch 图像（base64 或文件路径）
3. 交互服务模块
    - /init_model：初始化模型
    - /generate_batch：生成指定 batch 图像
    - /submit_selection：提交用户选择
    - /get_next_prompt：获取下一个 prompt
---
### 3.3 数据存储
1. 元数据 DB (JSON/CSV)
    - prompt
    - Golden seed
    - Bad seeds
    - 模型名/版本
    - 参数 (steps, scheduler, CFG 等)
    - 评分指标（CLIP-score 等）
2. 图像存储目录
    - 保存生成图像文件（按 prompt/seed 命名）
---
### 3.4 分析与研究
- Golden vs Bad Noise 对比
- 聚类分析（种子分布、Golden seed 簇）
- Noise Optimization
    - 训练 Seed Selector
    - Noise Embedding 学习
- 分析结果可反馈给前端工作台，辅助用户选择或推荐种子
---
## 4. 用户操作流程（时序概览）
1. 用户进入页面 → 默认 Tab1
2. 选择模型、参数、上传 Prompt → 点击初始化
3. 切换 Tab2 → 请求第一个 prompt + batch 图像
4. 对比选择 Golden → 提交
5. 点击“下一组” → 循环生成 → 选择 → 提交
6. 数据写入 DB / 图像目录 → 可用于后续分析
---
## 5. 开发指导
1. 前端
    - 单页设计 + 两 Tab 切换
    - 使用 grid 或两两对比展示 batch 图像
    - 快捷键支持提升标注效率
2. 后端
    - 模块化：模型管理 / 图像生成 / 交互服务
    - API 返回统一格式：包含图像路径、seed、参数信息
    - 支持多模型和多参数组合
3. 数据
    - 每条生成记录保存完整元信息
    - 保留 Bad Noise，用于后续对比分析
4. 扩展性
    - 后端可接入新模型
    - 前端可增加分析 Tab
    - 支持未来自动化辅助（Active Learning / CLIP-score 预筛选）
---
## 6. 建议开发顺序
1. 搭建 FastAPI 基础框架，能加载模型并生成图像
2. 构建前端两 Tab 页面，完成初始化 + 图像显示
3. 完成用户选择和提交接口
4. 数据存储落盘（DB + 图像）
5. 实现简单分析模块（统计、可视化）
6. 增加辅助功能（CLIP-score、快捷键、Grid 展示）
7. 可选：训练 Seed Selector 或 Noise Embedding

|阶段|模块|功能点|优先级|备注/目标|
|-|-|-|-|-|
|阶段1：基础框架搭建|后端 FastAPI|初始化项目结构、安装依赖|高|包含 Diffusers、FastAPI、Pydantic、数据库库|
| |后端 FastAPI|实现 /init_model 接口|高|能加载指定文生图模型|
| |后端 FastAPI|实现 /generate_batch 接口|高|输入 prompt + 参数 → 输出 batch 图像（base64 或文件路径）|
| |数据存储|创建 SQLite/JSON/CSV 文件结构|高|确保可以保存 prompt、seed、参数、评分等信息|
|阶段2：前端基础页面|前端 HTML/CSS/JS|单页 + 两 Tab 架构|高|Tab1: 参数设置，Tab2: 工作台|
| |前端 JS|Tab 切换逻辑|高|切换时保留状态，不丢失已选择数据|
| |前端 JS|上传 Prompt 文件 & 初始化模型|高|调用 /init_model 接口|
|阶段3：工作台功能|前端 JS|批量图像展示（grid 或两两对比）|高|每张图显示 seed 和多选框|
| |前端 JS|用户选择 Golden 图像 → 提交|高|调用 /submit_selection 接口，写入 DB|
| |前端 JS|“下一组”按钮 → 请求新 batch|高|调用 /generate_batch + /get_next_prompt|
|阶段4：数据管理|后端 / 数据库|保存生成图像到目录|高|按 prompt/seed 命名|
| |后端 / 数据库|保存元数据（prompt, Golden/Bad seed, 参数, 评分）|高|保证完整记录，用于分析|
|阶段5：辅助功能|前端 JS|快捷键支持|中|提高人工标注效率|
| |前端 JS|显示辅助指标（CLIP-score / 推荐 seed）|中|可选，用于加速选择|
| |后端 FastAPI|可选 CLIP 模型评分接口|中|对生成图像打分|
|阶段6：分析模块|分析 Python 脚本|Golden vs Bad Noise 对比|中|t-SNE/UMAP 可视化|
| |分析 Python 脚本|聚类分析（Golden seed 簇）|中|探索 noise 共性|
| |分析 Python 脚本|Noise Optimization 初步实验|中|Seed Selector 或 Noise Embedding|
|阶段7：扩展功能|前端|历史结果回顾 Tab|低|查看已选出的 Golden 图像|
| |后端|多模型支持|低|可切换不同文生图模型|
| |前端|Active Learning / 自动推荐候选|低|CLIP-score 预筛选节省人工成本|

---
## 7. 开发实施计划

### 7.1 当前实施步骤
1. 搭建FastAPI基础框架
   - 安装依赖（FastAPI, Diffusers, Transformers, Pydantic）
   - 实现FLux1.dev-schnell模型加载
   - 集成CLIP模型评分功能

2. 开发图像处理模块
   - 实现图像生成后自动计算CLIP score
   - 创建统一的API响应格式

3. 构建前端页面框架
   - 创建单页双Tab界面
   - 实现参数设置和工作台基本功能

4. 实现JSON数据存储
   - 设计数据结构
   - 实现数据持久化

### 7.2 当前进度
- [x] 完成需求分析和技术选型
- [x] 确定开发优先级
- [x] 明确具体实现细节
- [x] 开始基础框架搭建
- [x] 完成后端FastAPI框架搭建
- [x] 实现FLux1.dev-schnell模型加载
- [x] 集成CLIP模型评分功能
- [x] 开发图像处理模块
- [x] 实现图像生成后自动计算CLIP score
- [x] 创建统一的API响应格式
- [x] 构建前端页面框架
- [x] 创建单页双Tab界面
- [x] 实现参数设置和工作台基本功能
- [x] 实现JSON数据存储
- [x] 设计数据结构
- [x] 实现数据持久化


