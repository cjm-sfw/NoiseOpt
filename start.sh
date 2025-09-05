#!/bin/bash

# 启动GoldenNoise数据采集平台

# 检查是否已经运行，如果运行则尝试停止
if pgrep -f "python.*backend/main.py" > /dev/null || pgrep -f "python.*frontend/server.py" > /dev/null; then
    echo "平台已经在运行中，正在尝试停止现有进程..."
    ./stop.sh
    sleep 3
fi

# 创建日志目录
mkdir -p logs

# 启动后端服务
echo "正在启动后端服务..."
cd backend
nohup python main.py > backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# 等待后端启动
sleep 3

# 启动前端服务
echo "正在启动前端服务..."
cd frontend
nohup python server.py > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# 保存PID到文件
echo $BACKEND_PID > logs/backend.pid
echo $FRONTEND_PID > logs/frontend.pid

echo "平台已启动："
echo "  后端服务: http://localhost:8000 (PID: $BACKEND_PID)"
echo "  前端服务: http://localhost:3000 (PID: $FRONTEND_PID)"
echo "日志文件:"
echo "  后端日志: backend/backend.log"
echo "  前端日志: frontend/frontend.log"
echo "  启动脚本日志: logs/ 目录中"
echo "使用 stop.sh 停止服务"
