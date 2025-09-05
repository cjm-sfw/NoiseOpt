#!/bin/bash

# 清理GoldenNoise数据采集平台的所有进程

echo "开始清理所有相关进程..."

# 查找并终止所有相关的python进程
BACKEND_PIDS=$(pgrep -f "python.*backend/main.py")
FRONTEND_PIDS=$(pgrep -f "python.*frontend/server.py")

if [ ! -z "$BACKEND_PIDS" ]; then
    echo "正在终止后端服务 (PIDs: $BACKEND_PIDS)..."
    echo $BACKEND_PIDS | xargs kill -9 2>/dev/null || true
    # 等待进程终止
    for pid in $BACKEND_PIDS; do
        timeout 10 bash -c "while kill -0 $pid 2>/dev/null; do sleep 1; done" 2>/dev/null || kill -9 $pid 2>/dev/null || true
    done
    echo "后端服务已终止"
else
    echo "未找到运行中的后端服务"
fi

if [ ! -z "$FRONTEND_PIDS" ]; then
    echo "正在终止前端服务 (PIDs: $FRONTEND_PIDS)..."
    echo $FRONTEND_PIDS | xargs kill -9 2>/dev/null || true
    # 等待进程终止
    for pid in $FRONTEND_PIDS; do
        timeout 10 bash -c "while kill -0 $pid 2>/dev/null; do sleep 1; done" 2>/dev/null || kill -9 $pid 2>/dev/null || true
    done
    echo "前端服务已终止"
else
    echo "未找到运行中的前端服务"
fi

# 清理PID文件
rm -f logs/backend.pid logs/frontend.pid 2>/dev/null || true

# 清理僵尸进程
echo "清理僵尸进程..."
ps aux | grep defunct | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

# 清理所有与平台相关的进程
echo "清理所有相关进程..."
pkill -f "python.*NoiseOpt" 2>/dev/null || true

echo "进程清理完成"
