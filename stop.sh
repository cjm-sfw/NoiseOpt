#!/bin/bash

# 停止GoldenNoise数据采集平台

# 检查PID文件是否存在
if [ ! -f "logs/backend.pid" ] || [ ! -f "logs/frontend.pid" ]; then
    echo "PID文件不存在，尝试通过进程名查找并终止进程..."
    
    # 查找并终止所有相关的python进程
    BACKEND_PIDS=$(pgrep -f "python.*backend/main.py")
    FRONTEND_PIDS=$(pgrep -f "python.*frontend/server.py")
    
    if [ ! -z "$BACKEND_PIDS" ]; then
        echo "正在停止后端服务 (PIDs: $BACKEND_PIDS)..."
        echo $BACKEND_PIDS | xargs kill -9 2>/dev/null || true
        # 等待进程终止
        for pid in $BACKEND_PIDS; do
            timeout 10 bash -c "while kill -0 $pid 2>/dev/null; do sleep 1; done" 2>/dev/null || kill -9 $pid 2>/dev/null || true
        done
        echo "后端服务已停止"
    else
        echo "未找到运行中的后端服务"
    fi
    
    if [ ! -z "$FRONTEND_PIDS" ]; then
        echo "正在停止前端服务 (PIDs: $FRONTEND_PIDS)..."
        echo $FRONTEND_PIDS | xargs kill -9 2>/dev/null || true
        # 等待进程终止
        for pid in $FRONTEND_PIDS; do
            timeout 10 bash -c "while kill -0 $pid 2>/dev/null; do sleep 1; done" 2>/dev/null || kill -9 $pid 2>/dev/null || true
        done
        echo "前端服务已停止"
    else
        echo "未找到运行中的前端服务"
    fi
else
    # 读取PID
    BACKEND_PID=$(cat logs/backend.pid)
    FRONTEND_PID=$(cat logs/frontend.pid)

    # 停止后端服务
    if ps -p $BACKEND_PID > /dev/null; then
        echo "正在停止后端服务 (PID: $BACKEND_PID)..."
        kill -9 $BACKEND_PID 2>/dev/null || true
        # 等待进程终止
        timeout 10 bash -c "while kill -0 $BACKEND_PID 2>/dev/null; do sleep 1; done" 2>/dev/null || kill -9 $BACKEND_PID 2>/dev/null || true
        echo "后端服务已停止"
    else
        echo "后端服务未运行"
    fi

    # 停止前端服务
    if ps -p $FRONTEND_PID > /dev/null; then
        echo "正在停止前端服务 (PID: $FRONTEND_PID)..."
        kill -9 $FRONTEND_PID 2>/dev/null || true
        # 等待进程终止
        timeout 10 bash -c "while kill -0 $FRONTEND_PID 2>/dev/null; do sleep 1; done" 2>/dev/null || kill -9 $FRONTEND_PID 2>/dev/null || true
        echo "前端服务已停止"
    else
        echo "前端服务未运行"
    fi

    # 删除PID文件
    rm -f logs/backend.pid logs/frontend.pid
fi

# 清理僵尸进程
echo "清理僵尸进程..."
ps aux | grep defunct | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

echo "平台已停止"
