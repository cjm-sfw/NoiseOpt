#!/bin/bash

# Golden Noise System Management Script
# Features: Service status check, start, stop, restart

REDIS_PID_FILE="/tmp/redis-server.pid"
CELERY_PID_FILE="/tmp/celery_worker.pid"
FASTAPI_PID_FILE="/tmp/fastapi.pid"

check_redis() {
    if service redis-server status | grep -q "running"; then
        echo "Redis service: Running"
        return 0
    else
        echo "Redis service: Stopped"
        return 1
    fi
}

check_celery() {
    if ps -p $(cat $CELERY_PID_FILE 2>/dev/null) > /dev/null 2>&1; then
        echo "Celery Worker: Running"
        return 0
    else
        echo "Celery Worker: Stopped"
        return 1
    fi
}

check_fastapi() {
    if ps -p $(cat $FASTAPI_PID_FILE 2>/dev/null) > /dev/null 2>&1; then
        echo "FastAPI service: Running"
        return 0
    else
        echo "FastAPI service: Stopped"
        return 1
    fi
}

start_services() {
    echo "Starting system services..."
    
    # Start Redis
    if ! check_redis; then
        sudo service redis-server start
        sleep 2
    fi
    
    # Start Celery
    if ! check_celery; then
        celery -A tasks worker --loglevel=info --pool=solo -c 1 > celery.log 2>&1 &
        echo $! > $CELERY_PID_FILE
        sleep 2
    fi
    
    # Start FastAPI
    if ! check_fastapi; then
        uvicorn app:app --reload --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &
        echo $! > $FASTAPI_PID_FILE
        sleep 2
    fi
    
    echo "Services started successfully"
}

stop_services() {
    echo "Stopping system services..."
    
    # Stop FastAPI
    if check_fastapi; then
        kill $(cat $FASTAPI_PID_FILE)
        rm $FASTAPI_PID_FILE
    fi
    
    # Stop Celery
    if check_celery; then
        kill $(cat $CELERY_PID_FILE)
        rm $CELERY_PID_FILE
    fi
    
    # Stop Redis
    if check_redis; then
        sudo service redis-server stop
    fi
    
    echo "Services stopped successfully"
}

restart_services() {
    stop_services
    start_services
}

show_status() {
    echo "====== Service Status ======"
    check_redis
    check_celery
    check_fastapi
    echo "==========================="
}

# Main menu
while true; do
    echo ""
    echo "Golden Noise System Management"
    echo "1. Check service status"
    echo "2. Start all services"
    echo "3. Stop all services"
    echo "4. Restart all services"
    echo "5. Exit"
    echo ""
    read -p "Please select an option (1-5): " choice
    
    case $choice in
        1) show_status ;;
        2) start_services ;;
        3) stop_services ;;
        4) restart_services ;;
        5) exit 0 ;;
        *) echo "Invalid option" ;;
    esac
done
