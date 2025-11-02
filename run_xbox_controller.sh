#!/bin/bash
# Run Xbox controller in background with logging

LOG_FILE="/home/morgan/dogbot/xbox_controller.log"
PID_FILE="/home/morgan/dogbot/xbox_controller.pid"

# Function to stop the controller
stop_controller() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Stopping Xbox controller (PID: $PID)..."
        sudo kill $PID 2>/dev/null
        rm -f "$PID_FILE"
        echo "Xbox controller stopped."
    else
        echo "Xbox controller not running."
    fi
}

# Function to start the controller
start_controller() {
    stop_controller
    echo "Starting Xbox controller..."

    # Start in background and save PID
    sudo python3 /home/morgan/dogbot/xbox_api_controller.py > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"

    echo "Xbox controller started (PID: $PID)"
    echo "Logs: tail -f $LOG_FILE"
}

# Function to check status
status_controller() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null; then
            echo "Xbox controller is running (PID: $PID)"
            echo "Recent logs:"
            tail -10 "$LOG_FILE"
        else
            echo "Xbox controller stopped unexpectedly"
            rm -f "$PID_FILE"
        fi
    else
        echo "Xbox controller not running"
    fi
}

# Main script
case "$1" in
    start)
        start_controller
        ;;
    stop)
        stop_controller
        ;;
    restart)
        stop_controller
        start_controller
        ;;
    status)
        status_controller
        ;;
    logs)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac