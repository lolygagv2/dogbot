#!/bin/bash
# Control script for Xbox controller service
# This avoids EPERM issues by using systemd

SERVICE_NAME="xbox-controller"
SERVICE_FILE="/home/morgan/dogbot/xbox-controller.service"
SYSTEMD_PATH="/etc/systemd/system/xbox-controller.service"

install_service() {
    echo "Installing Xbox controller service..."
    sudo cp "$SERVICE_FILE" "$SYSTEMD_PATH"
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    echo "Service installed and enabled"
}

start_controller() {
    echo "Starting Xbox controller service..."
    sudo systemctl start $SERVICE_NAME
    sleep 2
    status_controller
}

stop_controller() {
    echo "Stopping Xbox controller service..."
    sudo systemctl stop $SERVICE_NAME
}

restart_controller() {
    echo "Restarting Xbox controller service..."
    sudo systemctl restart $SERVICE_NAME
    sleep 2
    status_controller
}

status_controller() {
    echo "Xbox controller service status:"
    sudo systemctl status $SERVICE_NAME --no-pager | head -10
}

logs_controller() {
    echo "Xbox controller logs (Ctrl+C to exit):"
    sudo journalctl -u $SERVICE_NAME -f
}

case "$1" in
    install)
        install_service
        ;;
    start)
        start_controller
        ;;
    stop)
        stop_controller
        ;;
    restart)
        restart_controller
        ;;
    status)
        status_controller
        ;;
    logs)
        logs_controller
        ;;
    *)
        echo "Usage: $0 {install|start|stop|restart|status|logs}"
        echo ""
        echo "  install - Install the systemd service (run once)"
        echo "  start   - Start the Xbox controller"
        echo "  stop    - Stop the Xbox controller"
        echo "  restart - Restart the Xbox controller"
        echo "  status  - Check if controller is running"
        echo "  logs    - Watch live logs"
        exit 1
        ;;
esac