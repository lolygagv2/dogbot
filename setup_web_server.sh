#!/bin/bash
# Setup script for DogBot Web Server
# Run this ON the Raspberry Pi

echo "======================================"
echo "DogBot Web Server Setup"
echo "======================================"

# Navigate to dogbot directory
cd /home/morgan/dogbot

# Install Flask and required packages
echo "Installing Flask and dependencies..."
pip3 install flask flask-cors

# Create necessary directories
echo "Creating directories..."
mkdir -p snapshots
mkdir -p logs
mkdir -p config

# Save the web interface HTML
echo "Creating web interface HTML file..."
cat > web_interface.html << 'EOF'
<!-- Copy the entire HTML from the web interface artifact here -->
<!-- This is a placeholder - replace with actual HTML content -->
<!DOCTYPE html>
<html>
<head><title>DogBot Control</title></head>
<body>
<h1>DogBot Web Interface</h1>
<p>Replace this with the actual HTML interface</p>
</body>
</html>
EOF

# Create a simple start script
echo "Creating start script..."
cat > start_web.sh << 'EOF'
#!/bin/bash
echo "Starting DogBot Web Server..."
python3 /home/morgan/dogbot/web_server.py
EOF

chmod +x start_web.sh

# Create systemd service (optional - for auto-start)
echo "Creating systemd service file..."
sudo tee /etc/systemd/system/dogbot-web.service << EOF
[Unit]
Description=DogBot Web Server
After=network.target

[Service]
Type=simple
User=morgan
WorkingDirectory=/home/morgan/dogbot
ExecStart=/usr/bin/python3 /home/morgan/dogbot/web_server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To run the web server:"
echo "  1. Manual start:"
echo "     cd /home/morgan/dogbot"
echo "     python3 web_server.py"
echo ""
echo "  2. Or use the start script:"
echo "     ./start_web.sh"
echo ""
echo "  3. To enable auto-start on boot:"
echo "     sudo systemctl enable dogbot-web.service"
echo "     sudo systemctl start dogbot-web.service"
echo ""
echo "Then access from:"
echo "  - This Pi: http://localhost:5000"
echo "  - Your PC: http://raspberrypi.local:5000"
echo ""
echo "======================================"