#!/bin/bash
# TreatBot LED Setup Script
# Installs dependencies and tests LED system

echo "🎨 TreatBot LED System Setup"
echo "=============================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root (use sudo)"
    exit 1
fi

# Enable SPI (required for NeoPixels)
echo "📡 Enabling SPI interface..."
raspi-config nonint do_spi 0

# Install Python dependencies
echo "📦 Installing Python packages..."
apt update
apt install -y python3-pip python3-dev python3-rpi.gpio

# Install NeoPixel library
pip3 install adafruit-circuitpython-neopixel

# Install additional dependencies
pip3 install board rpi_ws281x adafruit-blinka

# Check GPIO permissions
echo "🔐 Setting up GPIO permissions..."
usermod -a -G gpio pi
usermod -a -G spi pi

# Create LED test directory
LED_DIR="/home/pi/treatbot/led_system"
mkdir -p $LED_DIR
chown pi:pi $LED_DIR

echo "✅ LED dependencies installed!"
echo ""
echo "🔧 Hardware Checklist:"
echo "   □ NeoPixel ring connected to GPIO12 (Pin 32)"
echo "   □ NeoPixel ground connected to GND"
echo "   □ NeoPixel power connected to 3.3V or 5V"
echo "   □ Blue LED tube connected to GPIO25 (via MOSFET/relay)"
echo ""
echo "🚀 Ready to test! Run as pi user:"
echo "   python3 led_control_system.py"
echo ""
echo "📋 Quick Test Commands:"
echo "   # Test blue LED only:"
echo "   python3 -c \"import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BCM); GPIO.setup(25, GPIO.OUT); GPIO.output(25, GPIO.HIGH); input('Press Enter to turn off...'); GPIO.output(25, GPIO.LOW); GPIO.cleanup()\""
echo ""
echo "   # Test NeoPixel only:"
echo "   python3 -c \"import board; import neopixel; pixels = neopixel.NeoPixel(board.D12, 24, brightness=0.3); pixels.fill((255, 0, 0)); input('Press Enter to turn off...'); pixels.fill((0, 0, 0))\""
