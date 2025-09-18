#!/bin/bash
cd ~/dogbot
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install opencv-python mediapipe pyserial soundfile vosk adafruit-circuitpython-servokit
echo 'source ~/dogbot/env/bin/activate' >> ~/.bashrc
echo "✅ Virtual environment setup complete and auto-enabled."