#!/bin/bash

echo "================================"
echo "EMERGENCY MOTOR STOP"
echo "================================"

# Kill any Xbox controller processes
echo "Stopping Xbox controllers..."
pkill -f "xbox.*controller" 2>/dev/null
pkill -f "python.*xbox" 2>/dev/null

# Kill all gpioset processes (stops PWM)
echo "Killing gpioset processes..."
killall gpioset 2>/dev/null

# Force all motor pins to 0
echo "Clearing motor pins..."
gpioset gpiochip0 17=0 27=0 22=0 23=0 24=0 25=0 2>/dev/null

echo "✅ Motors stopped"
echo "================================"