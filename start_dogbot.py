#!/usr/bin/env python3
# Save this as: /home/morgan/dogbot/start_dogbot.py

# Import both the main DogBot and the GUI
from main import DogBotAI
from dogbot_gui_monitor import DogBotGUI

# Create the DogBot brain
dogbot = DogBotAI()

# Start the robot
dogbot.start()

# Create and start the GUI, giving it access to the robot
gui = DogBotGUI(dogbot_instance=dogbot)
gui.run()