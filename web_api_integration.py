from flask import Flask, jsonify, request
from mission_system import MissionController
import threading

app = Flask(__name__)
mission = MissionController()

@app.route('/api/mission/start', methods=['POST'])
def start_mission():
    success = mission.start_mission()
    return jsonify({'success': success, 'status': mission.get_mission_status()})

@app.route('/api/mission/stop', methods=['POST'])
def stop_mission():
    mission.stop_mission()
    return jsonify({'success': True, 'status': mission.get_mission_status()})

@app.route('/api/mission/status', methods=['GET'])
def mission_status():
    return jsonify(mission.get_mission_status())

@app.route('/api/dispense_treat', methods=['POST'])
def manual_treat():
    mission.dispense_treat({'behavior': 'manual', 'source': 'web_interface'})
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)