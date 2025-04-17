from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
import numpy as np
import os
from vehicle_detection import process_images_in_directory, model
from simulation import (
    Q_table, 
    encode_state, 
    allocate_time, 
    calculate_priorities, 
    simulate_traffic_light, 
    calculate_reward, 
    q_learning_update
)

# Flask app initialization
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
IMAGES_DIR = r"images"  # Directory containing input images
OUTPUT_DIR = r"output"  # Directory to save processed images
Q_TABLE_PATH = "Q_table.npy"

# Load Q-table or initialize if it doesn't exist
if os.path.exists(Q_TABLE_PATH):
    Q_table = np.load(Q_TABLE_PATH)
else:
    Q_table = np.zeros((5000, 4))  # 5000 possible states, 4 actions/lanes

# Endpoint: Process images and get detection results
@app.route('/api/process_images', methods=['GET'])
def process_images():
    try:
        results = process_images_in_directory(IMAGES_DIR, model)
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Endpoint: Run simulation and get time allocation
@app.route('/api/run_simulation', methods=['POST'])
def run_simulation():
    try:
        # Process images to get detection results
        results = process_images_in_directory(IMAGES_DIR, model)
        
        # Encode current state
        current_state = encode_state(results)
        
        # Allocate time for lanes
        action, time_allocation = allocate_time(results, current_state)
        
        # Calculate priorities for lanes
        priorities = calculate_priorities(results)
        
        # Generate reward and update Q-table
        reward = calculate_reward(results, time_allocation)
        next_state = encode_state(results)
        q_learning_update(current_state, action, reward, next_state)
        
        # Save updated Q-table
        np.save(Q_TABLE_PATH, Q_table)

        # Prepare simulation response
        simulation_response = []
        for rank, (lane, priority) in enumerate(priorities, 1):
            result = results[lane]
            simulation_response.append({
                "priority": rank,
                "lane": lane + 1,
                "image_name": result['image_name'],
                "total_vehicles": result['total_vehicles'],
                "emergency_vehicles": result['emergency_vehicles'],
                "accident_detected": result['accident_detected'],
                "green_time": time_allocation[lane]
            })

        return jsonify({"status": "success", "simulation": simulation_response})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Endpoint: Serve processed image
@app.route('/api/get_image/<filename>', methods=['GET'])
def get_image(filename):
    try:
        return send_from_directory(OUTPUT_DIR, filename)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Endpoint: Fetch Q-table for debugging or visualization
@app.route('/api/q_table', methods=['GET'])
def get_q_table():
    try:
        return jsonify({"status": "success", "q_table": Q_table.tolist()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
