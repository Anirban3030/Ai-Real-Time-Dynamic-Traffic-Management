import numpy as np
import random
from vehicle_detection import process_images_in_directory, model

# Constants
lanes = 4
time_per_cycle = 120  # Total time per cycle (in seconds)
Max_green_time = 60

# Q-learning parameters
Q_table = np.zeros((5000, lanes))  # 5000 possible states, 4 lanes/actions
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate

IMAGES_DIR = r"images"

# State representation helper
def encode_state(results):
    
    state = []
    for result in results:
        state.append(result['total_vehicles'])
        state.append(result['emergency_vehicles'])
    return hash(tuple(state)) % 5000  

# Reward calculation helper
def calculate_reward(results, time_allocation):

    congestion = sum(result['total_vehicles'] for result in results)
    emergency_handled = sum(
        result['emergency_vehicles'] > 0 and time_allocation[i] > 0
        for i, result in enumerate(results)
    )
    reward = -congestion + 50 * emergency_handled
    return reward

# Epsilon-greedy action selection
def get_action(state):
    
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, lanes - 1)  
    else:
        return np.argmax(Q_table[state])  

# Q-learning Update Function
def q_learning_update(state, action, reward, next_state):
    
    max_future_q = np.max(Q_table[next_state])  
    current_q = Q_table[state, action]  # Current Q-value
    Q_table[state, action] = current_q + alpha * (reward + gamma * max_future_q - current_q)

# Allocate green signal time using Q-learning
def allocate_time(results, state):
    
    action = get_action(state)
    time_allocation = [0] * lanes
    time_allocation[action] = min(Max_green_time, time_per_cycle)  # Priority for selected lane

    
    remaining_time = time_per_cycle - time_allocation[action]
    for lane, result in enumerate(results):
        if lane != action:
            time_allocation[lane] = min(result['total_vehicles'] * 2, remaining_time, Max_green_time)
            remaining_time -= time_allocation[lane]

    return action, time_allocation

# Calculate priorities
def calculate_priorities(results):
    
    priorities = []
    for lane, result in enumerate(results):
        priority = (
            result['emergency_vehicles'],  # Higher priority for emergencies
            result['accident_detected'],  # Second-highest for accidents
            result['total_vehicles']      # Then by total vehicle count
        )
        priorities.append((lane, priority))

    # Sort by priority
    priorities.sort(key=lambda x: (-x[1][0], -x[1][1], -x[1][2]))
    return priorities

# Simulate traffic lights
def simulate_traffic_light(results, time_allocation, priorities):
    
    print("\nTraffic Light Simulation with Q-learning:\n")
    for rank, (lane, priority) in enumerate(priorities, 1):
        result = results[lane]
        print(f"Priority {rank}: Lane {lane + 1} ({result['image_name']})")
        print(f" Total Vehicles: {result['total_vehicles']}")
        print(f" Emergency Vehicles: {result['emergency_vehicles']}")
        print(f" Accident Detected: {'Yes' if result['accident_detected'] else 'No'}")
        print(f" Green Time: {time_allocation[lane]} seconds\n")

# Main function to run the system
if __name__ == "__main__":
    for episode in range(5):  
        print(f"\n--- Episode {episode + 1} ---\n")
        results = process_images_in_directory(IMAGES_DIR, model)
        current_state = encode_state(results)
        action, time_allocation = allocate_time(results, current_state)

        priorities = calculate_priorities(results)
        simulate_traffic_light(results, time_allocation, priorities)
    
        reward = calculate_reward(results, time_allocation)
        next_state = encode_state(results)  
        q_learning_update(current_state, action, reward, next_state)
        np.save("Q_table.npy", Q_table)