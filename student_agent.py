import torch
import torch.nn as nn
import numpy as np
import random

# Import the QNetwork definition from your training script
from train_DQN import QNetwork  # Ensure QNetwork is in the same directory

# Load the trained model once at the start
MODEL_PATH = "./dqn_taxi.pth"
STATE_SIZE = 21  # Ensure this matches your environment
ACTION_SIZE = 6

# Initialize device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Set to evaluation mode (no gradient updates)

passenger_pick = 0
passenger_row = -1
passenger_col = -1
destination_row = -1
destination_col = -1

def get_action(obs):
    """ Uses trained DQN model to choose an action. """
    
    obs = obs + (passenger_row, passenger_col, destination_row, destination_col, passenger_pick)
            
    # Convert observation to tensor

    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # Add batch dimension
    
    # Get Q-values from the model
    with torch.no_grad():
        q_values = model(state_tensor)  # Output shape: [1, ACTION_SIZE]

    # Choose the action with the highest Q-value
    action = torch.argmax(q_values, dim=1).item()

    if passenger_row == -1:
        if (obs[0], obs[1]) in [(obs[2],obs[3]),(obs[4],obs[5]),(obs[6],obs[7]),(obs[8],obs[9])] and obs[14]:
            passenger_row = obs[0]
            passenger_col = obs[1]
    
    if destination_row == -1:
        if (obs[0], obs[1]) in [(obs[2],obs[3]),(obs[4],obs[5]),(obs[6],obs[7]),(obs[8],obs[9])] and obs[15]:
            destination_row = obs[0]
            destination_col = obs[1]

    if obs[0] == passenger_row and obs[1] == passenger_col:
        if action == 4:
            passenger_pick = 1

    if action == 5 and (obs[0] != destination_row or obs[1] != destination_col):
        passenger_pick = 0
        passenger_row = obs[0]
        passenger_col = obs[1]

    return action  # Return the best action

