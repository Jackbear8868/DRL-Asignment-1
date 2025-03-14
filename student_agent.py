import torch
import torch.nn as nn
import numpy as np
import random

# Import the QNetwork definition from your training script
from train_DQN import QNetwork  # Ensure QNetwork is in the same directory

# Load the trained model once at the start
MODEL_PATH = "dqn_taxi.pth"
STATE_SIZE = 16  # Ensure this matches your environment
ACTION_SIZE = 6

# Initialize device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Set to evaluation mode (no gradient updates)

def get_action(obs):
    """ Uses trained DQN model to choose an action. """
    
    # Convert observation to tensor
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # Add batch dimension
    
    # Get Q-values from the model
    with torch.no_grad():
        q_values = model(state_tensor)  # Output shape: [1, ACTION_SIZE]

    # Choose the action with the highest Q-value
    action = torch.argmax(q_values, dim=1).item()

    return action  # Return the best action
