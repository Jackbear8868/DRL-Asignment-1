import torch
import torch.nn as nn
import numpy as np
import random

# Import the QNetwork definition from your training script
from train_DQN import QNetwork  # 確保 QNetwork 在同一資料夾

# Load the trained model once at the start
MODEL_PATH = "./dqn_taxi.pth"
STATE_SIZE = 21  # 你在訓練時是 21 維
ACTION_SIZE = 6

# Initialize device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Set to evaluation mode (no gradient updates)

# ★ 全域變數，追蹤乘客/目的地資訊
passenger_pick = 0
passenger_row = -1
passenger_col = -1
destination_row = -1
destination_col = -1

def get_action(obs):
    """
    Uses trained DQN model to choose an action.

    obs: 16 維 (包含 taxi_row, taxi_col, station座標, obstacle資訊, passenger_look, destination_look)
    會額外拼上 (passenger_row, passenger_col, destination_row, destination_col, passenger_pick) => 21 維
    """
    # 告訴 Python，以下這些變數要用「外部的全域變數」而非函式區域變數
    global passenger_pick, passenger_row, passenger_col, destination_row, destination_col

    # 1) 將 16 維 obs 與 5 維乘客資訊拼在一起 => 21 維
    extended_obs = obs + (passenger_row, passenger_col, destination_row, destination_col, passenger_pick)

    # 2) 轉為 Tensor，餵進已載入的模型
    if not isinstance(extended_obs, np.ndarray):
        extended_obs = np.array(extended_obs)
    
    state_tensor = torch.FloatTensor(extended_obs).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)  # shape: [1, ACTION_SIZE]
    action = torch.argmax(q_values, dim=1).item()

    # 3) 根據 obs 與 action，更新全域變數 (乘客 / 目的地 / 是否載客)
    #    注意：這些邏輯其實更適合放在 env.step()，這裡為了示範盡量簡化

    # 若還沒確定 passenger_row/col，且 obs[14] = passenger_look == True，表示現在taxi位置有乘客
    if passenger_row == -1:
        if (obs[0], obs[1]) in [
            (obs[2], obs[3]),
            (obs[4], obs[5]),
            (obs[6], obs[7]),
            (obs[8], obs[9])
        ] and obs[14] == 1:
            passenger_row = obs[0]
            passenger_col = obs[1]
    
    # 同理，若還沒確定 destination_row/col，且 obs[15] = destination_look == True，表示這裡是目的地
    if destination_row == -1:
        if (obs[0], obs[1]) in [
            (obs[2], obs[3]),
            (obs[4], obs[5]),
            (obs[6], obs[7]),
            (obs[8], obs[9])
        ] and obs[15] == 1:
            destination_row = obs[0]
            destination_col = obs[1]

    # 若 taxi 與 passenger 在同格，執行 PICKUP(=4) => 乘客上車
    if obs[0] == passenger_row and obs[1] == passenger_col:
        if action == 4:  # PICKUP
            passenger_pick = 1

    # 若執行 DROPOFF(=5)，且不是在正確目的地 => 乘客下車(失敗) => 再度更新乘客座標為 taxi 位置
    if action == 5 and not (obs[0] == destination_row and obs[1] == destination_col):
        passenger_pick = 0
        passenger_row = obs[0]
        passenger_col = obs[1]

    return action
