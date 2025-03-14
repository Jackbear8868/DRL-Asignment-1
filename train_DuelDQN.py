import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple

# 若你的環境程式碼是放在同一個檔案，可以直接 import SimpleTaxiEnv；
# 否則請確保 SimpleTaxiEnv 類別跟以下代碼在同一個命名空間可以引用。
from taxi_env import TaxiEnv  # 假設環境檔案名稱是 taxi_env.py

# -----------------------------
# 1) Dueling Q-Network
# -----------------------------
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DuelingQNetwork, self).__init__()
        # 共享前兩層 (Feature Extractor)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # 分成兩個分支：一個輸出 Value，另一個輸出 Advantage
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)         # 輸出一個 scalar，對應 V(s)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size) # 輸出對應各動作的 A(s,a)
        )

        # 權重初始化，可自行調整策略
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        for layer in self.value_stream:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.advantage_stream:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # 先通過共享層
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 分別取得 Value 與 Advantage
        value = self.value_stream(x)         # shape: [batch_size, 1]
        advantage = self.advantage_stream(x) # shape: [batch_size, action_size]

        # Q(s,a) = V(s) + [A(s,a) - mean(A(s,a'))]
        advantage_mean = torch.mean(advantage, dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values


# -----------------------------
# 2) 建立 DQN Agent
# -----------------------------
class DQNAgent:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_size: int = 64,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 batch_size: int = 64,
                 max_memory_size: int = 10000,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 update_target_freq: int = 1000):
        """
        DQN Agent 初始化。
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_target_freq = update_target_freq

        # 1) 指定裝置 (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2) 建立 Q-network & target Q-network 並放到 device
        self.q_network = DuelingQNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DuelingQNetwork(state_size, action_size, hidden_size).to(self.device)
        # 同步參數
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 3) 建立優化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 4) 建立經驗回放
        self.memory = deque(maxlen=max_memory_size)

        # 5) 訓練步數計數，用於決定何時更新 target network
        self.train_step = 0

    def remember(self, state, action, reward, next_state, done):
        """ 將一次交互紀錄存入 Replay Buffer """
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, test_mode=False):
        """
        根據當前策略選取動作 (epsilon-greedy)。
        如果是測試模式 (test_mode=True)，則直接使用最優動作。
        """
        # 確保 state 是 numpy array，並轉到 self.device
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 若在測試階段或隨機數大於 epsilon，選擇網路給出的最大 Q 值動作
        if test_mode or random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        else:
            # 否則隨機探索
            action = random.randrange(self.action_size)
        
        return action

    def train_step_update(self):
        """
        從 Replay Buffer 抽樣一個批次，並執行一次 DQN 的更新。
        """
        if len(self.memory) < self.batch_size:
            return  # 不足一個批次就不訓練

        # 1) 從記憶體中隨機取樣一個 batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 2) 轉成 tensor 並放到同一個 device
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # ---------------------------
        # 3) 計算 Q(s,a) 與目標值
        # ---------------------------
        # Q(s,a) = Q_network(states)[range(batch_size), actions]
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 目標 = r + gamma * max(Q_target(next_state)) (若 done=1，則不加未來值)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q, _ = torch.max(next_q_values, dim=1)
            q_target = rewards + (1 - dones) * self.gamma * max_next_q

        # ---------------------------
        # 4) 計算損失 + 反向傳播
        # ---------------------------
        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1

        # 5) 隨機梯度下降 steps 累積到一定次數後，更新 target_network
        if self.train_step % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 6) 逐漸減少 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)


# -----------------------------
# 3) 主要訓練流程
# -----------------------------
def train_dqn(episodes: int = 2000):
    """
    使用 DQN 訓練 TaxiEnv。
    :param episodes: 總訓練回合數
    :return: 已訓練好的 agent
    """
    # 環境初始化，fuel_limit 可以開大一點，不然太容易沒油結束
    env = TaxiEnv(fuel_limit=5000)

    # 確認 state_size 與 action_size
    state_size = 16  # e.g., 依照 your TaxiEnv get_state() 回傳的長度
    action_size = 6  # e.g., [下、上、右、左、PICKUP、DROPOFF]

    # 建立 agent（如果需要更深層，hidden_size 可加大，例如 128）
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=128,     # 改大隱藏層
        lr=1e-3,             # 學習率
        gamma=0.99,
        batch_size=64,
        max_memory_size=10000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995, # 調整探索衰減速度
        update_target_freq=1000
    )

    scores = []
    for e in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.get_action(state, test_mode=False)
            next_state, reward, done, _ = env.step(action)

            # 將此次經驗存入記憶體
            agent.remember(state, action, reward, next_state, done)
            # 訓練一次
            agent.train_step_update()

            state = next_state
            total_reward += reward
            step_count += 1

        scores.append(total_reward)

        # 印出訓練狀況
        print(f"Episode {e} | Epsilon: {agent.epsilon:.3f} | Reward: {total_reward:.2f} | Steps: {step_count}")

        if (e + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"[Checkpoint] Episode: {e+1}/{episodes}, Avg Score (last 100): {avg_score:.2f}")

    # 儲存訓練後的模型
    torch.save(agent.q_network.state_dict(), "dqn_taxi.pth")
    print("Training finished. Model saved to dqn_taxi.pth.")

    return agent


# -----------------------------
# 4) 測試函式 (可選)
# -----------------------------
def test_dqn(agent: DQNAgent, episodes: int = 10):
    """
    使用訓練好的 agent 測試環境
    """
    env = TaxiEnv(fuel_limit=5000) 
    total_scores = []
    for e in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            # 以測試模式 (test_mode=True) 選取動作
            action = agent.get_action(state, test_mode=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        total_scores.append(total_reward)
        print(f"Test Episode {e+1}: score = {total_reward:.2f}")
    print(f"Average score over {episodes} test episodes: {np.mean(total_scores):.2f}")


# -----------------------------
# 5) 主程式入口
# -----------------------------
if __name__ == "__main__":
    # 進行訓練
    trained_agent = train_dqn(episodes=20)

    # 測試
    test_agent = DQNAgent(state_size=16, action_size=6, hidden_size=128)
    test_agent.q_network.load_state_dict(torch.load("dqn_taxi.pth"))
    test_agent.target_network.load_state_dict(test_agent.q_network.state_dict())

    test_dqn(test_agent, episodes=5)
