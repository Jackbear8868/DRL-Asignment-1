# student_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple

# 若你的環境程式碼是放在同一個檔案，可以直接 import SimpleTaxiEnv；
# 否則請確保 SimpleTaxiEnv 類別跟以下代碼在同一個命名空間可以引用。
from taxi_env import TaxiEnv  # 假設環境程式檔案名稱是 SimpleTaxiEnv.py

# -----------------------------
# 1) 建立 Q 網路 (DNN 模型)
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(QNetwork, self).__init__()
        # 這裡示範一個簡單的三層全連接網路
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        # 初始化權重（可自行嘗試不同初始化策略）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

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

        Args:
            state_size (int): 狀態維度（本例是 16）
            action_size (int): 行動空間大小（本例是 6）
            hidden_size (int): 隱藏層大小
            lr (float): 學習率
            gamma (float): 折扣因子
            batch_size (int): 批次大小
            max_memory_size (int): Replay buffer 最大容量
            epsilon_start (float): 起始 epsilon 值
            epsilon_end (float): 最小 epsilon 值
            epsilon_decay (float): epsilon 衰減率
            update_target_freq (int): 多少次訓練後更新一次 target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_target_freq = update_target_freq

        # 建立 Q-network 和 target Q-network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 優化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 建立經驗回放
        self.memory = deque(maxlen=max_memory_size)

        # 訓練步數計數，用於決定何時更新 target network
        self.train_step = 0

    def remember(self, state, action, reward, next_state, done):
        """ 將一次交互紀錄存入 Replay Buffer """
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, test_mode=False):
        """
        根據當前策略選取動作 (epsilon-greedy)。
        如果是測試模式 (test_mode=True)，則直接使用最優動作。
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 如果在測試階段，或隨機數大於 epsilon，就直接取 q-network 的最優動作
        if test_mode or random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        else:
            # 否則隨機選擇
            action = random.randrange(self.action_size)
        
        return action

    def train_step_update(self):
        """
        從 Replay Buffer 抽樣一個批次，並執行一次 DQN 的更新。
        """
        if len(self.memory) < self.batch_size:
            return  # 不足一個批次就不訓練
        
        # 從記憶體中隨機取樣一個 batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # ---------------------------
        # 計算 Q(s,a) 與目標值
        # ---------------------------
        # Q(s,a) = Q_network(states)[range(batch_size), actions]
        q_values      = self.q_network(states)
        q_values      = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q目標 = r + gamma * max(Q_target(next_state)) （若 done=1，則不加未來值）
        with torch.no_grad():
            # Q_target(next_state) = target_network(next_state).max(1)
            next_q_values = self.target_network(next_states)
            max_next_q, _ = torch.max(next_q_values, dim=1)
            q_target = rewards + (1 - dones) * self.gamma * max_next_q

        # ---------------------------
        # 計算損失 + 反向傳播
        # ---------------------------
        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        # 隨機梯度下降 steps 已經累積到一定次數後，更新 target_network
        if self.train_step % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 逐漸減少 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

# -----------------------------
# 3) 主要訓練流程
# -----------------------------
def train_dqn(episodes: int = 2000):
    """
    使用 DQN 訓練 SimpleTaxiEnv。
    :param episodes: 總訓練回合數
    :return: 已訓練好的 agent
    """
    # 環境初始化，fuel_limit 可以開大一點，不然太容易沒油結束
    env = TaxiEnv(fuel_limit=5000)  

    # state_size: 從env.get_state()可知每次回傳 16 個特徵
    state_size = 16
    action_size = 6  # 共有 6 種動作: 下、上、右、左、PICKUP、DROPOFF

    # 建立 agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=64,    # 你可以自行調整
        lr=1e-3,           # 學習率
        gamma=0.99,
        batch_size=64,
        max_memory_size=10000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99,
        update_target_freq=1000
    )

    # 開始訓練
    scores = []
    for e in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action = agent.get_action(state, test_mode=False)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.train_step_update()

            state = next_state
            total_reward += reward
            step_count += 1
        
        scores.append(total_reward)
        print(e, agent.epsilon, total_reward)
        # 每隔若干回合印出進度
        if (e + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {e+1}/{episodes}, Average Score (last 100): {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

    # 訓練完成後儲存 Q 網路
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
        print(f"Test Episode {e+1}: score = {total_reward}")
    print(f"Average score over {episodes} test episodes: {np.mean(total_scores):.2f}")

# -----------------------------
# 5) 主程式入口
# -----------------------------
if __name__ == "__main__":
    # 進行訓練
    trained_agent = train_dqn(episodes=20)

    # 若要測試：可以讀取剛剛的模型並進行測試
    # 1) 先新建同樣架構的 agent
    test_agent = DQNAgent(state_size=16, action_size=6)
    # 2) 載入模型權重
    test_agent.q_network.load_state_dict(torch.load("dqn_taxi.pth"))
    test_agent.target_network.load_state_dict(test_agent.q_network.state_dict())

    # 備註：若要串接原先的 run_agent 函式 (例如 run_agent("student_agent.py", env_config)),
    # 需要在 get_action() 或主檔案裡加入呼叫此 test_agent 的邏輯，
    # 並確保 run_agent 中的 get_action 會呼叫同樣的函式。
