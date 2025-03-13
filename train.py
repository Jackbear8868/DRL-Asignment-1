import numpy as np
import pickle
import random
import gym
import simple_custom_taxi_env

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        """
        Q-learning Agent with a dictionary-based Q-table.
        - Uses epsilon-greedy strategy.
        - Decays epsilon over time to shift from exploration to exploitation.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.q_table = {}  # Dictionary-based Q-table

    def get_state_key(self, state):
        """
        Convert the state into a hashable key (tuple) for Q-table lookup.
        """
        return tuple(state)  # Convert to tuple for dictionary key compatibility

    def get_action(self, state):
        """
        Select an action using the epsilon-greedy policy.
        """
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:  # Explore
            return random.randint(0, self.action_size - 1)
        
        # Exploit known best action
        return np.argmax(self.q_table.get(state_key, np.zeros(self.action_size)))

    def update_q_table(self, state, action, reward, next_state, done):
        """
        Apply Q-learning update rule.
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Initialize Q-values if not already in the table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        # Q-learning formula
        best_next_action = np.argmax(self.q_table[next_state_key])
        target = reward + (self.gamma * self.q_table[next_state_key][best_next_action] * (1 - int(done)))
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action])

    def decay_epsilon(self):
        """
        Reduce exploration rate (epsilon) over time.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_q_learning(env, agent, episodes=5000, max_steps=200):
    """
    Train a Q-learning agent and return the trained Q-table.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            # Update Q-table
            agent.update_q_table(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            if done:
                break

        # Decay exploration rate
        agent.decay_epsilon()

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    return agent.q_table


# Train the agent
env = simple_custom_taxi_env.SimpleTaxiEnv(grid_size=5, fuel_limit=50)  # Make sure you have this class defined
action_size = 6  # 6 possible actions
agent = QLearningAgent(state_size=(env.grid_size, env.grid_size, 2), action_size=action_size)
q_table = train_q_learning(env, agent, episodes=5000)

# Save the trained Q-table to a file
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete. Q-table saved as trained_q_table.pkl")
