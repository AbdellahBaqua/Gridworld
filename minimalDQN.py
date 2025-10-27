import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================
# Simple Grid Environment
# (No changes needed here)
# ==========================
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.agent = list(self.start)
        return self._get_state()

    def _get_state(self):
        # One-hot encoding of agent position
        state = np.zeros(self.size * self.size)
        idx = self.agent[0] * self.size + self.agent[1]
        state[idx] = 1.0
        return state

    def step(self, action):
        # 0=up, 1=down, 2=left, 3=right
        x, y = self.agent
        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(self.size - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(self.size - 1, y + 1)
        self.agent = [x, y]

        done = (tuple(self.agent) == self.goal)
        reward = 10 if done else -1
        return self._get_state(), reward, done


# ==========================
# Q-Network
# (No changes needed here)
# ==========================
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==========================
# DQN Training Loop (Fixed)
# ==========================
# ==========================
# DQN Training Loop (Fixed)
# ==========================
def train_dqn(episodes=300, size=5, gamma=0.9, lr=0.01):
    env = GridWorld(size)
    state_dim = size * size
    action_dim = 4

    q_net = QNetwork(state_dim, action_dim)
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    eps_start, eps_end = 1.0, 0.1
    eps_decay = (eps_start - eps_end) / episodes

    rewards_per_episode = []

    for ep in range(episodes):
        # --- FIX 1: Explicitly set dtype to torch.float for state tensor ---
        # The previous line: state = torch.FloatTensor(env.reset()) is replaced
        state = torch.tensor(env.reset(), dtype=torch.float) 
        
        epsilon = max(eps_end, eps_start - ep * eps_decay)
        done = False
        total_reward = 0

        while not done:
            # ε-greedy action
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    action = q_net(state).argmax().item()

            # Step
            next_state_np, reward, done = env.step(action)
            
            # --- FIX 2: Explicitly set dtype to torch.float for next_state ---
            next_state = torch.tensor(next_state_np, dtype=torch.float)

            # Target calculation (requires a second, minor fix for tensor creation)
            with torch.no_grad():
                # Get the max Q value for the next state
                next_q_max = 0
                if not done:
                    # Get the max Q from the network, convert to a Python float
                    next_q_max = q_net(next_state).max().item()
                
                # Calculate the TD Target (a Python float)
                target_q = reward + gamma * next_q_max
                
                # Convert the scalar target_q back to a *float* tensor 
                # (and ensure it's a 0-dim tensor for MSELoss)
                target = torch.tensor(target_q, dtype=torch.float)


            # Compute loss
            # Q-values for current state
            q_values = q_net(state)
            # The predicted Q-value for the action taken
            pred = q_values[action] 
            
            # Use the target tensor constructed above
            loss = loss_fn(pred, target)

            # Optimize
            optimizer.zero_grad()
            loss.backward() # This is where the error originally occurred
            optimizer.step()

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        print(f"Episode {ep+1}/{episodes}, Reward: {total_reward}, Eps: {epsilon:.2f}")

    return rewards_per_episode


# ==========================
# Run & Plot
# ==========================
if __name__ == "__main__":
    # Ensure reproducibility for debugging
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Note: Reduced episodes for faster execution in a simple example
    rewards = train_dqn(episodes=200, size=5) 
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Training Rewards per Episode (Simple DQN)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()