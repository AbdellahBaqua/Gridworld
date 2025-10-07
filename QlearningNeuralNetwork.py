import numpy as np
import matplotlib.pyplot as plt
import random
import time
import matplotlib.patches as patches
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# NEURAL NETWORK Q-FUNCTION APPROXIMATOR
# ==========================================
class QNetwork(nn.Module):
    """Neural network for approximating Q-values."""
    def __init__(self, state_size, action_size, hidden_sizes=[128, 64]):
        super(QNetwork, self).__init__()
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for training stability."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# ==========================================
# GRIDWORLD ENVIRONMENT (Same as before)
# ==========================================
class GridWorld:
    """A deterministic GridWorld environment for RL testing with a randomized goal."""
    def __init__(self, size=5, fixed_obstacles=3):
        self.size = size
        self.action_space = 4
        self.fixed_obstacles = fixed_obstacles
        self.start_pos = (0, 0)
        self.goal_pos = None
        self.obstacles = []
        self.reset()

    def _select_random_goal(self):
        possible_coords = [(r, c) for r in range(self.size) for c in range(self.size)]
        coords_to_exclude = {self.start_pos}
        if self.size > 2:
            coords_to_exclude.add((0, 1))
            coords_to_exclude.add((1, 0))
            if self.size > 3:
                coords_to_exclude.add((1, 1))

        valid_goals = [coord for coord in possible_coords if coord not in coords_to_exclude]
        potential_goals = [(r, c) for r, c in valid_goals if r >= self.size // 2 or c >= self.size // 2]
        
        if not potential_goals:
            return random.choice(valid_goals)
        return random.choice(potential_goals)

    def _generate_obstacles(self):
        if self.size < 4 or self.fixed_obstacles == 0:
            return []
        
        possible_coords = [(r, c) for r in range(self.size) for c in range(self.size)]
        coords_to_exclude = {self.start_pos, self.goal_pos}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx_s, ny_s = self.start_pos[0] + dx, self.start_pos[1] + dy
            if 0 <= nx_s < self.size and 0 <= ny_s < self.size:
                coords_to_exclude.add((nx_s, ny_s))
            nx_g, ny_g = self.goal_pos[0] + dx, self.goal_pos[1] + dy
            if 0 <= nx_g < self.size and 0 <= ny_g < self.size:
                coords_to_exclude.add((nx_g, ny_g))

        valid_obstacle_coords = [coord for coord in possible_coords if coord not in coords_to_exclude]
        max_possible_obs = len(valid_obstacle_coords) // 2
        num_obs = min(self.fixed_obstacles, max_possible_obs)
        
        if num_obs == 0 or not valid_obstacle_coords:
            return []
        obs = random.sample(valid_obstacle_coords, num_obs)
        return [list(o) for o in obs]

    def reset(self):
        self.goal_pos = self._select_random_goal()
        self.obstacles = self._generate_obstacles()
        self.agent_pos = list(self.start_pos)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        """Returns normalized state representation for neural network."""
        state = np.zeros(self.size * self.size + 4)
        # Agent position (one-hot encoded)
        idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[idx] = 1.0
        # Goal position (normalized)
        state[-4] = self.goal_pos[0] / self.size
        state[-3] = self.goal_pos[1] / self.size
        # Distance to goal (normalized)
        dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        state[-2] = dist / (2 * self.size)
        # Steps normalized
        state[-1] = self.steps / (self.size * self.size * 4)
        return state

    def _next_pos(self, pos, action):
        x, y = pos
        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1
        
        new_x = max(0, min(x, self.size - 1))
        new_y = max(0, min(y, self.size - 1))
        new_pos = [new_x, new_y]
        
        if new_pos in self.obstacles:
            return pos 
        return new_pos

    def step(self, action):
        self.steps += 1
        new_pos = self._next_pos(self.agent_pos, action)
        self.agent_pos = new_pos
        
        reward = -1
        done = False
        
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 10
            done = True
        
        if self.steps >= self.size * self.size * 4:
            done = True
            
        return self._get_state(), reward, done


# ==========================================
# DEEP Q-LEARNING CORE
# ==========================================
class DQNAgent:
    """Deep Q-Network agent with experience replay."""
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        self.loss_history = []
        self.q_value_history = []
        
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def get_q_values(self, state):
        """Get Q-values for a state for visualization."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_tensor).cpu().numpy()[0]
    
    def train_step(self, batch_size, gamma):
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        self.q_value_history.append(current_q_values.mean().item())
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


def run_dqn_episode(env, agent, epsilon, gamma, batch_size, train=True):
    """Runs a single DQN episode."""
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        
        if train:
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step(batch_size, gamma)
        
        state = next_state
        total_reward += reward
        steps += 1
        
    return total_reward, steps


# ==========================================
# METRICS TRACKING
# ==========================================
class MetricsTracker:
    """Tracks and visualizes training metrics."""
    def __init__(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []
        self.losses = []
        self.avg_q_values = []
        self.epsilon_values = []
        
    def update(self, reward, steps, success, loss, avg_q, epsilon):
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.success_rate.append(1 if success else 0)
        if loss is not None:
            self.losses.append(loss)
        if avg_q is not None:
            self.avg_q_values.append(avg_q)
        self.epsilon_values.append(epsilon)
    
    def get_success_rate(self, window=100):
        if len(self.success_rate) < window:
            return np.mean(self.success_rate) if self.success_rate else 0
        return np.mean(self.success_rate[-window:])
    
    def plot_metrics(self, grid_size):
        """Plot comprehensive training metrics."""
        plt.close('all')
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Rewards
        window = max(1, len(self.episode_rewards) // 20)
        smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label="Par épisode")
        axes[0, 0].plot(np.arange(window - 1, len(self.episode_rewards)), smoothed, 
                       color='red', label="Moyenne glissante")
        axes[0, 0].set_title("Récompenses par Épisode")
        axes[0, 0].set_xlabel("Épisode")
        axes[0, 0].set_ylabel("Récompense")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Steps per episode
        smoothed_steps = np.convolve(self.episode_steps, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(self.episode_steps, alpha=0.3, label="Par épisode")
        axes[0, 1].plot(np.arange(window - 1, len(self.episode_steps)), smoothed_steps,
                       color='green', label="Moyenne glissante")
        axes[0, 1].set_title("Étapes par Épisode")
        axes[0, 1].set_xlabel("Épisode")
        axes[0, 1].set_ylabel("Étapes")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate
        window_sr = min(100, len(self.success_rate))
        success_rate_smooth = [np.mean(self.success_rate[max(0, i-window_sr):i+1]) 
                              for i in range(len(self.success_rate))]
        axes[0, 2].plot(success_rate_smooth, color='purple', linewidth=2)
        axes[0, 2].set_title("Taux de Réussite (Fenêtre: 100)")
        axes[0, 2].set_xlabel("Épisode")
        axes[0, 2].set_ylabel("Taux de Réussite")
        axes[0, 2].set_ylim([0, 1.1])
        axes[0, 2].grid(True, alpha=0.3)
        
        # Loss
        if self.losses:
            axes[1, 0].plot(self.losses, color='orange', alpha=0.6)
            axes[1, 0].set_title("Perte d'Entraînement (MSE)")
            axes[1, 0].set_xlabel("Étape d'Entraînement")
            axes[1, 0].set_ylabel("Perte")
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Average Q-values
        if self.avg_q_values:
            axes[1, 1].plot(self.avg_q_values, color='teal', alpha=0.7)
            axes[1, 1].set_title("Valeurs Q Moyennes")
            axes[1, 1].set_xlabel("Étape d'Entraînement")
            axes[1, 1].set_ylabel("Q Moyen")
            axes[1, 1].grid(True, alpha=0.3)
        
        # Epsilon decay
        axes[1, 2].plot(self.epsilon_values, color='brown', linewidth=2)
        axes[1, 2].set_title("Décroissance Epsilon (Exploration)")
        axes[1, 2].set_xlabel("Épisode")
        axes[1, 2].set_ylabel("Epsilon")
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f"Métriques DQN - Grille {grid_size}x{grid_size}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(10.0)


# ==========================================
# VISUALIZATION FUNCTIONS (ADAPTED FOR DQN)
# ==========================================
def visualize_policy_and_animate_dqn(agent, env, title_prefix, metrics, speed=0.3):
    """Visualizes learned policy and animates agent using neural network."""
    size = env.size

    # Copy environment state (keep same goal/obstacles)
    temp_env = GridWorld(size=size, fixed_obstacles=env.fixed_obstacles)
    temp_env.goal_pos = env.goal_pos
    temp_env.obstacles = [list(o) for o in env.obstacles]
    temp_env.start_pos = env.start_pos
    temp_env.agent_pos = list(env.start_pos)
    temp_env.steps = 0

    if size > 10 and speed > 0.0:
        print(f"Skipping interactive animation for large grid {size}x{size}.")
        return

    plt.close('all')
    fig, ax = plt.subplots(figsize=(size, size))

    # Q-value heatmap
    max_Q = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if [i, j] in temp_env.obstacles:
                max_Q[i, j] = np.nan  # obstacle
            else:
                temp_env.agent_pos = [i, j]
                state = temp_env._get_state()
                q_values = agent.get_q_values(state)
                max_Q[i, j] = np.max(q_values)

    im = ax.imshow(max_Q, cmap='viridis', origin='upper')
    plt.colorbar(im, ax=ax, label='Max Q-Value ($V^*(s)$)')

    arrows = ['↑', '↓', '←', '→']
    marker_size = max(10, min(20, 200 // size))
    arrow_font_size = max(6, min(12, 120 // size))

    # Draw cells
    for i in range(size):
        for j in range(size):
            pos = (i, j)

            # Goal
            if pos == temp_env.goal_pos:
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2,
                                         edgecolor='red', facecolor='crimson', alpha=0.6)
                ax.add_patch(rect)
                ax.text(j, i, "🎯", ha='center', va='center', fontsize=marker_size + 4,
                        color='white', fontweight='bold')
            # Obstacle
            elif list(pos) in temp_env.obstacles:
                ax.text(j, i, "🧱", ha='center', va='center', fontsize=marker_size)
            # Start
            elif pos == temp_env.start_pos:
                ax.text(j, i, "🏠", ha='center', va='center', fontsize=marker_size)
            else:
                # Arrows for policy
                temp_env.agent_pos = [i, j]
                state = temp_env._get_state()
                q_values = agent.get_q_values(state)
                a = np.argmax(q_values)
                ax.text(j, i, arrows[a], ha='center', va='center', fontsize=arrow_font_size,
                        color='white', fontweight='bold', alpha=0.8)

    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linestyle='-', color='gray')

    # Animation setup
    plt.ion()
    done = False
    steps = 0
    max_animation_steps = 20
    temp_env.agent_pos = list(temp_env.start_pos)

    agent_dot, = ax.plot(temp_env.agent_pos[1], temp_env.agent_pos[0], 'o',
                         color='black', markersize=max(5, marker_size / 2),
                         label='Agent', zorder=10)

    success_rate = metrics.get_success_rate()
    ax.set_title(f"{title_prefix} | Success Rate: {success_rate:.1%}")
    plt.draw()
    plt.pause(0.3)

    # Animate agent movement
    state = temp_env._get_state()
    while not done and steps < max_animation_steps:
        agent_dot.set_data([temp_env.agent_pos[1]], [temp_env.agent_pos[0]])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(speed)

        action = agent.select_action(state, epsilon=0.0)
        state, _, done = temp_env.step(action)
        steps += 1

    agent_dot.set_data([temp_env.agent_pos[1]], [temp_env.agent_pos[0]])
    status = 'Goal' if tuple(temp_env.agent_pos) == temp_env.goal_pos else 'Timeout'
    ax.set_title(f"TERMINÉ en {steps} étapes ({status}) 🏁 | SR: {success_rate:.1%}")

    plt.ioff()
    fig.canvas.draw()
    plt.show(block=True)
    plt.close(fig)


# ==========================================
# INCREMENTAL LEARNING FUNCTION
# ==========================================
def learn_and_visualize_incrementally_dqn(grid_size, total_episodes, batch_size, 
                                         fixed_obstacles, viz_interval):
    """Trains DQN agent with incremental visualization."""
    env = GridWorld(size=grid_size, fixed_obstacles=fixed_obstacles)
    state_size = grid_size * grid_size + 4
    action_size = env.action_space
    
    agent = DQNAgent(state_size, action_size, learning_rate=0.001)
    metrics = MetricsTracker()
    
    gamma = 0.9
    eps_start = 1.0
    eps_end = 0.05
    decay_rate = (eps_start - eps_end) / total_episodes
    dqn_batch_size = 32
    target_update_freq = 10
    
    print(f"Training DQN with Neural Network on {grid_size}x{grid_size} grid")
    print(f"State size: {state_size}, Action size: {action_size}")
    
    for ep in range(total_episodes):
        epsilon = max(eps_end, eps_start - ep * decay_rate)
        reward, steps = run_dqn_episode(env, agent, epsilon, gamma, dqn_batch_size, train=True)
        
        success = (tuple(env.agent_pos) == env.goal_pos)
        avg_loss = np.mean(agent.loss_history[-10:]) if agent.loss_history else None
        avg_q = np.mean(agent.q_value_history[-10:]) if agent.q_value_history else None
        
        metrics.update(reward, steps, success, avg_loss, avg_q, epsilon)
        
        if (ep + 1) % target_update_freq == 0:
            agent.update_target_network()
        
        if (ep + 1) % viz_interval == 0:
            print(f"\n--- Episode {ep + 1}/{total_episodes} ---")
            print(f"Reward: {reward:.2f}, Steps: {steps}, Success Rate: {metrics.get_success_rate():.1%}")
            
            title = f"DQN Policy after {ep + 1}/{total_episodes} Eps (ε={epsilon:.2f})"
            visualize_policy_and_animate_dqn(agent, env, title, metrics, speed=0.3)
    
    metrics.plot_metrics(grid_size)
    
    final_success_rate = metrics.get_success_rate()
    print(f"\nFinal Success Rate: {final_success_rate:.1%}")
    
    return agent, metrics, final_success_rate


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    print("--- Deep Q-Learning avec Réseau de Neurones ---")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    grid_configs = {
        #5: {'episodes': 200, 'viz_interval': 40, 'obstacles': 2},
        10: {'episodes': 1000, 'viz_interval': 200, 'obstacles': 5},
    }
    
    results = {}
    
    for size, config in grid_configs.items():
        print(f"\n{'='*60}")
        print(f"### Grille {size}x{size} (Episodes: {config['episodes']}) ###")
        print(f"{'='*60}")
        
        agent, metrics, success_rate = learn_and_visualize_incrementally_dqn(
            size, config['episodes'], 32, config['obstacles'], config['viz_interval']
        )
        
        results[size] = success_rate
    
    # Final comparison
    plt.close('all')
    grid_sizes = list(results.keys())
    success_rates = list(results.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(grid_sizes, success_rates, color='darkblue', alpha=0.7, edgecolor='black')
    ax.set_title("Taux de Réussite Final vs. Taille de Grille (DQN) 🧠", fontsize=14, fontweight='bold')
    ax.set_xlabel("Taille de la grille (N x N)")
    ax.set_ylabel("Taux de Réussite")
    ax.set_ylim([0, 1.1])
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_xticks(grid_sizes)
    
    for i, (size, rate) in enumerate(zip(grid_sizes, success_rates)):
        ax.text(size, rate + 0.03, f'{rate:.1%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show(block=True)