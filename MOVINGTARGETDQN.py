import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.animation as animation

# ==========================================
# NEURAL NETWORK Q-FUNCTION APPROXIMATOR
# ==========================================
class QNetwork(nn.Module):
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
# GRIDWORLD ENVIRONMENT WITH MOVING GOAL
# ==========================================
class GridWorld:
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
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx_s, ny_s = self.start_pos[0]+dx, self.start_pos[1]+dy
            if 0<=nx_s<self.size and 0<=ny_s<self.size: coords_to_exclude.add((nx_s, ny_s))
            nx_g, ny_g = self.goal_pos[0]+dx, self.goal_pos[1]+dy
            if 0<=nx_g<self.size and 0<=ny_g<self.size: coords_to_exclude.add((nx_g, ny_g))
        valid_obstacle_coords = [coord for coord in possible_coords if coord not in coords_to_exclude]
        max_possible_obs = len(valid_obstacle_coords)//2
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
        state = np.zeros(self.size*self.size + 4)
        idx = self.agent_pos[0]*self.size + self.agent_pos[1]
        state[idx] = 1.0
        state[-4] = self.goal_pos[0]/self.size
        state[-3] = self.goal_pos[1]/self.size
        dist = abs(self.agent_pos[0]-self.goal_pos[0]) + abs(self.agent_pos[1]-self.goal_pos[1])
        state[-2] = dist/(2*self.size)
        state[-1] = self.steps/(self.size*self.size*4)
        return state

    def _next_pos(self, pos, action):
        x, y = pos
        if action==0: x-=1
        elif action==1: x+=1
        elif action==2: y-=1
        elif action==3: y+=1
        new_x = max(0,min(x,self.size-1))
        new_y = max(0,min(y,self.size-1))
        new_pos = [new_x,new_y]
        if new_pos in self.obstacles:
            return pos
        return new_pos

    def move_goal(self):
        moves = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
        random.shuffle(moves)
        for dx, dy in moves:
            nx, ny = self.goal_pos[0]+dx, self.goal_pos[1]+dy
            if 0<=nx<self.size and 0<=ny<self.size and [nx, ny] not in self.obstacles and [nx, ny] != self.agent_pos:
                self.goal_pos = (nx, ny)
                break

    def step(self, action):
        self.steps += 1
        new_pos = self._next_pos(self.agent_pos, action)
        self.agent_pos = new_pos
        if self.steps % 3 == 0:  # Move goal occasionally
            self.move_goal()
        reward = -1
        done = False
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 10
            done = True
        if self.steps >= self.size*self.size*4:
            done = True
        return self._get_state(), reward, done

# ==========================================
# DEEP Q-LEARNING AGENT
# ==========================================
class DQNAgent:
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
        if random.random()<epsilon:
            return random.randint(0,self.action_size-1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def get_q_values(self,state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_tensor).cpu().numpy()[0]

    def train_step(self,batch_size,gamma):
        if len(self.replay_buffer)<batch_size: return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + gamma*next_q_values*(1-dones)
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
# METRICS TRACKER
# ==========================================
class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []
        self.losses = []
        self.avg_q_values = []
        self.epsilon_values = []
    def update(self,reward,steps,success,loss,avg_q,epsilon):
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.success_rate.append(1 if success else 0)
        if loss is not None: self.losses.append(loss)
        if avg_q is not None: self.avg_q_values.append(avg_q)
        self.epsilon_values.append(epsilon)
    def get_success_rate(self,window=100):
        if len(self.success_rate)<window:
            return np.mean(self.success_rate) if self.success_rate else 0
        return np.mean(self.success_rate[-window:])
    def plot_metrics(self,grid_size):
        plt.close('all')
        fig, axes = plt.subplots(2,3,figsize=(18,10))
        window=max(1,len(self.episode_rewards)//20)
        smoothed = np.convolve(self.episode_rewards,np.ones(window)/window,mode='valid')
        axes[0,0].plot(self.episode_rewards,alpha=0.3,label="Par épisode")
        axes[0,0].plot(np.arange(window-1,len(self.episode_rewards)),smoothed,color='red',label="Moyenne glissante")
        axes[0,0].set_title("Récompenses par Épisode")
        axes[0,0].set_xlabel("Épisode"); axes[0,0].set_ylabel("Récompense"); axes[0,0].legend(); axes[0,0].grid(True,alpha=0.3)
        smoothed_steps=np.convolve(self.episode_steps,np.ones(window)/window,mode='valid')
        axes[0,1].plot(self.episode_steps,alpha=0.3,label="Par épisode")
        axes[0,1].plot(np.arange(window-1,len(self.episode_steps)),smoothed_steps,color='green',label="Moyenne glissante")
        axes[0,1].set_title("Étapes par Épisode"); axes[0,1].set_xlabel("Épisode"); axes[0,1].set_ylabel("Étapes"); axes[0,1].legend(); axes[0,1].grid(True,alpha=0.3)
        window_sr=min(100,len(self.success_rate))
        success_rate_smooth=[np.mean(self.success_rate[max(0,i-window_sr):i+1]) for i in range(len(self.success_rate))]
        axes[0,2].plot(success_rate_smooth,color='purple',linewidth=2)
        axes[0,2].set_title("Taux de Réussite (Fenêtre: 100)"); axes[0,2].set_xlabel("Épisode"); axes[0,2].set_ylabel("Taux de Réussite"); axes[0,2].set_ylim([0,1.1]); axes[0,2].grid(True,alpha=0.3)
        if self.losses: axes[1,0].plot(self.losses,color='orange',alpha=0.6); axes[1,0].set_title("Perte d'Entraînement (MSE)"); axes[1,0].set_xlabel("Étape d'Entraînement"); axes[1,0].set_ylabel("Perte"); axes[1,0].set_yscale('log'); axes[1,0].grid(True,alpha=0.3)
        if self.avg_q_values: axes[1,1].plot(self.avg_q_values,color='teal',alpha=0.7); axes[1,1].set_title("Valeurs Q Moyennes"); axes[1,1].set_xlabel("Étape d'Entraînement"); axes[1,1].set_ylabel("Q Moyen"); axes[1,1].grid(True,alpha=0.3)
        axes[1,2].plot(self.epsilon_values,color='brown',linewidth=2); axes[1,2].set_title("Décroissance Epsilon (Exploration)"); axes[1,2].set_xlabel("Épisode"); axes[1,2].set_ylabel("Epsilon"); axes[1,2].grid(True,alpha=0.3)
        plt.suptitle(f"Métriques DQN - Grille {grid_size}x{grid_size}",fontsize=16,fontweight='bold'); plt.tight_layout(); plt.show(block=False); plt.pause(10.0)

# ==========================================
# VISUALIZATION WITH MOVING GOAL & GIF
# ==========================================
def visualize_moving_goal(agent, env, title_prefix, metrics, speed=0.3, gif_path=None):
    size=env.size
    fig, ax = plt.subplots(figsize=(size,size))
    agent_dot, = ax.plot([], [], 'o', color='black', markersize=12, zorder=10)
    goal_dot, = ax.plot([], [], 's', color='red', markersize=14, zorder=11)
    ax.set_xticks(np.arange(size)); ax.set_yticks(np.arange(size))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(True, linestyle='-', color='gray'); plt.title(title_prefix)
    state=env.reset(); done=False
    def update(frame_num):
        nonlocal state, done
        if done: return agent_dot, goal_dot
        action = agent.select_action(state, epsilon=0.0)
        state, _, done = env.step(action)
        agent_dot.set_data(env.agent_pos[1], env.agent_pos[0])
        goal_dot.set_data(env.goal_pos[1], env.goal_pos[0])
        return agent_dot, goal_dot
    ani = animation.FuncAnimation(fig, update, frames=50, interval=speed*1000, blit=True)
    if gif_path: ani.save(gif_path, writer='pillow')
    plt.show()

# ==========================================
# INCREMENTAL LEARNING
# ==========================================
def learn_and_visualize_incrementally_dqn(grid_size,total_episodes,batch_size,fixed_obstacles,viz_interval):
    env = GridWorld(size=grid_size,fixed_obstacles=fixed_obstacles)
    state_size=grid_size*grid_size + 4
    action_size=env.action_space
    agent=DQNAgent(state_size, action_size, learning_rate=0.001)
    metrics=MetricsTracker()
    gamma=0.9; eps_start=1.0; eps_end=0.05
    decay_rate=(eps_start-eps_end)/total_episodes; dqn_batch_size=32; target_update_freq=10
    for ep in range(total_episodes):
        epsilon=max(eps_end, eps_start - ep*decay_rate)
        reward, steps = run_dqn_episode(env, agent, epsilon, gamma, dqn_batch_size, train=True)
        success=(tuple(env.agent_pos)==env.goal_pos)
        avg_loss=np.mean(agent.loss_history[-10:]) if agent.loss_history else None
        avg_q=np.mean(agent.q_value_history[-10:]) if agent.q_value_history else None
        metrics.update(reward, steps, success, avg_loss, avg_q, epsilon)
        if (ep+1) % target_update_freq==0: agent.update_target_network()
        if (ep+1) % viz_interval==0:
            title=f"DQN Policy after {ep+1}/{total_episodes} Eps (ε={epsilon:.2f})"
            print(f"\nEpisode {ep+1}: Reward={reward}, Steps={steps}, Success Rate={metrics.get_success_rate():.1%}")
            visualize_moving_goal(agent, env, title, metrics, speed=0.3, gif_path=f"dqn_episode_{ep+1}.gif")
    metrics.plot_metrics(grid_size)
    print(f"\nFinal Success Rate: {metrics.get_success_rate():.1%}")
    return agent, metrics, metrics.get_success_rate()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__=='__main__':
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    grid_configs={10:{'episodes':1000,'viz_interval':200,'obstacles':5}}
    results={}
    for size, config in grid_configs.items():
        print(f"\n### Grille {size}x{size} ###")
        agent, metrics, success_rate = learn_and_visualize_incrementally_dqn(
            size, config['episodes'], 32, config['obstacles'], config['viz_interval']
        )
        results[size] = success_rate
    plt.close('all')
    grid_sizes=list(results.keys()); success_rates=list(results.values())
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(grid_sizes, success_rates,color='darkblue',alpha=0.7,edgecolor='black')
    ax.set_title("Taux de Réussite Final vs. Taille de Grille (DQN)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Taille de la grille (N x N)"); ax.set_ylabel("Taux de réussite final"); ax.set_ylim(0,1.0)
    for i, v in enumerate(success_rates): ax.text(grid_sizes[i]-0.25, v+0.02, f"{v:.1%}", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3); plt.show()
