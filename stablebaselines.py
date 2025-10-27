import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os

# =======================================================
# 1. Custom Environment (Gymnasium Interface)
# =======================================================
class StableBaselinesEnv(gym.Env):
    """
    A wrapper for the MovingTargetEnv to make it Gymnasium-compatible.
    """
    def __init__(self, size=7, max_steps=50):
        super(StableBaselinesEnv, self).__init__()
        self.size = size
        self.max_steps = max_steps  # ADD: Maximum steps per episode
        
        # Define Action and Observation Space
        # Action space: 0=left, 1=right (Discrete)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: 2 * size (One-hot for agent + One-hot for target)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.size * 2,), dtype=np.float32
        )
        
        # Initialize state variables
        self.agent = 0
        self.target = 0
        self.steps = 0  # ADD: Step counter

    def _get_state(self):
        # One-hot agent position + one-hot target position
        s = np.zeros(self.size * 2, dtype=np.float32)
        s[self.agent] = 1.0
        s[self.size + self.target] = 1.0
        return s

    def reset(self, seed=None, options=None):
        # Must call super().reset() for proper seeding
        super().reset(seed=seed) 
        self.agent = 0
        self.steps = 0  # Reset step counter
        
        # Use self.np_random.integers for proper seeding
        self.target = self.np_random.integers(0, self.size) 
        
        observation = self._get_state()
        info = {}
        return observation, info

    def step(self, action):
        self.steps += 1  # Increment step counter
        
        # Actions: 0=left, 1=right
        if action == 0:
            self.agent = max(0, self.agent - 1)
        elif action == 1:
            self.agent = min(self.size - 1, self.agent + 1)

        # Target moves randomly each step
        move = self.np_random.choice([-1, 0, 1]) 
        self.target = int(np.clip(self.target + move, 0, self.size - 1))

        # Reward & Done condition
        hit = self.agent == self.target
        reward = 1.0 if hit else -0.1
        
        # Episode terminates on hit or max steps
        terminated = hit
        truncated = self.steps >= self.max_steps
        
        observation = self._get_state()
        info = {}
        
        return observation, reward, terminated, truncated, info


# =======================================================
# 2. Stable Baselines3 Training
# =======================================================

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

TIMESTEPS = 30000 
LOG_DIR = "./sb3_dqn_target_log/"

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# 1. Environment creation function (with proper seeding)
def make_env():
    env = StableBaselinesEnv(size=7, max_steps=50)
    env = Monitor(env, LOG_DIR)
    return env

# 2. Vectorize the environment
vec_env = make_vec_env(
    make_env,
    n_envs=1,
    seed=42
)

# 3. Create the DQN Model
model = DQN(
    "MlpPolicy",                   
    vec_env,                       
    learning_rate=0.001,           
    buffer_size=10000,             
    learning_starts=1000,          
    batch_size=64,                 
    gamma=0.95,                    
    exploration_fraction=0.5,      
    exploration_final_eps=0.05,    
    target_update_interval=250,    
    verbose=1,                     
    tensorboard_log=LOG_DIR,       
    policy_kwargs=dict(net_arch=[64, 64]),
    seed=42
)

# 4. Train the agent
print("Starting SB3 training...")
model.learn(
    total_timesteps=TIMESTEPS, 
    log_interval=10,  # Log more frequently
    tb_log_name="DQN_MovingTarget"
)
print("SB3 Training finished. 🎉")

# Save the trained model
model.save("dqn_moving_target")
print("Model saved!")


# =======================================================
# 3. Visualization (Plotting from SB3 logs)
# =======================================================
def plot_results(log_folder, title='SB3 Learning Curve (Smoothed Reward)'):
    """
    Plots the cumulative episode rewards from the Stable Baselines3 monitor files.
    """
    try:
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        print(f"Loaded {len(y)} episodes from logs")
    except Exception as e:
        print(f"Error loading results from {log_folder}: {e}")
        return

    if len(y) == 0:
        print("No episode data found. Make sure episodes are completing.")
        return

    # Simple smoothing (rolling average) for clarity
    window_size = min(50, max(1, len(y) // 10))  # Adaptive window size
    
    plt.figure(figsize=(12, 5))
    
    # Plot raw data
    plt.subplot(1, 2, 1)
    plt.plot(x, y, alpha=0.3, label='Raw')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Raw Episode Rewards')
    plt.grid(True)
    plt.legend()
    
    # Plot smoothed data
    plt.subplot(1, 2, 2)
    if len(y) >= window_size:
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        x_smooth = x[window_size - 1:]
        plt.plot(x_smooth, y_smooth, label=f'Smoothed (window={window_size})')
    else:
        plt.plot(x, y, label='Episode Reward')
    
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward (Smoothed)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Run the plotting function
plot_results(LOG_DIR)