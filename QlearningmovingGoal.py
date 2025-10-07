import numpy as np

import matplotlib.pyplot as plt

import random

import time

import matplotlib.patches as patches


# ==========================================

# GRIDWORLD ENVIRONMENT (Deterministic)

# ==========================================

class GridWorld:

    """A deterministic GridWorld environment for RL testing with a randomized goal."""

    def __init__(self, size=5, fixed_obstacles=3):

        self.size = size

        self.action_space = 4

        self.fixed_obstacles = fixed_obstacles

        

        self.start_pos = (0, 0)

        

        # --- MODIFICATION: Initialize goal/obstacles on first creation/reset ---

        self.goal_pos = (0, 0)  # Placeholder

        self.obstacles = []     # Placeholder

        self._setup_new_environment() # Call setup to set initial goal and obstacles

        

        self.reset()

        

    def _setup_new_environment(self):

        """Generates a new, random goal and set of obstacles."""

        self.goal_pos = self._select_random_goal()

        self.obstacles = self._generate_obstacles()


    def _select_random_goal(self):

        """Selects a random goal position, avoiding the start (0,0) and immediate neighbors."""

        possible_coords = [(r, c) for r in range(self.size) for c in range(self.size)]

        

        # Exclude start and neighbors near start

        coords_to_exclude = {self.start_pos}

        if self.size > 2:

            coords_to_exclude.add((0, 1))

            coords_to_exclude.add((1, 0))

            if self.size > 3:

                coords_to_exclude.add((1, 1))


        valid_goals = [coord for coord in possible_coords if coord not in coords_to_exclude]

        

        # Prefer a goal in the second half of the grid if possible

        potential_goals = [

            (r, c) for r, c in valid_goals 

            if r >= self.size // 2 or c >= self.size // 2

        ]

        

        if not potential_goals:

            return random.choice(valid_goals)

        return random.choice(potential_goals)


    def _generate_obstacles(self):

        """Generates a fixed number of random obstacles, avoiding current start and goal."""

        if self.size < 4 or self.fixed_obstacles == 0:

            return []

        

        possible_coords = [(r, c) for r in range(self.size) for c in range(self.size)]

        

        # Exclude start, goal, and immediate neighbors of start/goal

        coords_to_exclude = {self.start_pos, self.goal_pos}

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

            # Neighbors of start

            nx_s, ny_s = self.start_pos[0] + dx, self.start_pos[1] + dy

            if 0 <= nx_s < self.size and 0 <= ny_s < self.size:

                coords_to_exclude.add((nx_s, ny_s))

            # Neighbors of goal

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


    # --- MODIFICATION: Call _setup_new_environment in reset() to randomize goal ---

    def reset(self):

        """Resets the agent position and sets a new random goal/obstacle configuration."""

        self._setup_new_environment() 

        self.agent_pos = list(self.start_pos)

        self.steps = 0

        return tuple(self.agent_pos)


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

        success = False # Added success flag

        

        if tuple(self.agent_pos) == self.goal_pos:

            reward = 10

            done = True

            success = True # Set success flag

        

        if self.steps >= self.size * self.size * 4:

            done = True

            

        # Return success in the info dictionary

        return tuple(self.agent_pos), reward, done, {'success': success}



# ==========================================

# Q-LEARNING CORE   

# ==========================================

# --- MODIFICATION: Update function signature to return success ---

def run_q_learning_episode(env, Q, alpha, gamma, epsilon):

    """Runs a single Q-learning episode and updates the Q-table."""

    state = env.reset()

    total_reward = 0

    done = False

    success = False # Initialize success flag

    

    while not done:

        x, y = state

        

        if random.random() < epsilon:

            action = random.randint(0, env.action_space - 1) 

        else:

            action = np.argmax(Q[x, y, :]) 


        # Retrieve success from the step's info

        next_state, reward, done, info = env.step(action)

        success = info['success']

        

        nx, ny = next_state

        

        # Check if next_state is terminal before calculating max Q

        Q_next = 0 if done else np.max(Q[nx, ny, :])

        

        Q[x, y, action] += alpha * (reward + gamma * Q_next - Q[x, y, action])

        

        state = next_state

        total_reward += reward

        

    # Return success flag along with Q-table and reward

    return Q, total_reward, success


def train_q_learning_full(grid_size, episodes, fixed_obstacles):

    """Standard full training function with equiprobable initialization."""

    alpha = 0.1

    gamma = 0.9

    eps_start = 1.0

    eps_end = 0.05

    decay_rate = (eps_start - eps_end) / episodes

    

    env = GridWorld(size=grid_size, fixed_obstacles=fixed_obstacles)

    

    # Equiprobable Policy Initialization

    Q = np.random.uniform(low=0.0, high=0.001, size=(grid_size, grid_size, env.action_space))

    

    rewards = []

    successes = [] # --- MODIFICATION: New metric to track success rate ---

    

    for ep in range(episodes):

        epsilon = max(eps_end, eps_start - ep * decay_rate)

        

        # Retrieve success from the episode

        Q, reward, success = run_q_learning_episode(env, Q, alpha, gamma, epsilon)

        rewards.append(reward)

        successes.append(success) # Store success boolean

        

    return Q, rewards, env, successes



# ==========================================

# VISUALIZATION FUNCTIONS 

# ==========================================

# --- MODIFICATION: Plotting function updated to include Success Rate ---

def plot_learning_metrics(all_rewards, all_successes, grid_size):

    """Plots the cumulative reward and the success rate per episode."""

    plt.close('all') 

    

    # Calculate Success Rate metric (Moving Average)

    window = max(1, len(all_successes) // 20)

    smoothed_rewards = np.convolve(all_rewards, np.ones(window)/window, mode='valid')

    smoothed_successes = np.convolve(all_successes, np.ones(window)/window, mode='valid')

    

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    

    # --- Plot 1: Reward Convergence ---

    ax1.plot(all_rewards, alpha=0.3, label="Récompense par épisode")

    ax1.plot(np.arange(window - 1, len(all_rewards)), smoothed_rewards, color='red', label=f"Moyenne glissante Récompense ({window} épisodes)")

    ax1.set_title(f"Performance Q-Learning {grid_size}x{grid_size} (Goal Randomisé) 🎯")

    ax1.set_ylabel("Somme des récompenses")

    ax1.legend(loc='lower right')

    ax1.grid(True, linestyle='--')

    

    # --- Plot 2: Success Rate Metric ---

    ax2.plot(np.arange(window - 1, len(all_successes)), smoothed_successes, color='blue', label=f"Taux de Succès ({window} épisodes)")

    ax2.axhline(0.7, color='green', linestyle=':', label='Cible de Succès (70%)')

    ax2.set_xlabel("Épisode")

    ax2.set_ylabel("Taux de Succès Moyen")

    ax2.legend(loc='lower right')

    ax2.grid(True, linestyle='--')

    ax2.set_ylim(-0.05, 1.05)

    

    plt.tight_layout()

    plt.show(block=False) 

    plt.pause(0.5) 


# --- Definition of the missing function (must be present before its call) ---

def visualize_policy_and_animate(Q, env, title_prefix, speed=0.3):

    """

    Combines policy visualization (heatmap, arrows) and agent animation 

    into a single dynamic plot, with proper figure management.

    """

    size = env.size


    # Close any existing figures and create fresh one

    plt.close('all')

    fig, ax = plt.subplots(figsize=(size, size))

    

    # 1. Static Background Setup (Q-Value Heatmap)

    max_Q = np.max(Q, axis=2)

    non_terminal_q_values = [max_Q[i, j] for i in range(size) for j in range(size) 

                             if (i, j) != env.goal_pos and [i, j] not in env.obstacles]

    vmax = np.max(non_terminal_q_values) if non_terminal_q_values else 1.0

    vmin = np.min(non_terminal_q_values) if non_terminal_q_values else 0.0

    

    q_map = np.full((size, size), vmin)

    for i in range(size):

        for j in range(size):

            if (i, j) != env.goal_pos and list((i, j)) not in env.obstacles: # Ensure goal and obs are skipped

                q_map[i, j] = max_Q[i, j]


    im = ax.imshow(q_map, cmap='viridis', origin='upper', vmin=vmin, vmax=vmax)

    plt.colorbar(im, ax=ax, label='Max Q-Value ($V^*(s)$)')

    

    # 2. Static Elements and Policy Arrows

    arrows = ['↑', '↓', '←', '→']

    marker_size = max(10, min(20, 200 // size))

    font_size = max(8, min(15, 150 // size))

    arrow_font_size = max(6, min(12, 120 // size))


    for i in range(size):

        for j in range(size):

            pos = (i, j)

            

            if pos == env.goal_pos:

                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0, 

                                         edgecolor='none', facecolor='red', alpha=0.4)

                ax.add_patch(rect)

                ax.text(j, i, "🎯", ha='center', va='center', fontsize=marker_size, 

                        color='white', fontweight='bold')

            elif list(pos) in env.obstacles:

                ax.text(j, i, "🧱", ha='center', va='center', fontsize=marker_size)

            elif pos == env.start_pos:

                ax.text(j, i, "🏠", ha='center', va='center', fontsize=marker_size)

            

            # Policy arrows

            if pos != env.goal_pos and list(pos) not in env.obstacles:

                a = np.argmax(Q[i, j, :])

                ax.text(j, i, arrows[a], ha='center', va='center', fontsize=arrow_font_size, 

                        color='white', fontweight='bold', alpha=0.7)


    ax.set_xticks(np.arange(size))

    ax.set_yticks(np.arange(size))

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.grid(True, linestyle='-', color='gray')

    

    # 3. Agent Animation - Use interactive mode properly

    plt.ion()

    # IMPORTANT: Reset the environment again to start animation from (0,0) in the current goal setup

    state = env.reset() 

    done = False

    steps = 0

    max_animation_steps = 30

    

    agent_dot, = ax.plot(state[1], state[0], 'ko', markersize=max(5, marker_size / 2), 

                         label='Agent', zorder=10)

    

    plt.draw()

    plt.pause(0.1) 


    start_time = time.time()

    while not done and steps < max_animation_steps:

        x, y = state

        

        ax.set_title(f"{title_prefix} | Étape: {steps} | Récompense: {env.steps * -1}")

        

        agent_dot.set_data([y], [x])

        

        fig.canvas.draw()

        fig.canvas.flush_events()

        plt.pause(speed)

        

        action = np.argmax(Q[x, y, :])

        # Only use env.step, ignore reward/done/info for visualization

        state, _, done, _ = env.step(action) 

        steps += 1

        

    end_time = time.time()

    

    # Final position

    agent_dot.set_data([state[1]], [state[0]])

    

    status = 'Goal' if tuple(env.agent_pos) == env.goal_pos else 'Timeout'

    ax.set_title(f"TERMINÉ en {steps} étapes ({status}) 🏁")

    

    plt.ioff()

    fig.canvas.draw()

    plt.show(block=True)

    plt.close(fig)

    

    print(f"Animation terminée en {steps} étapes. (Durée: {end_time - start_time:.2f}s)")



# ==========================================

# INCREMENTAL LEARNING FUNCTION

# ==========================================

def learn_and_visualize_incrementally(grid_size, total_episodes, batch_size, fixed_obstacles):

    """Trains the Q-learning agent in batches with visualizations."""

    alpha = 0.2

    gamma = 0.9

    eps_start = 1.0

    eps_end = 0.05

    decay_rate = (eps_start - eps_end) / total_episodes

    

    # The environment now randomizes goal on every reset()

    env = GridWorld(size=grid_size, fixed_obstacles=fixed_obstacles) 

    Q = np.random.uniform(low=0.0, high=0.001, size=(grid_size, grid_size, env.action_space))

    all_rewards = []

    all_successes = [] # Track all successes


    # Starting goal is only for initial printout, it will change frequently

    print(f"Initial Goal (will be randomized each episode): {env.goal_pos}")


    for current_ep in range(0, total_episodes, batch_size):

        epsilon = max(eps_end, eps_start - current_ep * decay_rate)

        

        batch_rewards = []

        batch_successes = []

        for _ in range(batch_size):

            # Retrieve success from the episode

            Q, reward, success = run_q_learning_episode(env, Q, alpha, gamma, epsilon)

            batch_rewards.append(reward)

            batch_successes.append(success)

            epsilon = max(eps_end, epsilon - decay_rate)

        

        all_rewards.extend(batch_rewards)

        all_successes.extend(batch_successes)

        

        total_ep_count = current_ep + batch_size

        

        # The title now indicates the generalized learning objective

        title = f"Policy après {total_ep_count}/{total_episodes} Eps (ε={epsilon:.2f}) | BUT RANDOMISÉ"

        

        # Calculate moving average success rate for this batch

        avg_success = np.mean(all_successes[-batch_size:])

        

        print(f"\n--- Batch finished. Progress after Episode {total_ep_count} ---")

        print(f"Goal for animation: {env.goal_pos} | Last Batch Success Rate: {avg_success:.2f}")

        

        # Visualize the policy on the *last* random environment generated by the batch

        visualize_policy_and_animate(Q, env, title)

        

    final_avg_reward = np.mean(all_rewards[-batch_size:])

    final_success_rate = np.mean(all_successes[-batch_size:])

    

    print(f"\nFinal average reward (last {batch_size} episodes): {final_avg_reward:.2f}")

    print(f"Final average SUCCESS RATE (last {batch_size} episodes): {final_success_rate:.2f}")

    

    # Plot both metrics for final analysis

    plot_learning_metrics(all_rewards, all_successes, grid_size)

    

    # Return the final success rate as the primary metric

    return final_success_rate, "Randomized" 



# ==========================================

# MAIN EXECUTION

# ==========================================

if __name__ == '__main__':

    print("--- Démarrage de l'entraînement Q-Learning avec BUT RANDOMISÉ ---")

    

    # Configurations

    grid_configs = {

        # Increased episodes significantly because the task is much harder 

        # (agent must learn to identify the goal based on the state, not memorize a path)
        #5: {'episodes': 100, 'batch_size': 20, 'obstacles': 2, 'incremental': True},
        15: {'episodes': 1000, 'batch_size': 200, 'obstacles': 2, 'incremental': True}

    }

    

    results = {}

    

    for size, config in grid_configs.items():

        episodes = config['episodes']

        obstacles = config['obstacles']

        

        print(f"\n=======================================================")

        print(f"### Démarrage de la grille {size}x{size} (Épisodes: {episodes}) - BUT RANDOMISÉ ###")

        print(f"=======================================================")


        if config['incremental']:

            batch_size = config['batch_size']

            # The function now returns final success rate

            final_metric, goal_pos_status = learn_and_visualize_incrementally(

                size, episodes, batch_size, obstacles

            )

            results[size] = {'metric': final_metric, 'goal_status': goal_pos_status}


    # --- FINAL COMPARISON ---

    plt.close('all')

    grid_sizes = list(results.keys())

    # Plotting final success rate instead of just reward

    final_metrics = [r['metric'] for r in results.values()]

    

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(grid_sizes, final_metrics, marker='o', linestyle='-', color='darkgreen', 

            linewidth=2, markersize=8)

    ax.set_title("Performance Finale vs. Taille de la grille (But Randomisé) 📈")

    ax.set_xlabel("Taille de la grille (N x N)")

    ax.set_ylabel("Taux de Succès Moyen Final")

    ax.set_ylim(0, 1.05)

    ax.grid(True, linestyle='--')

    ax.set_xticks(grid_sizes)

    plt.tight_layout()

    plt.show(block=True)

    plt.close(fig)