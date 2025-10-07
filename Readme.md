# 🟢 GridWorld Q-Learning Agent

**A deterministic GridWorld environment with a Q-learning agent capable of learning to reach a randomized goal while avoiding obstacles. Includes visualization of policy and agent movement.**

---

## 📝 Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Visualization](#visualization)  
- [Configuration](#configuration)  
- [Results](#results)  
- [License](#license)  

---

## 🔹 Project Overview

This project demonstrates **Reinforcement Learning** using **Q-learning** in a grid-based environment.  

The agent learns to navigate from a fixed start position to a **randomized goal** while avoiding static obstacles. The environment is deterministic, and the Q-learning algorithm updates a Q-table for optimal decision-making.  

Key educational goals:

- Understand basic **RL concepts**: states, actions, rewards.  
- Learn Q-learning updates and exploration-exploitation trade-offs.  
- Visualize **learning metrics** (reward & success rate).  
- Animate agent trajectories with policy arrows and goal markers.  

---

## ✨ Features

- Deterministic **GridWorld environment** with customizable size.  
- Randomized goals per episode.  
- Fixed number of obstacles placed strategically.  
- Q-learning with **epsilon-greedy policy** and adjustable hyperparameters.  
- Incremental batch training with real-time visualization.  
- Metrics tracking: **reward convergence** and **success rate**.  
- Policy visualization using **arrows and heatmap**.  
- Agent animation with matplotlib.  

---

## ⚙️ Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/gridworld-qlearning.git
cd gridworld-qlearning
