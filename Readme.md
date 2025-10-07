# 🟢 GridWorld Q-Learning Agent

**A deterministic GridWorld environment with a Q-learning agent capable of learning to reach a randomized goal while avoiding obstacles. Includes visualization of policy and agent movement.**



## 🔹 Project Overview

This project demonstrates **Reinforcement Learning** using **Q-learning** in a grid-based environment.  

The agent learns to navigate from a fixed start position to a **randomized goal** while avoiding static obstacles. The environment is deterministic, and the Q-learning algorithm updates a Q-table for optimal decision-making. 

---

## FILES DESCRIPTION AND RETROSPECTION(MUST READ)

there are currently 6 files 

    the first 3 files (livrable_1/2/3_Baqua_Abdellah)  : implements the previous work that has been demanded .

    QLearningFixedGoal.py   : this file simulates the environment using Qlearning while the goal is fixed , the agent quickly learns to win the game ,we can assume that it    either memorizes the map(locations of goal and obstacles and how to move around them) , or it does learn how to navigate around. its early to tell.

    QlearningmovingGoal.py  : this file also simulates the environment using Qlearning but the goal changes position each time, the agent struggles to learn the underlying process of navigation and have been only memorizing locations , raising the episodes number doesnt help , its safe to assume that the agent model has reached its full capability.

    QlearningNeuralNetwork.py: in this file we introduce neuralnetworks as function approximator for the Q-values , this allows the agent to generalize across different large state spaces where Tabular Qlearning doesnt work.

To modify the gridworld simulation settings 
Search for the code block :

    grid_configs = {
            a: {'episodes': b, 'viz_interval': c, 'obstacles': d},
            10: {'episodes': 1000, 'viz_interval': 200, 'obstacles': 5},
    }

where a² is the surface of the grid , b the number of episodes , c the limit of steps , and d is the number of obstacles .
 



---

##  Features

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
git clone https://github.com/AbdellahBaqua/Gridworld
cd gridworld-qlearning
