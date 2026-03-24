# Cloud Resource Management Dashboard

A simple reinforcement learning project that simulates how a cloud system might decide to scale resources up, keep them steady, or scale them down based on changing workload conditions.

This project includes:

- a Streamlit dashboard for interactive visualization
- a Python console version for basic simulation runs
- a small Q-learning setup that demonstrates how an agent improves its decisions over time

## Project Goal

The goal of this project is to show how reinforcement learning can be used for cloud resource management.

In real cloud systems, workloads change continuously. At some moments the system is idle, at other moments traffic spikes. A good resource manager should:

- reduce resources when demand is very low
- maintain resources when the system is balanced
- increase resources when demand grows

This project simulates that idea with a lightweight Q-learning agent.

## How It Works

The agent observes a simulated environment with 5 possible states:

- `Idle`
- `Low load`
- `Balanced`
- `Busy`
- `Peak load`

For each state, the agent can choose 1 of 3 actions:

- `Decrease resources`
- `Maintain allocation`
- `Increase resources`

The agent receives rewards based on whether the action makes sense for the current state. Over many training episodes, it updates a Q-table and learns which action is best for each state.

## Reinforcement Learning Logic

This project uses Q-learning.

Main parameters:

- `alpha`: learning rate
- `gamma`: discount factor
- `epsilon`: exploration rate

Q-learning update used in the project:

```text
Q(state, action) = Q(state, action) + alpha * (reward + gamma * max(Q(next_state)) - Q(state, action))
```

## Features

### Streamlit Dashboard

The dashboard provides:

- training controls from the sidebar
- reward trend chart
- KPI summary cards
- Q-table display
- learned policy table
- recent decision history

### Console Version

The console version prints:

- periodic training progress
- final learned policy
- average reward

## Project Structure

```text
soft-comp/
├── cloud_rl_app.py   # Streamlit dashboard
├── main.py           # Console simulation
├── .gitignore
└── README.md
```

## Requirements

The project uses:

- Python 3.10+
- Streamlit
- NumPy
- Pandas
- Matplotlib

## Installation

### 1. Open the project folder

```powershell
cd C:\Users\tejpr\Downloads\soft-comp
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install streamlit numpy pandas matplotlib
```

## How To Run

### Run the dashboard

```powershell
python -m streamlit run cloud_rl_app.py
```

After running the command, open:

```text
http://localhost:8501
```

### Run the console simulation

```powershell
python main.py
```

## Dashboard Overview

When you open the Streamlit app, you can:

1. choose training episodes
2. adjust RL parameters such as alpha, gamma, and epsilon
3. run training
4. inspect the reward curve
5. review the Q-table and learned policy

This makes the project useful for learning, demonstration, and simple experimentation.

## Example Use Case

Imagine a cloud platform that hosts applications or APIs.

- If traffic is low, reducing resources can save cost.
- If traffic is stable, keeping the same allocation avoids unnecessary changes.
- If traffic rises sharply, increasing resources can improve reliability and response time.

This project models that decision-making process in a simplified way.

## Why This Project Is Useful

This project is useful for:

- students learning reinforcement learning
- beginners exploring Streamlit dashboards
- developers who want a small RL demo project
- portfolio projects related to AI, cloud, or optimization

## Limitations

This is a simplified simulation, not a production cloud autoscaling system.

Current limitations:

- environment states are randomly generated
- rewards are manually defined
- there is no real cloud provider integration
- there is no historical traffic dataset
- the model is intentionally small for clarity

## Possible Future Improvements

You can improve this project by adding:

- real cloud metrics or monitoring data
- a more realistic reward function
- state transitions based on previous state instead of random sampling
- deployment cost analysis
- CPU and memory utilization charts
- action confidence or policy comparison views
- model persistence using saved Q-tables

## Deployment

This app can be deployed with Streamlit Community Cloud.

Basic deployment flow:

1. push the project to GitHub
2. add a `requirements.txt` file
3. open Streamlit Community Cloud
4. choose the GitHub repository
5. set `cloud_rl_app.py` as the main app file
6. deploy

Suggested `requirements.txt`:

```txt
streamlit
numpy
pandas
matplotlib
```

## Troubleshooting

### `No module named streamlit`

Install Streamlit in the active environment:

```powershell
python -m pip install streamlit
```

### `No module named numpy`

Install NumPy:

```powershell
python -m pip install numpy
```

### Browser shows `ERR_CONNECTION_REFUSED`

This usually means the Streamlit server is not running or the terminal was closed.

Start it again:

```powershell
python -m streamlit run cloud_rl_app.py
```

### VS Code shows import warnings

Select the correct Python interpreter in VS Code:

- `Ctrl+Shift+P`
- `Python: Select Interpreter`
- choose your project virtual environment

## Learning Summary

This project demonstrates the following ideas in a simple and readable form:

- reinforcement learning basics
- Q-learning update logic
- decision-making from rewards
- interactive model visualization with Streamlit
- translating ML logic into a usable dashboard

## Author Notes

This project is intentionally designed to be small, understandable, and easy to extend. It is best treated as a learning and demonstration project rather than a production-ready cloud scaling engine.

## License

You can add your preferred license here, for example:

- MIT License
- Apache 2.0
- GPL

If you plan to publish this on GitHub, adding a `LICENSE` file is recommended.
