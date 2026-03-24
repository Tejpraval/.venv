import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


NUM_STATES = 5
NUM_ACTIONS = 3
ACTION_LABELS = {
    0: "Decrease resources",
    1: "Maintain allocation",
    2: "Increase resources",
}
STATE_LABELS = {
    0: "Idle",
    1: "Low load",
    2: "Balanced",
    3: "Busy",
    4: "Peak load",
}


def sense_environment() -> int:
    return random.randint(0, NUM_STATES - 1)


def get_reward(state: int, action: int) -> int:
    if state == 2 and action == 1:
        return 10
    if state == 0 and action == 0:
        return -5
    if state == 4 and action == 2:
        return -5
    return 1


def choose_action(state: int, q_table: np.ndarray, epsilon: float) -> int:
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    return int(np.argmax(q_table[state]))


def learn(
    q_table: np.ndarray,
    state: int,
    action: int,
    reward: int,
    next_state: int,
    alpha: float,
    gamma: float,
) -> None:
    predict = q_table[state][action]
    target = reward + gamma * np.max(q_table[next_state])
    q_table[state][action] += alpha * (target - predict)


def run_rl_simulation(episodes: int, alpha: float, gamma: float, epsilon: float) -> dict:
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
    reward_track = []
    decisions = []

    for episode in range(1, episodes + 1):
        state = sense_environment()
        action = choose_action(state, q_table, epsilon)
        next_state = sense_environment()
        reward = get_reward(state, action)
        learn(q_table, state, action, reward, next_state, alpha, gamma)
        reward_track.append(reward)
        decisions.append(
            {
                "Episode": episode,
                "State": STATE_LABELS[state],
                "Action": ACTION_LABELS[action],
                "Reward": reward,
                "Next State": STATE_LABELS[next_state],
            }
        )

    policy = []
    for state in range(NUM_STATES):
        best_action = int(np.argmax(q_table[state]))
        policy.append(
            {
                "State": STATE_LABELS[state],
                "Recommended Action": ACTION_LABELS[best_action],
                "Q Score": round(float(np.max(q_table[state])), 2),
            }
        )

    return {
        "q_table": q_table,
        "reward_track": reward_track,
        "decisions": decisions,
        "policy": policy,
    }


def moving_average(values: list[int], window: int = 25) -> list[float]:
    if not values:
        return []

    averages = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averages.append(sum(values[start : index + 1]) / (index - start + 1))
    return averages


def reward_chart(reward_track: list[int]) -> None:
    trend = moving_average(reward_track)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(reward_track, color="#1d4ed8", linewidth=1.2, label="Episode reward")
    ax.plot(trend, color="#f97316", linewidth=2, label="Moving average")
    ax.set_title("Reward Trend")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def init_page() -> None:
    st.set_page_config(page_title="Cloud RL Dashboard", layout="wide")
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
            div[data-testid="stMetric"] {
                background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
                border: 1px solid #bfdbfe;
                padding: 0.9rem;
                border-radius: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls() -> tuple[int, float, float, float]:
    st.sidebar.header("Training Controls")
    episodes = st.sidebar.slider("Training Episodes", 100, 3000, 800, 100)
    alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, 0.10, 0.01)
    gamma = st.sidebar.slider("Discount Factor (gamma)", 0.10, 0.99, 0.90, 0.01)
    epsilon = st.sidebar.slider("Exploration Rate (epsilon)", 0.00, 1.0, 0.20, 0.05)
    return episodes, alpha, gamma, epsilon


def render_dashboard(results: dict) -> None:
    rewards = results["reward_track"]
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    best_reward = max(rewards) if rewards else 0
    latest_reward = rewards[-1] if rewards else 0
    stable_states = sum(
        1 for item in results["policy"] if item["Recommended Action"] == ACTION_LABELS[1]
    )

    st.subheader("Training Summary")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Episodes", len(rewards))
    metric_cols[1].metric("Average Reward", f"{avg_reward:.2f}")
    metric_cols[2].metric("Best Reward", best_reward)
    metric_cols[3].metric("Stable-State Decisions", stable_states)

    top_left, top_right = st.columns((1.5, 1))
    with top_left:
        st.subheader("Reward Performance")
        reward_chart(rewards)
    with top_right:
        st.subheader("Latest Signal")
        latest = results["decisions"][-1]
        st.metric("Current State", latest["State"])
        st.metric("Chosen Action", latest["Action"])
        st.metric("Latest Reward", latest_reward)
        st.caption("The agent learns which scaling move best fits each observed load state.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.subheader("Q-Table")
        q_df = pd.DataFrame(
            results["q_table"],
            index=[STATE_LABELS[idx] for idx in range(NUM_STATES)],
            columns=[ACTION_LABELS[idx] for idx in range(NUM_ACTIONS)],
        )
        st.dataframe(q_df.style.highlight_max(axis=1, color="#dbeafe"), use_container_width=True)
    with bottom_right:
        st.subheader("Learned Policy")
        st.dataframe(pd.DataFrame(results["policy"]), use_container_width=True)

    st.subheader("Recent Decisions")
    recent_df = pd.DataFrame(results["decisions"]).tail(12)
    st.dataframe(recent_df, use_container_width=True)


def main() -> None:
    init_page()
    st.title("Cloud Resource Management Dashboard")
    st.caption("Reinforcement learning simulation for resource scaling decisions.")

    episodes, alpha, gamma, epsilon = sidebar_controls()

    if "results" not in st.session_state:
        st.session_state.results = None

    hero_left, hero_right = st.columns((1.4, 1))
    with hero_left:
        st.markdown(
            """
            ### Operational View
            Train an RL agent to respond to changing cloud load conditions and inspect
            how its resource policy evolves across episodes.
            """
        )
    with hero_right:
        st.info(
            "Balanced load should keep allocation steady. Extreme states should push the "
            "agent toward safer scaling behavior."
        )

    if st.button("Run Training", type="primary", use_container_width=True):
        st.session_state.results = run_rl_simulation(episodes, alpha, gamma, epsilon)

    if st.session_state.results:
        render_dashboard(st.session_state.results)
    else:
        st.warning("Run training to generate dashboard metrics, policy recommendations, and charts.")


if __name__ == "__main__":
    main()
