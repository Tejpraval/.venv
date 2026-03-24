import random

import numpy as np


NUM_STATES = 5
NUM_ACTIONS = 3
EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
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


def choose_action(state: int, q_table: np.ndarray) -> int:
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, NUM_ACTIONS - 1)
    return int(np.argmax(q_table[state]))


def learn(q_table: np.ndarray, state: int, action: int, reward: int, next_state: int) -> None:
    predict = q_table[state][action]
    target = reward + GAMMA * np.max(q_table[next_state])
    q_table[state][action] += ALPHA * (target - predict)


def run_simulation() -> None:
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
    rewards = []

    for episode in range(1, EPISODES + 1):
        state = sense_environment()
        action = choose_action(state, q_table)
        next_state = sense_environment()
        reward = get_reward(state, action)
        learn(q_table, state, action, reward, next_state)
        rewards.append(reward)

        if episode % 100 == 0:
            print(
                f"Episode {episode:4d} | "
                f"State={STATE_LABELS[state]:9s} | "
                f"Action={ACTION_LABELS[action]:20s} | "
                f"Reward={reward:2d}"
            )

    print("\nFinal policy")
    for state in range(NUM_STATES):
        action = int(np.argmax(q_table[state]))
        print(f"{STATE_LABELS[state]:9s} -> {ACTION_LABELS[action]}")

    print(f"\nAverage reward: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    run_simulation()
