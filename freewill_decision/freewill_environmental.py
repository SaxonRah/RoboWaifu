import numpy as np
from typing import Tuple, Dict, Union
import random

"""
Environmental Interaction
    1. Link decision-making to sensory input and allow the system to "experience" its environment.
    2. Incorporate feedback loops where the system's actions affect the environment
        and vice versa, encouraging adaptive responses.
"""


class Environment:
    def __init__(self, size: int = 10):
        self.size = size
        self.grid = np.zeros((size, size))
        self.agent_pos = (size // 2, size // 2)
        self.resources = []
        self.threats = []
        self.step_count = 0

        # Initialize random resources and threats with constraints
        self._place_elements()

    def _place_elements(self):
        """Place resources and threats with minimum spacing between them"""
        def place_element(existing_elements, marker):
            while True:
                pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if (
                    pos != self.agent_pos
                    and pos not in existing_elements
                    and all(np.linalg.norm(np.array(pos) - np.array(other)) > 2 for other in existing_elements)
                ):
                    return pos

        for _ in range(self.size // 2):
            pos = place_element(self.resources, marker=1)
            self.resources.append(pos)
            self.grid[pos] = 1  # 1 represents resource

        for _ in range(self.size // 3):
            pos = place_element(self.threats + self.resources, marker=-1)
            self.threats.append(pos)
            self.grid[pos] = -1  # -1 represents threat

    def get_sensor_data(self, radius: int = 2) -> np.ndarray:
        """Return a view of the environment around the agent, dynamically scaling radius based on size."""
        radius = min(radius, self.size // 4)  # Dynamically scale sensor radius
        x, y = self.agent_pos
        sensor_view = np.zeros((2 * radius + 1, 2 * radius + 1))

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                new_x, new_y = x + i, y + j
                if 0 <= new_x < self.size and 0 <= new_y < self.size:
                    sensor_view[i + radius, j + radius] = self.grid[new_x, new_y]

        return sensor_view

    def adapt_environment(self):
        """Adapt environment dynamically based on agent actions."""
        if self.step_count % 50 == 0:
            # Add new resources and threats periodically
            if len(self.resources) < self.size:
                self._place_elements()

    def update(self, given_action: Tuple[int, int]) -> Tuple[float, bool]:
        """Update environment based on agent action and return reward"""
        self.step_count += 1

        new_x = self.agent_pos[0] + given_action[0]
        new_y = self.agent_pos[1] + given_action[1]

        # Check boundaries
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            return -0.5 - 0.1 * self.step_count, False  # Scaling penalty for hitting boundaries

        self.agent_pos = (new_x, new_y)

        # Check for resource collection or threat encounter
        if self.agent_pos in self.resources:
            self.resources.remove(self.agent_pos)
            self.grid[self.agent_pos] = 0
            return 1.0, False  # Reward for collecting resource
        elif self.agent_pos in self.threats:
            return -1.0, True  # Penalty for hitting threat and end episode

        # Small penalty for each move to encourage efficiency
        return -0.1 - 0.01 * self.step_count, False


class AdaptiveAgent:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table: Dict[str, Dict[Tuple[int, int], float]] = {}
        self.possible_actions = [
            (-1, 0),  # Up
            (1, 0),  # Down
            (0, -1),  # Left
            (0, 1)  # Right
        ]

    @staticmethod
    def _get_state_key(sensor_data: Union[str, np.ndarray]) -> str:
        """Convert sensor data to a string key for Q-table"""
        if isinstance(sensor_data, np.ndarray):
            return sensor_data.tobytes().hex()
        elif isinstance(sensor_data, str):
            return sensor_data
        else:
            raise ValueError(f"Unsupported state type: {type(sensor_data)}")

    def choose_action(self, sensor_data: Union[str, np.ndarray], epsilon: float = 0.1) -> Tuple[int, int]:
        """
        Choose action using epsilon-greedy policy.
        Now supports both string and numpy array states.
        """
        state_key = self._get_state_key(sensor_data)

        # Initialize state in Q-table if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.possible_actions}

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return random.choice(self.possible_actions)
        else:
            return max(self.q_table[state_key].items(), key=lambda x: x[1])[0]

    def learn(self,
              state: Union[str, np.ndarray],
              action: Tuple[int, int],
              reward: float,
              next_state: Union[str, np.ndarray]) -> None:
        """
        Update Q-values based on experience.
        Now supports both string and numpy array states.
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # Initialize states in Q-table if not present
        for key in [state_key, next_state_key]:
            if key not in self.q_table:
                self.q_table[key] = {action: 0.0 for action in self.possible_actions}

        # Q-learning update
        current_q = self.q_table[state_key][action]
        next_max_q = max(self.q_table[next_state_key].values())
        new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state_key][action] = new_q


def train_agent(episodes: int = 1000, max_steps: int = 100):
    """Train the agent in the environment."""
    env = Environment()
    agent = AdaptiveAgent()

    for episode in range(episodes):
        env = Environment()  # Reset environment
        total_reward = 0

        for step in range(max_steps):
            # Get current state
            current_state = env.get_sensor_data()

            # Choose and perform action
            action = agent.choose_action(current_state, epsilon=max(0.01, 0.5 - episode / episodes))

            # Get reward and next state
            reward, done = env.update(action)
            next_state = env.get_sensor_data()

            # Learn from experience
            agent.learn(current_state, action, reward, next_state)

            total_reward += reward

            if done or len(env.resources) == 0:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

    return agent


def main():
    # Train the agent
    trained_agent = train_agent()

    # Test the trained agent
    test_env = Environment()
    test_steps = 50
    total_reward = 0

    print("\nTesting trained agent:")
    for step in range(test_steps):
        state = test_env.get_sensor_data()
        action = trained_agent.choose_action(state, epsilon=0.0)  # No exploration
        reward, done = test_env.update(action)
        total_reward += reward

        if done or len(test_env.resources) == 0:
            break

    print(f"Test completed. Total reward: {total_reward:.2f}")
    print(f"Resources remaining: {len(test_env.resources)}")


if __name__ == "__main__":
    main()
