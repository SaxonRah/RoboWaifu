import pygame
import numpy as np
from dataclasses import dataclass
import scipy.sparse as sparse
from typing import List, Tuple, Dict, Optional
import math
from enum import Enum, auto


# UCS Configuration
@dataclass
class UCSConfig:
    """Configuration for UCS system"""
    dimensions: int = 11  # Dimensions
    hd_dimension: int = 1000  # Dimension of hypervectors
    temporal_window: float = 2.0  # Time window
    decay_rate: float = 0.15  # Temporal decay
    learning_rate: float = 0.01  # For graph updates
    max_weight: float = 1.0  # Maximum edge weight
    strategy_threshold: float = 0.7  # Threshold for strategy switching
    prediction_horizon: float = 0.5  # Seconds to look ahead


class Strategy(Enum):
    """AI Strategy States"""
    PURSUE = auto()  # Direct pursuit
    INTERCEPT = auto()  # Predictive interception
    FLANK = auto()  # Flanking maneuver
    RETREAT = auto()  # Strategic retreat
    PATROL = auto()  # Patrol pattern


class HDCEncoder:
    """Enhanced HDC encoder for game states"""

    def __init__(self, input_dim: int, output_dim: int):
        self.projection = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.projection /= np.linalg.norm(self.projection, axis=0)
        self.pattern_memory = {}  # Store successful patterns

    def encode(self, x: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode game state with optional context"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        # Include context in encoding if provided
        if context is not None:
            context_norm = context / (np.linalg.norm(context) + 1e-8)
            x_norm = np.concatenate([x_norm, context_norm.reshape(1, -1)], axis=1)

        # Check for dimension compatibility before matrix multiplication
        if x_norm.shape[1] != self.projection.shape[0]:
            raise ValueError(f"Input dimension mismatch: {x_norm.shape[1]} != {self.projection.shape[0]}")

        # Flatten the output to ensure it is 1D
        return np.tanh(x_norm @ self.projection).flatten()

    def store_pattern(self, state: np.ndarray, success_score: float):
        """Store successful movement patterns"""
        padded_state = np.zeros(self.projection.shape[0])  # Match encoder input dimension
        padded_state[:len(state)] = state  # Copy `state` into the padded vector

        encoded = self.encode(padded_state)
        # print(f"Encoded shape: {encoded.shape}")

        # Convert first 3 dimensions of the encoded vector to a fully hashable tuple
        # key = tuple(float(value) if np.isscalar(value) else float(value.item()) for value in encoded[:3])
        key = tuple(float(value) for value in encoded[:3])

        if key not in self.pattern_memory or success_score > self.pattern_memory[key][1]:
            self.pattern_memory[key] = (state, success_score)


class DynamicGraph:
    """Enhanced dynamic graph for strategy relationships"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.weights = sparse.lil_matrix((0, 0))
        self.nodes = []
        self.node_features = {}
        self.strategy_success = {strategy: 0.0 for strategy in Strategy}

    def update_weights(self, features: Dict[int, np.ndarray], success_rate: float) -> None:
        """Update graph weights with selective updates."""
        n = len(self.nodes)
        if n == 0 or self.weights.shape != (n, n):
            return

        # Select recent nodes or neighbors for updates
        recent_nodes = self.nodes[-min(10, n):]
        for i in recent_nodes:
            for j in recent_nodes:
                if i != j:
                    f_i = features[self.nodes[i]]
                    f_j = features[self.nodes[j]]

                    sim = self._bounded_similarity(f_i, f_j)
                    grad = 2 * (self.weights[i, j] - sim * success_rate) + \
                           2 * self.config.learning_rate * self.weights[i, j]
                    new_weight = self.weights[i, j] - self.config.learning_rate * grad
                    self.weights[i, j] = self.weights[j, i] = np.clip(new_weight, 0, self.config.max_weight)

    def _bounded_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute similarity with proven bounds in [0,1]"""
        x = x.ravel()
        y = y.ravel()
        cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
        return 0.5 * (cos_sim + 1)

    def update_strategy_success(self, strategy: Strategy, success_delta: float):
        """Update success rate for given strategy"""
        self.strategy_success[strategy] = np.clip(
            self.strategy_success[strategy] + success_delta, 0.0, 1.0
        )


class TemporalProcessor:
    """Enhanced temporal pattern processor"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.time_buffer: List[Tuple[float, np.ndarray]] = []
        self.pattern_history: List[Tuple[float, np.ndarray]] = []
        self.prediction_model = None
        self.last_pattern_history = 500

    def process(self, t: float, x: np.ndarray) -> np.ndarray:
        """Optimized temporal integration."""
        self.time_buffer.append((t, x))
        cutoff_time = t - self.config.temporal_window
        self.time_buffer = [(t_i, x_i) for t_i, x_i in self.time_buffer if t_i > cutoff_time]

        # Add to pattern history
        self.pattern_history.append((t, x))
        if len(self.pattern_history) > self.last_pattern_history:
            self.pattern_history.pop(0)  # Keep the history size manageable

        weights = np.exp(-self.config.decay_rate * (t - np.array([t_i for t_i, _ in self.time_buffer])))

        # Weighted sum of states
        weighted_states = np.array([x_i for _, x_i in self.time_buffer]) * weights[:, None]
        result = x + weighted_states.sum(axis=0)

        # Predict future state
        prediction = self.predict_future_state(t + self.config.prediction_horizon)
        if prediction is not None:
            # Extract the first two dimensions (X and Y) for prediction adjustment
            adjusted_result = result.copy()  # Create a copy to modify only the relevant parts
            adjusted_result[:2] += 0.3 * (prediction - adjusted_result[:2])  # Update X and Y only
            return adjusted_result  # Return the modified result

        return result

    def predict_future_2state(self, future_t: float) -> Optional[np.ndarray]:
        """Predict future state based on pattern history."""
        if len(self.pattern_history) < 2:
            print("Not enough history for prediction. Returning player position.")
            return self.pattern_history[-1][1][:2] if self.pattern_history else np.zeros(2)

        # Retrieve the last two states
        recent_states = self.pattern_history[-2:]
        t1, x1 = recent_states[0]
        t2, x2 = recent_states[1]

        # Prevent division by zero
        if t2 == t1:
            print("Insufficient time difference; returning last state.")
            return x2[:2]  # Use only X and Y

        if t2 > t1:
            # Calculate velocity and predict the future state
            dt = future_t - t2

            recent_velocities = [
                (x2[:2] - x1[:2]) / (t2 - t1)
                for (t1, x1), (t2, x2) in zip(self.pattern_history[:-1], self.pattern_history[1:])
            ]

            weighted_velocity = np.mean(recent_velocities, axis=0)
            predicted = x2[:2] + weighted_velocity * dt

            # Clip predictions to within bounds
            screen_bounds = np.array([[0, 0], [800, 600]])
            predicted = np.clip(predicted, screen_bounds[0], screen_bounds[1])
            return predicted

        else:
            print("Invalid time difference; using last known state.")
            return x2[:2]

    def predict_future_state(self, future_t: float) -> Optional[np.ndarray]:
        """Predict future state based on pattern history."""
        if len(self.pattern_history) < 2:
            print("Insufficient pattern history; using the latest known state.")
            return self.pattern_history[-1][1][:2] if self.pattern_history else np.zeros(2)

        # Calculate velocities for all consecutive pairs
        recent_velocities = []
        for (t1, x1), (t2, x2) in zip(self.pattern_history[:-1], self.pattern_history[1:]):
            if t2 > t1:
                velocity = (x2[:2] - x1[:2]) / (t2 - t1)
                recent_velocities.append(velocity)

        if not recent_velocities:
            print("No valid velocity data; using the latest known state.")
            return self.pattern_history[-1][1][:2]

        # Weighted average of recent velocities
        weighted_velocity = np.mean(recent_velocities, axis=0)

        # Predict future position
        last_t, last_x = self.pattern_history[-1]
        dt = future_t - last_t
        predicted = last_x[:2] + weighted_velocity * dt

        # Clip predictions to within bounds
        return np.clip(predicted, [0, 0], [800, 600])


class AdaptiveGameAI:
    def __init__(self, config: UCSConfig):
        self.config = config
        self.hdc = HDCEncoder(config.dimensions, config.hd_dimension)
        self.graph = DynamicGraph(config)
        self.temporal = TemporalProcessor(config)

        self.position = np.array([400.0, 300.0])
        self.velocity = np.array([0.0, 0.0])
        self.target_position = np.array([400.0, 300.0])
        self.predicted_pos = np.array([0.0, 0.0])

        self.current_strategy = Strategy.PURSUE
        self.strategy_timer = 0.0
        self.t = 0.0

        # Strategy parameters
        self.patrol_points = [
            np.array([150, 150]),
            np.array([650, 150]),
            np.array([650, 550]),
            np.array([150, 550])
        ]
        self.patrol_index = 0
        self.flank_offset = np.array([0.0, 0.0])
        self.success_rate = 0.5

    def update(self, player_pos: np.array, dt: float) -> None:
        """Update AI state and strategy"""
        self.t += dt
        self.strategy_timer += dt

        # Create state vector with context
        state = np.concatenate([
            self.position,
            self.velocity,
            player_pos,
        ])

        # Create context based on current strategy
        context = np.zeros(len(Strategy))  # Dynamically set size based on number of strategies
        context[self.current_strategy.value - 1] = 1

        # Encode state using HDC
        hd_state = self.hdc.encode(state, context)

        # Process temporal patterns
        temporal_state = self.temporal.process(self.t, hd_state)

        # Update graph
        node_id = len(self.graph.nodes)
        self.graph.nodes.append(node_id)
        self.graph.node_features[node_id] = temporal_state
        self.graph.update_weights(self.graph.node_features, self.success_rate)

        # Predict player's future position
        if len(self.temporal.pattern_history) >= 2:
            player_prediction = self.temporal.predict_future_state(self.t + self.config.prediction_horizon)
            if player_prediction is not None:
                self.predicted_pos = player_prediction[:2] + player_pos
                print(f"Predicted position: {self.predicted_pos}")
            else:
                self.predicted_pos = player_pos
                print("No prediction available, using current player position.")
        else:
            print("Not enough pattern history for prediction.")

        if self.predicted_pos[0] < 0 or self.predicted_pos[1] < 0 or self.predicted_pos[0] > 800 or self.predicted_pos[1] > 600:
            self.predicted_pos = player_pos  # Fallback to player position
            print("Prediction out of bounds, using player position.")

        # Update strategy based on state
        self._update_strategy(player_pos)

        # Execute current strategy
        self._execute_strategy(player_pos, self.predicted_pos, dt)

        # Store successful patterns
        success_score = self.calculate_success(player_pos, self.predicted_pos)
        self.hdc.store_pattern(state, success_score)
        # Update success rate
        self.success_rate = 0.95 * self.success_rate + 0.05 * success_score

    def calculate_success(self, player_pos: np.ndarray, predicted_pos: np.ndarray) -> float:
        """Calculate success rate based on the current strategy."""
        distance_to_player = np.linalg.norm(player_pos - self.position)
        distance_to_predicted = np.linalg.norm(predicted_pos - self.position)

        if self.current_strategy == Strategy.PURSUE:
            # Reward being closer to the player
            return 1.0 / (1.0 + distance_to_player)

        elif self.current_strategy == Strategy.RETREAT:
            # Reward being farther from the player
            return distance_to_player / (distance_to_player + 1.0)

        elif self.current_strategy == Strategy.INTERCEPT:
            # Reward being closer to the predicted position
            return 1.0 / (1.0 + distance_to_predicted)

        elif self.current_strategy == Strategy.FLANK:
            # Ideal flank target offset from the player
            flank_target = player_pos + self.flank_offset

            # Direction vectors
            direction_to_player = self.position - player_pos
            ideal_direction = flank_target - player_pos

            # Normalize vectors
            norm_dir_to_player = direction_to_player / (np.linalg.norm(direction_to_player) + 1e-8)
            norm_ideal_direction = ideal_direction / (np.linalg.norm(ideal_direction) + 1e-8)

            # Angular alignment (cosine similarity)
            alignment = np.dot(norm_dir_to_player, norm_ideal_direction)
            alignment_reward = max(0.0, (alignment + 1.0) / 2.0)  # Map [-1, 1] to [0, 1]

            # Distance to player compared to desired flanking radius
            desired_distance = np.linalg.norm(self.flank_offset)
            current_distance = np.linalg.norm(direction_to_player)
            distance_deviation = abs(current_distance - desired_distance)
            distance_reward = max(0.0, 1.0 - (distance_deviation / desired_distance))

            # Encourage movement by tracking angular velocity
            previous_direction = self.temporal.pattern_history[-1][1][:2] if self.temporal.pattern_history else None
            movement_reward = 1.0
            if previous_direction is not None:
                previous_norm = previous_direction / (np.linalg.norm(previous_direction) + 1e-8)
                movement_alignment = np.dot(previous_norm, norm_dir_to_player)
                movement_reward = max(0.0, (movement_alignment + 1.0) / 2.0)

            # Final success score combines alignment, distance, and movement
            return 0.4 * alignment_reward + 0.4 * distance_reward + 0.2 * movement_reward

        elif self.current_strategy == Strategy.PATROL:
            # Current patrol target
            patrol_target = self.patrol_points[self.patrol_index]

            # Distance to the current patrol target
            distance_to_patrol = np.linalg.norm(self.position - patrol_target)

            # Reward for approaching the target
            approach_reward = max(0.0, 1.0 / (1.0 + distance_to_patrol))

            # Bonus reward for reaching the target
            reach_reward = 0.0
            if distance_to_patrol < 50:  # Threshold to consider "reached"
                reach_reward = 1
                # Prepare to move to the next patrol point
                self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)

            # Combine rewards
            return 0.6 * approach_reward + 0.4 * reach_reward

        # Default fallback
        return 0.0

    def _update_strategy(self, player_pos: np.ndarray):
        """Update AI strategy based on state"""
        distance = np.linalg.norm(player_pos - self.position)

        # Strategy switching logic
        if self.strategy_timer >= 2.0:  # Check every 2 seconds
            self.strategy_timer = 0.0

            # Get the most successful strategy
            best_strategy = max(self.graph.strategy_success.items(), key=lambda x: x[1])[0]

            # Add a penalty for frequent switches
            if self.current_strategy != best_strategy:
                self.graph.update_strategy_success(self.current_strategy, -0.05)

            # Strategy selection based on state
            if distance < 50:  # Too close
                new_strategy = Strategy.RETREAT
            elif distance > 400:  # Too far
                new_strategy = Strategy.PURSUE
            elif 100 < distance <= 200:  # Medium range, consider FLANK
                if np.random.random() < 0.5:  # 50% chance
                    new_strategy = Strategy.FLANK
                else:
                    new_strategy = Strategy.INTERCEPT
            elif 200 < distance <= 400:  # Long range, consider INTERCEPT or PATROL
                if np.random.random() < 0.3:  # 30% chance
                    new_strategy = Strategy.PATROL
                else:
                    if np.random.random() < 0.3:  # 30% chance
                        new_strategy = Strategy.INTERCEPT
                    else:
                        new_strategy = Strategy.FLANK
            elif self.success_rate < 0.03:  # Current strategy not working
                new_strategy = best_strategy
            else:
                new_strategy = self.current_strategy

            # Update strategy success based on distance change
            success_delta = 0.1 if distance < 100 else -0.1
            self.graph.update_strategy_success(self.current_strategy, success_delta)

            # Apply the new strategy
            self.current_strategy = new_strategy

    def pad_input(self, vector: np.ndarray) -> np.ndarray:
        """Pad the input vector with zeros to match the encoder's input dimension."""
        target_size = self.hdc.projection.shape[0]  # Ensure it matches the encoder's input_dim
        padded = np.zeros(target_size)
        padded[:len(vector)] = vector
        return padded

    def _execute_strategy(self, player_pos: np.ndarray, predicted_pos: np.ndarray, dt: float):
        """Execute current strategy"""
        target = np.copy(player_pos)
        base_speed = 200.0
        speed = base_speed

        if self.current_strategy == Strategy.PURSUE:
            print(f"Switching to PURSUE strategy at time {self.t:.2f}")
            # Direct pursuit
            target = player_pos
            speed = base_speed

        elif self.current_strategy == Strategy.INTERCEPT:
            print(f"Switching to INTERCEPT strategy at time {self.t:.2f}")
            # Move to predicted position
            # target = predicted_pos

            # Mix predicted position with player position
            # target = 0.8 * predicted_pos + 0.2 * player_pos

            # Use a prediction confidence
            prediction_confidence = self.success_rate  # Or another metric for prediction confidence
            # target = prediction_confidence * predicted_pos + (1 - prediction_confidence) * player_pos

            distance_to_predicted = np.linalg.norm(predicted_pos - self.position)
            distance_to_player = np.linalg.norm(player_pos - self.position)
            weight = distance_to_predicted / (distance_to_predicted + distance_to_player + 1e-8)

            # target = weight * predicted_pos + (1 - weight) * player_pos
            target = prediction_confidence * weight * predicted_pos + (1 - weight) * player_pos

            speed = base_speed * 1.2

        elif self.current_strategy == Strategy.FLANK:
            print(f"Switching to FLANK strategy at time {self.t:.2f}")
            # Calculate flanking position
            direction = player_pos - self.position
            perpendicular = np.array([-direction[1], direction[0]])
            perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)
            self.flank_offset = perpendicular * 100.0
            target = player_pos + self.flank_offset
            speed = base_speed * 1.1

        elif self.current_strategy == Strategy.RETREAT:
            print(f"Switching to RETREAT strategy at time {self.t:.2f}")
            # Move away from player
            direction = self.position - player_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            target = self.position + direction * 100.0
            speed = base_speed * 0.8

        elif self.current_strategy == Strategy.PATROL:
            print(f"Switching to PATROL strategy at time {self.t:.2f}")
            # Move between patrol points
            target = self.patrol_points[self.patrol_index]
            if np.linalg.norm(self.position - target) < 20:
                self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
            speed = base_speed * 0.7

        # Calculate movement
        direction = target - self.position
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance
            desired_velocity = direction * speed

            # Pad `direction` and calculate temporal influence
            padded_direction = self.pad_input(direction)
            temporal_influence = np.sum(self.temporal.process(self.t, self.hdc.encode(padded_direction))) * 0.1
            desired_velocity *= (1.0 + np.clip(temporal_influence, -0.2, 0.2))  # Limit influence to a small range

            # Smooth velocity change
            self.velocity += (desired_velocity - self.velocity) * dt * 2
        else:
            self.velocity = np.zeros_like(self.velocity)  # Stop movement if no direction

        # Update position
        self.position += self.velocity * dt
        self.position = np.clip(self.position, [0, 0], [800, 600])  # Ensure in bounds

        # Debugging
        if not np.isfinite(self.position).all():
            print(f"Invalid AI Position: {self.position}")
        if not np.isfinite(self.velocity).all():
            print(f"Invalid Velocity: {self.velocity}")


class Game:
    """Enhanced game visualization with graph drawing"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Enhanced UCS Adaptive Game AI")

        self.config = UCSConfig()
        self.ai = AdaptiveGameAI(self.config)

        self.player_pos = np.array([400.0, 300.0])
        self.predicted_pos = self.ai.predicted_pos
        self.running = True
        self.clock = pygame.time.Clock()

        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)

        # Graph visualization setup
        self.node_positions = {}
        self.font = pygame.font.Font(None, 24)

    def handle_events(self):
        """Handle game events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEMOTION:
                self.player_pos = np.array(event.pos)

    def update(self, dt):
        """Update game state"""
        self.ai.update(self.player_pos, dt)
        self.predicted_pos = self.ai.predicted_pos

        # Update graph node positions
        if len(self.ai.graph.nodes) > 0:
            self.node_positions = {
                i: (np.random.randint(100, 700), np.random.randint(100, 500))
                for i in range(len(self.ai.graph.nodes))
            }

    def draw_graph(self):
        """Draw the dynamic graph"""
        if len(self.node_positions) == 0 or self.ai.graph.weights.shape[0] == 0:
            return

        # Draw edges
        for i, j in zip(*self.ai.graph.weights.nonzero()):
            start_pos = self.node_positions[i]
            end_pos = self.node_positions[j]
            weight = self.ai.graph.weights[i, j]
            pygame.draw.line(
                self.screen, self.GREEN, start_pos, end_pos, max(1, int(weight * 10))
            )

        # Draw nodes
        for idx, pos in self.node_positions.items():
            radius = 5  # Fixed radius for simplicity
            pygame.draw.circle(self.screen, self.BLUE, pos, radius)

    def draw(self):
        """Draw game state"""
        self.screen.fill(self.WHITE)

        # Draw player
        pygame.draw.circle(self.screen, self.BLUE, tuple(self.player_pos.astype(int)), 10)

        # Draw AI agent
        pygame.draw.circle(self.screen, self.RED, tuple(self.ai.position.astype(int)), 10)

        # Draw velocity vector
        end_pos = self.ai.position + self.ai.velocity * 0.5
        pygame.draw.line(self.screen, self.BLACK, tuple(self.ai.position.astype(int)), tuple(end_pos.astype(int)), 2)

        # Draw patrol points if applicable
        if self.ai.current_strategy == Strategy.PATROL:
            for point in self.ai.patrol_points:
                pygame.draw.circle(self.screen, self.GREEN, tuple(point.astype(int)), 5)

        # Draw predicted position
        pygame.draw.circle(self.screen, (0, 255, 255), tuple(self.predicted_pos.astype(int)), 5)

        # Display AI strategy
        strategy_text = self.font.render(f"Strategy: {self.ai.current_strategy.name}", True, self.BLACK)
        self.screen.blit(strategy_text, (10, 10))

        # Display success rate
        success_text = self.font.render(f"Success Rate: {self.ai.success_rate:.2f}", True, self.BLACK)
        self.screen.blit(success_text, (10, 30))

        # Draw the dynamic graph
        self.draw_graph()

        pygame.display.flip()

    def run(self):
        """Main game loop"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Delta time in seconds

            self.handle_events()
            self.update(dt)
            self.draw()

        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.run()
