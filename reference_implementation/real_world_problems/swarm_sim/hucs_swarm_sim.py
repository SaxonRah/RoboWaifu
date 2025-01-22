import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import scipy.sparse as sparse


@dataclass
class HybridConfig:
    """Configuration for hybrid UCS system"""
    hd_dimension: int = 1000  # Reduced from 10000 for visualization
    input_dimension: int = 4  # x, y, velocity_x, velocity_y
    temporal_window: float = 1.0
    decay_rate: float = 0.1
    learning_rate: float = 0.01
    max_weight: float = 1.0
    reg_lambda: float = 0.001
    nn_hidden_dim: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralHDCEncoder(nn.Module):
    """Enhanced HDC encoder with neural preprocessing"""

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        # Neural preprocessing
        self.preprocessor = nn.Sequential(
            nn.Linear(config.input_dimension, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, config.input_dimension)
        ).to(config.device)

        # Random projection matrix for HDC
        self.projection = torch.randn(config.input_dimension, config.hd_dimension).to(config.device)
        self.projection /= torch.norm(self.projection, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Neural preprocessing
        enhanced_x = self.preprocessor(x)
        # HDC encoding
        hd_vector = torch.tanh(enhanced_x @ self.projection)
        return hd_vector


class NeuralTemporalProcessor(nn.Module):
    """Enhanced temporal processor with neural attention"""

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.time_buffer: List[Tuple[float, torch.Tensor]] = []

        # Neural attention for temporal weighting
        self.attention = nn.Sequential(
            nn.Linear(config.hd_dimension, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, 1),
            nn.Sigmoid()
        ).to(config.device)

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        # Update buffer
        self.time_buffer.append((t, x))
        cutoff_time = t - self.config.temporal_window
        self.time_buffer = [(t_i, x_i) for t_i, x_i in self.time_buffer if t_i > cutoff_time]

        # Apply neural attention to temporal samples
        if len(self.time_buffer) > 1:
            times, samples = zip(*self.time_buffer)
            samples_tensor = torch.stack(samples)
            attention_weights = self.attention(samples_tensor)
            weighted_result = (samples_tensor * attention_weights).mean(dim=0)

            # Combine with exponential decay
            alpha = 0.7
            decay_result = torch.zeros_like(x)
            for t_i, x_i in self.time_buffer[:-1]:
                weight = np.exp(-self.config.decay_rate * (t - t_i))
                decay_result += weight * (x_i - x)

            return alpha * decay_result + (1 - alpha) * weighted_result

        return x


class HybridDynamicGraph:
    """Enhanced dynamic graph with neural edge prediction"""

    def __init__(self, config: HybridConfig, initial_size: int = 5):
        self.config = config
        # Initialize with size that matches number of robots
        self.weights = sparse.lil_matrix((initial_size, initial_size))
        self.nodes = list(range(initial_size))
        self.node_features = {}

        # Neural edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(config.hd_dimension * 2, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, 1),
            nn.Sigmoid()
        ).to(config.device)

        # Initialize weights to avoid overlapping
        for i in range(initial_size):
            for j in range(initial_size):
                if i != j:
                    self.weights[i, j] = 0.5  # Default medium repulsion

    def update_weights(self, features: Dict[int, torch.Tensor]) -> None:
        n = len(self.nodes)
        if n == 0:
            return

        if self.weights.shape != (n, n):
            self.weights = sparse.lil_matrix((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                f_i = features[i]
                f_j = features[j]

                # Concatenate features for edge prediction
                combined = torch.cat([f_i, f_j])

                # Predict refined edge weight
                with torch.no_grad():
                    neural_weight = self.edge_predictor(combined.unsqueeze(0)).item()

                # Compute baseline similarity
                sim = F.cosine_similarity(f_i.unsqueeze(0), f_j.unsqueeze(0)).item()

                # Combine with neural prediction
                alpha = 0.7
                hybrid_weight = alpha * sim + (1 - alpha) * neural_weight
                hybrid_weight = min(self.config.max_weight, max(0, hybrid_weight))

                self.weights[i, j] = self.weights[j, i] = hybrid_weight


class Robot:
    def __init__(self, x: float, y: float, config: HybridConfig):
        self.x = x
        self.y = y
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.sensor_range = 100
        self.color = (50, 150, 250)
        self.config = config

        # Hybrid UCS components
        self.hdc_encoder = NeuralHDCEncoder(config)
        self.temporal_processor = NeuralTemporalProcessor(config)
        self.t = 0.0

    def get_state_tensor(self) -> torch.Tensor:
        return torch.tensor([self.x, self.y, self.velocity_x, self.velocity_y],
                            dtype=torch.float32).to(self.config.device)

    def process_state(self, t: float) -> torch.Tensor:
        # Get current state
        state = self.get_state_tensor()

        # HDC encoding
        hd_state = self.hdc_encoder(state.unsqueeze(0))

        # Temporal processing
        temporal_state = self.temporal_processor(t, hd_state)

        return temporal_state

    def update(self, target_x: float, target_y: float, neighbors: List['Robot'], weights: sparse.lil_matrix,
               dt: float = 0.1):
        self.t += dt

        # Target attraction
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        if distance > 0:
            speed = 100  # pixels per second
            target_force_x = (dx / distance) * speed
            target_force_y = (dy / distance) * speed
        else:
            target_force_x = target_force_y = 0

        # Repulsion from neighbors using graph weights
        repulsion_x = 0
        repulsion_y = 0
        min_distance = 50  # Increased minimum separation distance

        my_idx = neighbors.index(self)
        for i, neighbor in enumerate(neighbors):
            if neighbor != self:
                try:
                    # Get weight from graph safely
                    weight = weights[my_idx, i]

                    # Calculate distance to neighbor
                    dx_n = neighbor.x - self.x
                    dy_n = neighbor.y - self.y
                    dist = np.sqrt(dx_n ** 2 + dy_n ** 2)

                    if dist < min_distance:
                        # Stronger repulsion for higher graph weights
                        repulsion_strength = (1 + weight) * (min_distance - dist) / min_distance * speed * 1.5
                        if dist > 0:  # Avoid division by zero
                            repulsion_x -= (dx_n / dist) * repulsion_strength
                            repulsion_y -= (dy_n / dist) * repulsion_strength
                except IndexError:
                    continue  # Skip if weight access fails

        # Combine forces
        self.velocity_x = target_force_x + repulsion_x
        self.velocity_y = target_force_y + repulsion_y

        # Update position
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt

        # Boundary conditions
        self.x = np.clip(self.x, 0, 800)
        self.y = np.clip(self.y, 0, 600)


class SwarmSimulation:
    def __init__(self, width: int = 800, height: int = 600, num_robots: int = 5):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Hybrid UCS Swarm Simulation")

        # Initialize Hybrid UCS components
        self.config = HybridConfig()
        self.robots: List[Robot] = []
        self.graph = HybridDynamicGraph(self.config, initial_size=num_robots)
        self.clock = pygame.time.Clock()
        self.running = True
        self.target_x = width // 2
        self.target_y = height // 2
        self.t = 0.0

        # Initialize robots with proper spacing
        for i in range(num_robots):
            # Position robots in a circle initially
            angle = (2 * np.pi * i) / num_robots
            radius = 100  # Initial circle radius
            x = width / 2 + radius * np.cos(angle)
            y = height / 2 + radius * np.sin(angle)
            self.robots.append(Robot(x, y, self.config))

    def update_graph(self):
        """Update graph using Hybrid UCS"""
        # Process states for all robots
        features = {}
        for i, robot in enumerate(self.robots):
            state = robot.process_state(self.t)
            features[i] = state.squeeze()

        # Update graph weights using neural prediction
        self.graph.update_weights(features)

    def draw(self):
        """Draw the simulation state"""
        self.screen.fill((0, 0, 0))

        # Draw connections between robots based on graph weights
        for i, robot1 in enumerate(self.robots):
            for j, robot2 in enumerate(self.robots):
                if i < j:
                    weight = self.graph.weights[i, j]
                    if weight > 0.5:  # Only draw strong connections
                        alpha = int(255 * weight)
                        pygame.draw.line(self.screen, (0, 255, 0, alpha),
                                         (robot1.x, robot1.y),
                                         (robot2.x, robot2.y), 1)

        # Draw robots
        for robot in self.robots:
            # Draw sensor range
            pygame.draw.circle(self.screen, (30, 30, 30),
                               (int(robot.x), int(robot.y)),
                               robot.sensor_range, 1)

            # Draw robot
            pygame.draw.circle(self.screen, robot.color,
                               (int(robot.x), int(robot.y)), 10)

        # Draw target
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (int(self.target_x), int(self.target_y)), 15, 2)

        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.target_x, self.target_y = pygame.mouse.get_pos()

    def run(self):
        """Main simulation loop"""
        while self.running:
            dt = 1 / 60.0
            self.t += dt

            self.handle_events()

            # Update robot positions with collision avoidance
            for robot in self.robots:
                robot.update(self.target_x, self.target_y, self.robots, self.graph.weights, dt)

            # Update graph using Hybrid UCS
            self.update_graph()

            # Draw current state
            self.draw()

            # Control frame rate
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    simulation = SwarmSimulation()
    simulation.run()