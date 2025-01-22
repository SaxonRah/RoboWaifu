import pygame
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import scipy.sparse as sparse


@dataclass
class UCSConfig:
    """Configuration for unified cognitive system"""
    hd_dimension: int = 10000
    input_dimension: int = 4  # 2 joint angles + 2 velocities
    temporal_window: float = 1.0
    decay_rate: float = 0.1
    learning_rate: float = 0.01
    max_weight: float = 1.0
    reg_lambda: float = 0.001
    nn_hidden_dim: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralHDCEncoder(nn.Module):
    """Enhanced HDC encoder with neural preprocessing"""

    def __init__(self, config: UCSConfig):
        super().__init__()
        self.config = config

        # Neural preprocessing
        self.preprocessor = nn.Sequential(
            nn.Linear(config.input_dimension, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, config.input_dimension)
        ).to(config.device)

        # Traditional HDC encoder
        self.projection = np.random.randn(config.input_dimension, config.hd_dimension) / np.sqrt(config.input_dimension)
        self.projection /= np.linalg.norm(self.projection, axis=0)

    def forward(self, x: torch.Tensor) -> np.ndarray:
        # Neural preprocessing
        enhanced_x = self.preprocessor(x)
        # Convert to numpy for HDC encoding
        enhanced_x_np = enhanced_x.detach().cpu().numpy()
        # HDC encoding
        x_norm = enhanced_x_np / (np.linalg.norm(enhanced_x_np) + 1e-8)
        hd_vector = np.tanh(x_norm @ self.projection)
        return hd_vector


class NeuralTemporalProcessor(nn.Module):
    """Enhanced temporal processor with neural attention"""

    def __init__(self, config: UCSConfig):
        super().__init__()
        self.config = config
        self.time_buffer = []

        # Neural attention for temporal weighting
        self.attention = nn.Sequential(
            nn.Linear(config.hd_dimension, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, 1),
            nn.Sigmoid()
        ).to(config.device)

    def forward(self, t: float, x: np.ndarray) -> np.ndarray:
        # Update buffer
        self.time_buffer.append((t, x))
        cutoff_time = t - self.config.temporal_window
        self.time_buffer = [(t_i, x_i) for t_i, x_i in self.time_buffer if t_i > cutoff_time]

        if len(self.time_buffer) > 1:
            times, samples = zip(*self.time_buffer)
            samples_tensor = torch.FloatTensor(np.array(samples)).to(self.config.device)
            # Apply neural attention
            attention_weights = self.attention(samples_tensor)
            weighted_samples = (samples_tensor * attention_weights).mean(dim=0)
            attended_result = weighted_samples.detach().cpu().numpy()

            # Combine with exponential decay
            decay_weights = np.array([np.exp(-self.config.decay_rate * (t - t_i)) for t_i in times])
            decay_result = np.average(samples, axis=0, weights=decay_weights)

            # Final result combines both
            alpha = 0.7
            return alpha * decay_result + (1 - alpha) * attended_result

        return x


class HybridDynamicGraph:
    """Enhanced dynamic graph with neural edge prediction"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.weights = sparse.lil_matrix((0, 0))
        self.nodes = []
        self.node_features = {}

        # Neural edge predictor
        feature_dim = config.hd_dimension
        self.edge_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, 1),
            nn.Sigmoid()
        ).to(config.device)

    def update_weights(self, features: Dict[int, np.ndarray]) -> None:
        """Update graph weights using both similarity and neural prediction"""
        n = len(self.nodes)
        if n == 0:
            return

        if self.weights.shape != (n, n):
            self.weights = sparse.lil_matrix((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                f_i = features[i]
                f_j = features[j]

                # Traditional similarity
                sim = np.dot(f_i, f_j) / (np.linalg.norm(f_i) * np.linalg.norm(f_j) + 1e-8)
                sim = 0.5 * (sim + 1)  # Map to [0,1]

                # Neural edge prediction
                combined = torch.FloatTensor(np.concatenate([f_i, f_j])).to(self.config.device)
                neural_weight = self.edge_predictor(combined.unsqueeze(0)).item()

                # Combine predictions
                alpha = 0.7
                hybrid_weight = alpha * sim + (1 - alpha) * neural_weight

                # Update weight matrix
                self.weights[i, j] = self.weights[j, i] = hybrid_weight

        self.weights = self.weights.tocsr()


class RoboticArm:
    def __init__(self, base_pos: Tuple[int, int], lengths: List[float]):
        self.base_pos = base_pos
        self.lengths = lengths
        self.angles = [0.0] * len(lengths)
        self.velocities = [0.0] * len(lengths)
        self.last_angles = self.angles.copy()
        self.dt = 1 / 60.0  # For velocity calculation

    def get_state(self) -> np.ndarray:
        """Get full state: angles and velocities"""
        self.velocities = [(a - la) / self.dt for a, la in zip(self.angles, self.last_angles)]
        self.last_angles = self.angles.copy()
        return np.array(self.angles + self.velocities)

    def forward_kinematics(self, angles: Optional[List[float]] = None) -> List[Tuple[float, float]]:
        if angles is None:
            angles = self.angles

        points = [self.base_pos]
        current_point = np.array(self.base_pos)
        current_angle = 0

        for i, length in enumerate(self.lengths):
            current_angle += angles[i]
            dx = length * math.cos(current_angle)
            dy = length * math.sin(current_angle)
            current_point = current_point + np.array([dx, dy])
            points.append(tuple(current_point))

        return points

    def inverse_kinematics(self, target_x: float, target_y: float) -> None:
        x = target_x - self.base_pos[0]
        y = target_y - self.base_pos[1]

        l1, l2 = self.lengths
        d = math.sqrt(x * x + y * y)

        if d > sum(self.lengths):
            angle_to_target = math.atan2(y, x)
            self.angles = [angle_to_target, 0]
            return

        cos_a2 = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        cos_a2 = np.clip(cos_a2, -1.0, 1.0)
        a2 = math.acos(cos_a2)

        a1 = math.atan2(y, x) - math.atan2(l2 * math.sin(a2), l1 + l2 * math.cos(a2))

        self.angles = [a1, a2]


class UnifiedCognitiveSystem:
    """Complete UCS implementation for arm state estimation"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.encoder = NeuralHDCEncoder(config)
        self.temporal = NeuralTemporalProcessor(config)
        self.graph = HybridDynamicGraph(config)
        self.t = 0.0

        # Additional neural processing for final output
        self.decoder = nn.Sequential(
            nn.Linear(config.hd_dimension, 256),  # Process HD vector directly
            nn.ReLU(),
            nn.Linear(256, config.input_dimension)
        ).to(config.device)

    def process(self, state: np.ndarray) -> np.ndarray:
        """Process input through complete UCS pipeline"""
        # Convert input to tensor
        state_tensor = torch.FloatTensor(state).to(self.config.device)

        # 1. Enhanced HDC encoding
        hd_state = self.encoder(state_tensor)

        # 2. Enhanced temporal processing
        self.t += self.config.decay_rate
        temporal_state = self.temporal(self.t, hd_state)

        # 3. Update graph
        node_id = len(self.graph.nodes)
        self.graph.nodes.append(node_id)
        self.graph.node_features[node_id] = temporal_state
        self.graph.update_weights(self.graph.node_features)

        # 4. Graph embedding
        if self.graph.weights.shape[0] > 0:
            laplacian = sparse.eye(self.graph.weights.shape[0]) - self.graph.weights
            try:
                eigenvals, eigenvects = sparse.linalg.eigs(laplacian, k=min(4, laplacian.shape[0] - 1), which='SM')
                embedding = np.real(eigenvects[:, 1:])  # Skip first trivial eigenvector
            except:
                embedding = np.zeros((1, 4))
        else:
            embedding = np.zeros((1, 4))

        # 5. Process temporal state directly
        features_tensor = torch.FloatTensor(temporal_state).to(self.config.device)

        # 6. Decode to state estimate
        estimated_state = self.decoder(features_tensor).detach().cpu().numpy()

        return estimated_state


class ArmEstimationSimulation:
    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Full UCS Robot Arm State Estimation")

        # Initialize components
        self.arm = RoboticArm((width // 2, height // 2), [100, 80])
        self.config = UCSConfig()
        self.ucs = UnifiedCognitiveSystem(self.config)

        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.BLACK = (0, 0, 0)

    def draw_arm(self, points: List[Tuple[float, float]], color: Tuple[int, int, int]):
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            pygame.draw.line(self.screen, color, start, end, 5)
            pygame.draw.circle(self.screen, color, (int(start[0]), int(start[1])), 8)
        pygame.draw.circle(self.screen, color, (int(points[-1][0]), int(points[-1][1])), 8)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        mouse_pos = (self.width // 2, self.height // 2)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    mouse_pos = event.pos

            # Update arm position based on mouse
            self.arm.inverse_kinematics(mouse_pos[0], mouse_pos[1])

            # Get current state and run through UCS
            current_state = self.arm.get_state()
            estimated_state = self.ucs.process(current_state)

            # Separate angles and velocities
            estimated_angles = estimated_state[:2]
            estimated_velocities = estimated_state[2:]

            # Clear screen
            self.screen.fill(self.WHITE)

            # Draw target position
            pygame.draw.circle(self.screen, self.GREEN, mouse_pos, 10)

            # Draw actual arm position (blue)
            actual_points = self.arm.forward_kinematics(self.arm.angles)
            self.draw_arm(actual_points, self.BLUE)

            # Draw estimated arm position (red)
            estimated_points = self.arm.forward_kinematics(estimated_angles)
            self.draw_arm(estimated_points, self.RED)

            # Draw debug info
            font = pygame.font.Font(None, 24)
            debug_info = [
                f"Actual angles: [{self.arm.angles[0]:.2f}, {self.arm.angles[1]:.2f}]",
                f"Est. angles: [{estimated_angles[0]:.2f}, {estimated_angles[1]:.2f}]",
                f"Actual velocities: [{self.arm.velocities[0]:.2f}, {self.arm.velocities[1]:.2f}]",
                f"Est. velocities: [{estimated_velocities[0]:.2f}, {estimated_velocities[1]:.2f}]",
                f"Time step: {self.ucs.t:.2f}"
            ]

            # Draw right-side debug panel
            panel_x = self.width - 300
            for i, text in enumerate(debug_info):
                text_surface = font.render(text, True, self.BLACK)
                self.screen.blit(text_surface, (panel_x, 20 + i * 25))

            # Draw legend
            legend_items = [
                ("Blue: Actual Arm", self.BLUE),
                ("Red: UCS Estimate", self.RED),
                ("Green: Target", self.GREEN)
            ]
            for i, (text, color) in enumerate(legend_items):
                text_surface = font.render(text, True, color)
                self.screen.blit(text_surface, (10, 20 + i * 25))

            # Draw UCS component status
            ucs_info = [
                "UCS Components:",
                "- HDC Encoder: Active",
                "- Neural Temporal: Active",
                "- Dynamic Graph: Active",
                f"- Graph Nodes: {len(self.ucs.graph.nodes)}"
            ]
            for i, text in enumerate(ucs_info):
                text_surface = font.render(text, True, self.BLACK)
                self.screen.blit(text_surface, (10, self.height - 140 + i * 25))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    sim = ArmEstimationSimulation()
    sim.run()
