import pygame
import numpy as np
from dataclasses import dataclass
import scipy.sparse as sparse
from typing import Dict, List, Tuple


@dataclass
class UCSConfig:
    """Configuration parameters for UCS visualization"""
    window_width: int = 800
    window_height: int = 600
    hd_dimension: int = 100  # Reduced for visualization
    input_dimension: int = 2
    temporal_window: float = 1.0
    decay_rate: float = 0.1
    learning_rate: float = 0.01
    max_weight: float = 1.0
    reg_lambda: float = 0.001


class HDCEncoder:
    """Hyperdimensional Computing encoder"""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dimension = input_dim
        self.output_dimension = output_dim
        self.projection = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.projection /= np.linalg.norm(self.projection, axis=0)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input vector into hyperdimensional space"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        hd_vector = x_norm @ self.projection
        return np.tanh(hd_vector)


class DynamicGraph:
    """Dynamic graph with visualization support"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.weights = sparse.lil_matrix((0, 0))
        self.nodes = []
        self.node_features = {}
        self.node_positions = {}  # For visualization

    def update_weights(self, features: Dict[int, np.ndarray]) -> None:
        """Update graph weights using similarity"""
        n = len(self.nodes)
        if n == 0:
            return

        if self.weights.shape != (n, n):
            self.weights = sparse.lil_matrix((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                f_i = features[self.nodes[i]]
                f_j = features[self.nodes[j]]
                sim = self._bounded_similarity(f_i, f_j)
                # Simplified weight update to make edges more visible
                new_weight = sim  # Direct similarity-to-weight mapping
                new_weight = np.clip(new_weight, 0, self.config.max_weight)
                self.weights[i, j] = self.weights[j, i] = new_weight

    def _bounded_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute similarity with proven bounds in [0,1]"""
        x = x.ravel()
        y = y.ravel()
        cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
        return 0.5 * (cos_sim + 1)


class TemporalProcessor:
    """Temporal processor with visualization support"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.time_buffer: List[Tuple[float, np.ndarray]] = []
        self.visualization_buffer: List[Tuple[float, float]] = []  # Store x,y coordinates

    def process(self, t: float, x: np.ndarray) -> np.ndarray:
        """Process input using temporal integration"""
        self.time_buffer.append((t, x))
        cutoff_time = t - self.config.temporal_window
        self.time_buffer = [(t_i, x_i) for t_i, x_i in self.time_buffer if t_i > cutoff_time]

        result = x.copy()
        for t_i, x_i in self.time_buffer[:-1]:
            weight = np.exp(-self.config.decay_rate * (t - t_i))
            result += weight * (x_i - x) * (t - t_i)

        return result


class UCSVisualizer:
    """Pygame-based visualizer for UCS"""

    def __init__(self, config: UCSConfig):
        pygame.init()
        self.config = config
        self.screen = pygame.display.set_mode((config.window_width, config.window_height))
        pygame.display.set_caption("Unified Cognitive System Visualization")

        self.hdc = HDCEncoder(config.input_dimension, config.hd_dimension)
        self.graph = DynamicGraph(config)
        self.temporal = TemporalProcessor(config)

        self.t = 0.0
        self.running = True
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        # Store mouse positions for trail
        self.position_history = []

    def process_input(self, pos: Tuple[int, int]) -> np.ndarray:
        """Process mouse input through UCS pipeline"""
        # Store position for trail
        self.position_history.append(pos)
        if len(self.position_history) > 50:  # Keep last 50 points
            self.position_history.pop(0)

        # Normalize input coordinates
        x = np.array([pos[0] / self.config.window_width,
                      pos[1] / self.config.window_height])

        # 1. Encode input in HDC space
        hd_x = self.hdc.encode(x)

        # 2. Apply temporal processing
        self.t += 0.01
        temporal_x = self.temporal.process(self.t, hd_x)

        # 3. Update graph structure
        node_id = len(self.graph.nodes)
        self.graph.nodes.append(node_id)
        self.graph.node_features[node_id] = temporal_x
        self.graph.node_positions[node_id] = pos
        self.graph.update_weights(self.graph.node_features)

        return temporal_x

    def draw(self):
        """Draw the visualization"""
        self.screen.fill(self.WHITE)

        # Draw graph edges
        n = len(self.graph.nodes)
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    weight = float(self.graph.weights[i, j])  # Convert from matrix to float
                    if weight > 0.01:  # Lower threshold to see more edges
                        start_pos = self.graph.node_positions[self.graph.nodes[i]]
                        end_pos = self.graph.node_positions[self.graph.nodes[j]]
                        color = (int(255 * (1 - weight)), int(255 * (1 - weight)), 255)
                        pygame.draw.line(self.screen, color, start_pos, end_pos,
                                         max(1, int(weight * 5)))

        # Draw nodes
        for node_id in self.graph.nodes:
            pos = self.graph.node_positions[node_id]
            pygame.draw.circle(self.screen, self.RED, pos, 5)

        # Draw temporal trail
        if len(self.position_history) > 1:
            pygame.draw.lines(self.screen, self.GREEN, False, self.position_history, 2)

        # Draw info text
        text = self.font.render(f"Nodes: {len(self.graph.nodes)}", True, self.BLACK)
        self.screen.blit(text, (10, 10))

        # Add debug info for edges
        if n > 1:
            edge_count = sum(1 for i in range(n) for j in range(i + 1, n)
                             if float(self.graph.weights[i, j]) > 0.01)
            edge_text = self.font.render(f"Edges: {edge_count}", True, self.BLACK)
            self.screen.blit(edge_text, (10, 50))

        pygame.display.flip()

    def run(self):
        """Main loop"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    self.process_input(pos)
                elif event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[0]:  # Left button pressed
                        pos = pygame.mouse.get_pos()
                        self.process_input(pos)

            self.draw()
            self.clock.tick(60)

        pygame.quit()


def main():
    """Run the UCS visualization"""
    config = UCSConfig()
    visualizer = UCSVisualizer(config)
    visualizer.run()


if __name__ == "__main__":
    main()
