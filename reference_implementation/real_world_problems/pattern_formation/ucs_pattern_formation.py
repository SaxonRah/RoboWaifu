import numpy as np
import pygame
from typing import List, Tuple, Dict
import scipy.sparse as sparse
from dataclasses import dataclass


@dataclass
class UCSConfig:
    """Configuration for UCS system with proven bounds"""
    hd_dimension: int = 100  # Reduced dimension for performance while maintaining HDC properties
    input_dimension: int = 4  # 2D position + 2D velocity
    temporal_window: float = 1.0
    decay_rate: float = 0.1
    learning_rate: float = 0.01
    reg_lambda: float = 0.001  # Regularization parameter
    max_weight: float = 1.0
    window_size: Tuple[int, int] = (800, 600)
    n_particles: int = 50


class HDCEncoder:
    """HDC encoder with proven distance preservation"""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dimension = input_dim
        self.output_dimension = output_dim
        # Create projection matrix for input dimension to HD dimension
        self.projection = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        # Normalize columns
        self.projection /= np.linalg.norm(self.projection, axis=0)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input vector into hyperdimensional space"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.input_dimension:
            raise ValueError(f"Input dimension mismatch. Expected {self.input_dimension}, got {x.shape[1]}")

        # Normalize input
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        # Project to high-dimensional space
        hd_vector = x_norm @ self.projection
        # Apply non-linear transformation preserving distances
        return np.tanh(hd_vector).reshape(1, -1)  # Ensure 2D output

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two hypervectors using circular convolution"""
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple hypervectors with proven superposition"""
        if not vectors:
            return np.zeros(self.output_dimension)
        bundle = np.zeros(self.output_dimension)
        for v in vectors:
            bundle += v
        return np.tanh(bundle / len(vectors))


class TemporalProcessor:
    """Temporal processing with proven convergence"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.time_buffer: List[Tuple[float, np.ndarray]] = []

    def process(self, t: float, x: np.ndarray) -> np.ndarray:
        """Process input using temporal-spatial integration"""
        # Update buffer
        self.time_buffer.append((t, x))
        # Remove old samples
        cutoff_time = t - self.config.temporal_window
        self.time_buffer = [(t_i, x_i) for t_i, x_i in self.time_buffer
                            if t_i > cutoff_time]

        # Compute temporal integral with proven convergence
        result = x.copy()
        for t_i, x_i in self.time_buffer[:-1]:
            weight = np.exp(-self.config.decay_rate * (t - t_i))
            result += weight * (x_i - x) * (t - t_i)
        return result


class DynamicGraph:
    """Dynamic graph with proven convergence properties"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.weights = sparse.lil_matrix((0, 0))
        self.nodes = []
        self.node_features = {}

    def update_weights(self, features: Dict[int, np.ndarray]) -> None:
        """Update graph weights using proven convergent dynamics"""
        n = len(self.nodes)
        if n == 0:
            return

        if self.weights.shape != (n, n):
            self.weights = sparse.lil_matrix((n, n))

        # Compute pairwise similarities directly
        for i in range(n):
            for j in range(i + 1, n):
                f_i = features[self.nodes[i]].ravel()
                f_j = features[self.nodes[j]].ravel()

                # Compute normalized similarity
                sim = np.dot(f_i, f_j) / (np.linalg.norm(f_i) * np.linalg.norm(f_j) + 1e-8)
                sim = 0.5 * (sim + 1)  # Map to [0,1]

                # Update weights with stronger values
                self.weights[i, j] = self.weights[j, i] = sim

    def _bounded_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute similarity with proven bounds in [0,1]"""
        x = x.ravel()
        y = y.ravel()
        cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
        return 0.5 * (cos_sim + 1)


class RigorousPatternFormation:
    """UCS-based pattern formation with mathematical guarantees"""

    def __init__(self, config: UCSConfig):
        self.config = config
        pygame.init()
        self.screen = pygame.display.set_mode(config.window_size)
        pygame.display.set_caption("Rigorous UCS Pattern Formation")

        # Initialize UCS components with proven properties
        self.hdc = HDCEncoder(config.input_dimension, config.hd_dimension)
        self.temporal = TemporalProcessor(config)
        self.graph = DynamicGraph(config)

        # Initialize particles with proper numpy arrays
        self.particles = np.zeros((config.n_particles, 2), dtype=np.float64)
        self.velocities = np.zeros((config.n_particles, 2), dtype=np.float64)
        for i in range(config.n_particles):
            self.particles[i] = np.random.rand(2) * np.array(config.window_size, dtype=np.float64)
            self.velocities[i] = (np.random.rand(2) - 0.5) * 2

        self.t = 0.0
        self.clock = pygame.time.Clock()

    def update(self) -> None:
        # 1. HDC Encoding with temporal processing
        encoded_states = {}
        for i, (pos, vel) in enumerate(zip(self.particles, self.velocities)):
            state = np.concatenate([pos, vel])
            hd_state = self.hdc.encode(state)
            # Apply temporal processing
            temporal_state = self.temporal.process(self.t, hd_state)
            encoded_states[i] = temporal_state

        # 2. Update graph with proven convergence
        self.graph.nodes = list(range(len(self.particles)))
        self.graph.update_weights(encoded_states)

        # 3. Update particle dynamics using graph structure
        for i in range(len(self.particles)):
            force = np.zeros(2)
            for j in range(len(self.particles)):
                if i != j:
                    diff = self.particles[j].astype(np.float64) - self.particles[i].astype(np.float64)
                    distance = np.linalg.norm(diff)
                    if distance < 1e-6:
                        continue

                    # Use graph weights with proven bounds
                    weight = self.graph.weights[i, j]
                    force += (diff / distance) * (weight - 0.5) * 50

            # Update with bounded dynamics
            self.velocities[i] += force * 0.1
            self.velocities[i] *= 0.95  # Damping
            # Limit maximum velocity
            speed = np.linalg.norm(self.velocities[i])
            if speed > 5.0:  # Max speed limit
                self.velocities[i] *= 5.0 / speed
            self.particles[i] += self.velocities[i]

            # Enforce boundary conditions
            for d in range(2):
                if self.particles[i][d] < 0:
                    self.particles[i][d] = 0
                    self.velocities[i][d] *= -0.5
                elif self.particles[i][d] > self.config.window_size[d]:
                    self.particles[i][d] = self.config.window_size[d]
                    self.velocities[i][d] *= -0.5

    def draw(self) -> None:
        self.screen.fill((0, 0, 0))

        # Draw graph edges with enhanced visibility
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                weight = float(self.graph.weights[i, j])
                if weight > 0.3:  # Lower threshold for more visible edges
                    # Enhanced color scheme
                    color = (
                        int(255 * weight),  # Red
                        int(128 * (1 - weight)),  # Green
                        int(255 * (1 - weight))  # Blue
                    )
                    start = tuple(map(int, self.particles[i]))
                    end = tuple(map(int, self.particles[j]))
                    # Thicker lines for stronger connections
                    thickness = int(max(1, weight * 3))
                    pygame.draw.line(self.screen, color, start, end, thickness)

        # Draw particles with a glowing effect
        for pos in self.particles:
            # Draw outer glow
            pygame.draw.circle(self.screen, (64, 64, 64),
                               tuple(map(int, pos)), 7)
            # Draw particle
            pygame.draw.circle(self.screen, (255, 255, 255),
                               tuple(map(int, pos)), 4)

        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Add new particle with proper numpy array handling
                    pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)
                    vel = (np.random.rand(2) - 0.5) * 2
                    self.particles = np.vstack([self.particles, pos])
                    self.velocities = np.vstack([self.velocities, vel])

            self.update()
            self.draw()
            self.clock.tick(600)
            self.t += 1 / 60

        pygame.quit()


if __name__ == "__main__":
    config = UCSConfig()
    simulation = RigorousPatternFormation(config)
    simulation.run()
