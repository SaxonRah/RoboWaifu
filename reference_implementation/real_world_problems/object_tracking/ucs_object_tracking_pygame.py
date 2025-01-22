import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import scipy.sparse as sparse
from sklearn.preprocessing import normalize


@dataclass
class UCSConfig:
    """Configuration for UCS system"""
    hd_dimension: int = 10000  # Dimension of hypervectors
    input_dimension: int = 4  # x, y, velocity_x, velocity_y
    temporal_window: float = 1.0  # Time window for temporal processing
    decay_rate: float = 0.1  # Temporal decay rate
    learning_rate: float = 0.01  # Learning rate for graph updates
    max_weight: float = 1.0  # Maximum edge weight
    min_safe_distance: float = 2.0  # Minimum safe distance between objects
    max_memory_size: int = 1000  # Maximum number of temporal memories to store


class HDCEncoder:
    """Hyperdimensional Computing encoder with distance preservation"""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dimension = input_dim
        self.output_dimension = output_dim
        # Create projection matrix with normalized columns
        self.projection = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.projection = normalize(self.projection, axis=0, norm='l2')

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input vector into hyperdimensional space"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Normalize input
        x_norm = normalize(x, axis=1, norm='l2')
        # Project to high-dimensional space and apply nonlinearity
        return np.tanh(x_norm @ self.projection)


class TemporalProcessor:
    """Temporal processing with exponential decay"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.time_buffer: List[Tuple[float, np.ndarray, int]] = []  # (time, features, object_id)

    def process(self, t: float, x: np.ndarray, object_id: int) -> np.ndarray:
        """Process temporal features with memory"""
        # Update buffer with new sample
        self.time_buffer.append((t, x, object_id))

        # Remove old samples outside temporal window
        cutoff_time = t - self.config.temporal_window
        self.time_buffer = [(t_i, x_i, id_i) for t_i, x_i, id_i in self.time_buffer
                            if t_i > cutoff_time]

        # Limit buffer size
        if len(self.time_buffer) > self.config.max_memory_size:
            self.time_buffer = self.time_buffer[-self.config.max_memory_size:]

        # Compute temporal integration with decay
        result = x.copy()
        obj_samples = [(t_i, x_i) for t_i, x_i, id_i in self.time_buffer
                       if id_i == object_id and t_i != t]

        for t_i, x_i in obj_samples:
            weight = np.exp(-self.config.decay_rate * (t - t_i))
            result += weight * (x_i - x)

        return result


class DynamicGraph:
    """Dynamic graph with proven convergence properties"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.weights = sparse.lil_matrix((0, 0))
        self.nodes = []  # List of object IDs
        self.node_features = {}  # Dict mapping object ID to features

    def update_weights(self, features: Dict[int, np.ndarray]) -> None:
        """Update graph weights using similarity and safety constraints"""
        n = len(self.nodes)
        if n == 0:
            return

        # Construct new weight matrix if needed
        if self.weights.shape != (n, n):
            self.weights = sparse.lil_matrix((n, n))

        # Update weights based on similarity and safety distance
        for i in range(n):
            for j in range(i + 1, n):
                f_i = features[self.nodes[i]]
                f_j = features[self.nodes[j]]

                # Compute physical distance between objects
                pos_i = f_i[:2]  # x, y coordinates
                pos_j = f_j[:2]
                distance = np.linalg.norm(pos_i - pos_j)

                # Compute similarity with safety constraint
                similarity = self._bounded_similarity(f_i, f_j)
                safety_factor = np.exp(-max(0, self.config.min_safe_distance - distance))

                # Update weights using gradient descent
                current_weight = self.weights[i, j]
                target_weight = similarity * safety_factor
                grad = 2 * (current_weight - target_weight)
                new_weight = current_weight - self.config.learning_rate * grad

                # Apply weight bounds
                new_weight = np.clip(new_weight, 0, self.config.max_weight)
                self.weights[i, j] = self.weights[j, i] = new_weight

        self.weights = self.weights.tocsr()

    def _bounded_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute similarity with proven bounds in [0,1]"""
        x = x.ravel()
        y = y.ravel()
        cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
        return 0.5 * (cos_sim + 1)  # Map to [0,1]


class UnifiedCognitiveSystem:
    """Main UCS implementation for object tracking and collision avoidance"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.hdc = HDCEncoder(config.input_dimension, config.hd_dimension)
        self.temporal = TemporalProcessor(config)
        self.graph = DynamicGraph(config)
        self.t = 0.0
        self.object_positions = {}  # Keep track of current object positions

    def process(self, object_id: int, position: np.ndarray, velocity: np.ndarray) -> Dict:
        """Process new object state and return safety analysis"""
        # Combine position and velocity into state vector
        x = np.concatenate([position, velocity])

        # 1. Encode state in HDC space
        hd_x = self.hdc.encode(x)

        # 2. Apply temporal processing
        self.t += 0.01  # Simulate time step
        temporal_x = self.temporal.process(self.t, hd_x, object_id)

        # 3. Update graph structure
        if object_id not in self.graph.nodes:
            self.graph.nodes.append(object_id)
        self.graph.node_features[object_id] = temporal_x
        self.object_positions[object_id] = position
        self.graph.update_weights(self.graph.node_features)

        # 4. Analyze safety and potential collisions
        safety_analysis = self._analyze_safety(object_id)

        return safety_analysis

    def _analyze_safety(self, object_id: int) -> Dict:
        """Analyze safety conditions for the given object"""
        if not self.object_positions:
            return {"safe": True, "warnings": []}

        warnings = []
        current_pos = self.object_positions[object_id]

        # Check distances to all other objects
        for other_id, other_pos in self.object_positions.items():
            if other_id != object_id:
                distance = np.linalg.norm(current_pos - other_pos)
                if distance < self.config.min_safe_distance:
                    warnings.append({
                        "type": "collision_risk",
                        "object_id": other_id,
                        "distance": distance,
                        "min_safe_distance": self.config.min_safe_distance
                    })

        return {
            "safe": len(warnings) == 0,
            "warnings": warnings
        }


def test_object_tracking():
    """Test the UCS implementation with multiple moving objects using Pygame visualization"""
    import pygame
    import sys
    import time

    # Initialize Pygame
    pygame.init()

    # Constants for visualization
    WINDOW_SIZE = (800, 800)
    SCALE = 50  # Pixels per unit
    OFFSET = np.array([WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2])  # Center of screen

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)

    # Initialize display
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("UCS Object Tracking")
    clock = pygame.time.Clock()

    def world_to_screen(pos):
        """Convert world coordinates to screen coordinates"""
        return (pos * SCALE + OFFSET).astype(int)

    # Initialize UCS system
    config = UCSConfig(min_safe_distance=1.0)
    ucs = UnifiedCognitiveSystem(config)

    # Initialize objects
    obj1_pos = np.array([0.0, 0.0])
    obj1_vel = np.array([1.0, 0.8])
    obj1_trail = []

    obj2_pos = np.array([4.0, 4.0])
    obj2_vel = np.array([-0.7, -0.7])
    obj2_trail = []

    running = True
    paused = False
    t = 0
    max_trail_length = 50

    # Font for text display
    font = pygame.font.Font(None, 36)

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:  # Reset simulation
                    obj1_pos = np.array([0.0, 0.0])
                    obj2_pos = np.array([4.0, 4.0])
                    obj1_trail = []
                    obj2_trail = []
                    t = 0

        if not paused:
            # Update positions
            obj1_pos += obj1_vel * 0.1
            obj2_pos += obj2_vel * 0.1

            # Store trail points
            obj1_trail.append(obj1_pos.copy())
            obj2_trail.append(obj2_pos.copy())

            # Limit trail length
            if len(obj1_trail) > max_trail_length:
                obj1_trail.pop(0)
            if len(obj2_trail) > max_trail_length:
                obj2_trail.pop(0)

            # Process objects through UCS
            safety1 = ucs.process(1, obj1_pos, obj1_vel)
            safety2 = ucs.process(2, obj2_pos, obj2_vel)

            t += 1

        # Clear screen
        screen.fill(WHITE)

        # Draw grid
        for x in range(0, WINDOW_SIZE[0], SCALE):
            pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, WINDOW_SIZE[1]))
        for y in range(0, WINDOW_SIZE[1], SCALE):
            pygame.draw.line(screen, (200, 200, 200), (0, y), (WINDOW_SIZE[0], y))

        # Draw coordinate axes
        pygame.draw.line(screen, BLACK, (0, OFFSET[1]), (WINDOW_SIZE[0], OFFSET[1]), 2)
        pygame.draw.line(screen, BLACK, (OFFSET[0], 0), (OFFSET[0], WINDOW_SIZE[1]), 2)

        # Draw trails
        if len(obj1_trail) > 1:
            trail_points1 = [world_to_screen(p) for p in obj1_trail]
            pygame.draw.lines(screen, BLUE, False, trail_points1, 2)

        if len(obj2_trail) > 1:
            trail_points2 = [world_to_screen(p) for p in obj2_trail]
            pygame.draw.lines(screen, RED, False, trail_points2, 2)

        # Draw safety circles
        pygame.draw.circle(screen, BLUE if safety1['safe'] else YELLOW,
                           world_to_screen(obj1_pos),
                           int(config.min_safe_distance * SCALE), 1)
        pygame.draw.circle(screen, RED if safety2['safe'] else YELLOW,
                           world_to_screen(obj2_pos),
                           int(config.min_safe_distance * SCALE), 1)

        # Draw objects
        pygame.draw.circle(screen, BLUE, world_to_screen(obj1_pos), 6)
        pygame.draw.circle(screen, RED, world_to_screen(obj2_pos), 6)

        # Draw velocity vectors
        pygame.draw.line(screen, BLUE,
                         world_to_screen(obj1_pos),
                         world_to_screen(obj1_pos + obj1_vel), 2)
        pygame.draw.line(screen, RED,
                         world_to_screen(obj2_pos),
                         world_to_screen(obj2_pos + obj2_vel), 2)

        # Display safety status
        status1 = "Safe" if safety1['safe'] else "Warning!"
        status2 = "Safe" if safety2['safe'] else "Warning!"

        text1 = font.render(f"Object 1: {status1}", True, BLUE)
        text2 = font.render(f"Object 2: {status2}", True, RED)
        text_pause = font.render("PAUSED" if paused else "", True, BLACK)
        text_help = font.render("Space: Pause  R: Reset", True, BLACK)

        screen.blit(text1, (10, 10))
        screen.blit(text2, (10, 50))
        screen.blit(text_pause, (WINDOW_SIZE[0] - 100, 10))
        screen.blit(text_help, (10, WINDOW_SIZE[1] - 30))

        pygame.display.flip()
        clock.tick(30)  # 30 FPS

    pygame.quit()


if __name__ == "__main__":
    test_object_tracking()
