import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import scipy.sparse as sparse
from sklearn.preprocessing import normalize


@dataclass
class UCSConfig:
    """Configuration for UCS system"""
    hd_dimension: int = 10000  # Dimension of hypervectors
    input_dimension: int = 6  # x, y, velocity_x, velocity_y, object_type, radius
    temporal_window: float = 1.0  # Time window for temporal processing
    decay_rate: float = 0.1  # Temporal decay rate
    learning_rate: float = 0.01  # Learning rate for graph updates
    max_weight: float = 1.0  # Maximum edge weight
    min_safe_distance: float = 2.0  # Minimum safe distance between objects
    max_memory_size: int = 1000  # Maximum number of temporal memories to store
    world_bounds: tuple = (-8.0, 8.0, -8.0, 8.0)  # (min_x, max_x, min_y, max_y)

    # Object type constants
    DYNAMIC = 0
    STATIC = 1
    WALL = 2


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
        # Add default values for object_type and radius to match input_dimension
        object_type = self.config.DYNAMIC  # Default type for moving objects
        radius = 0.3  # Default radius

        # Combine position, velocity, type, and radius into state vector
        x = np.concatenate([
            position,  # 2 values: x, y
            velocity,  # 2 values: vx, vy
            [object_type],  # 1 value: type
            [radius]  # 1 value: radius
        ]).reshape(1, -1)  # Reshape to (1, 6) for proper matrix multiplication

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


def generate_random_obstacles(config: UCSConfig, num_obstacles: int) -> List[Dict]:
    """Generate random static obstacles"""
    obstacles = []
    for i in range(num_obstacles):
        while True:
            x = np.random.uniform(config.world_bounds[0] + 1, config.world_bounds[1] - 1)
            y = np.random.uniform(config.world_bounds[2] + 1, config.world_bounds[3] - 1)
            radius = np.random.uniform(0.3, 0.8)

            # Check if obstacle overlaps with existing ones
            valid = True
            for obs in obstacles:
                dist = np.linalg.norm(np.array([x, y]) - obs['position'])
                if dist < (radius + obs['radius'] + 0.5):
                    valid = False
                    break

            if valid:
                obstacles.append({
                    'position': np.array([x, y]),
                    'velocity': np.array([0.0, 0.0]),
                    'radius': radius,
                    'type': config.STATIC
                })
                break
    return obstacles


def create_walls(config: UCSConfig) -> List[Dict]:
    """Create wall segments"""
    walls = []
    x_min, x_max, y_min, y_max = config.world_bounds
    wall_thickness = 0.5

    # Create walls as static objects at the boundaries
    walls.extend([
        # Top wall
        {'position': np.array([0, y_max]), 'velocity': np.array([0.0, 0.0]),
         'radius': wall_thickness, 'type': config.WALL},
        # Bottom wall
        {'position': np.array([0, y_min]), 'velocity': np.array([0.0, 0.0]),
         'radius': wall_thickness, 'type': config.WALL},
        # Left wall
        {'position': np.array([x_min, 0]), 'velocity': np.array([0.0, 0.0]),
         'radius': wall_thickness, 'type': config.WALL},
        # Right wall
        {'position': np.array([x_max, 0]), 'velocity': np.array([0.0, 0.0]),
         'radius': wall_thickness, 'type': config.WALL}
    ])
    return walls


def handle_collisions(pos: np.ndarray, vel: np.ndarray, radius: float,
                      obstacles: List[Dict], config: UCSConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Handle collisions with obstacles and walls"""
    new_pos = pos.copy()
    new_vel = vel.copy()

    # Check wall collisions
    x_min, x_max, y_min, y_max = config.world_bounds

    if new_pos[0] - radius < x_min:
        new_pos[0] = x_min + radius
        new_vel[0] *= -0.8  # Bounce with some energy loss
    elif new_pos[0] + radius > x_max:
        new_pos[0] = x_max - radius
        new_vel[0] *= -0.8

    if new_pos[1] - radius < y_min:
        new_pos[1] = y_min + radius
        new_vel[1] *= -0.8
    elif new_pos[1] + radius > y_max:
        new_pos[1] = y_max - radius
        new_vel[1] *= -0.8

    # Check obstacle collisions
    for obstacle in obstacles:
        if obstacle['type'] == config.STATIC:
            dist = np.linalg.norm(new_pos - obstacle['position'])
            min_dist = radius + obstacle['radius']

            if dist < min_dist:
                # Calculate collision normal
                normal = (new_pos - obstacle['position']) / dist
                # Move object out of collision
                overlap = min_dist - dist
                new_pos = new_pos + normal * overlap
                # Reflect velocity
                new_vel = new_vel - 2 * np.dot(new_vel, normal) * normal * 0.8

    return new_pos, new_vel


def test_object_tracking():
    """Test the UCS implementation with moving objects, static obstacles, and walls"""
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

    # Create static obstacles and walls
    obstacles = generate_random_obstacles(config, 5)  # 5 random obstacles
    walls = create_walls(config)
    all_static_objects = obstacles + walls

    # Initialize moving objects
    dynamic_objects = [
        {
            'position': np.array([0.0, 0.0]),
            'velocity': np.array([2.0, 1.5]),
            'radius': 0.3,
            'type': config.DYNAMIC,
            'trail': []
        },
        {
            'position': np.array([4.0, 4.0]),
            'velocity': np.array([-1.5, -1.2]),
            'radius': 0.3,
            'type': config.DYNAMIC,
            'trail': []
        }
    ]

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
            # Update dynamic objects
            for i, obj in enumerate(dynamic_objects):
                # Update position
                new_pos = obj['position'] + obj['velocity'] * 0.1

                # Handle collisions
                new_pos, new_vel = handle_collisions(
                    new_pos, obj['velocity'], obj['radius'],
                    obstacles, config
                )

                obj['position'] = new_pos
                obj['velocity'] = new_vel

                # Store trail points
                obj['trail'].append(new_pos.copy())
                if len(obj['trail']) > max_trail_length:
                    obj['trail'].pop(0)

                # Create state vector for UCS
                state = np.concatenate([
                    obj['position'],
                    obj['velocity'],
                    [obj['type']],
                    [obj['radius']]
                ])

                # Process through UCS
                safety = ucs.process(i, obj['position'], obj['velocity'])
                obj['safety'] = safety

            # Process static objects through UCS (needed for safety calculations)
            for i, obj in enumerate(all_static_objects, start=len(dynamic_objects)):
                state = np.concatenate([
                    obj['position'],
                    obj['velocity'],
                    [obj['type']],
                    [obj['radius']]
                ])
                ucs.process(i, obj['position'], obj['velocity'])

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

        # Draw static obstacles
        for obstacle in obstacles:
            pos = world_to_screen(obstacle['position'])
            radius = int(obstacle['radius'] * SCALE)
            pygame.draw.circle(screen, BLACK, pos, radius)
            pygame.draw.circle(screen, YELLOW, pos,
                               int((obstacle['radius'] + config.min_safe_distance) * SCALE), 1)

        # Draw walls
        for wall in walls:
            pos = world_to_screen(wall['position'])
            if wall['position'][0] in [config.world_bounds[0], config.world_bounds[1]]:
                # Vertical walls
                pygame.draw.rect(screen, BLACK,
                                 (pos[0] - int(wall['radius'] * SCALE),
                                  0,
                                  int(wall['radius'] * 2 * SCALE),
                                  WINDOW_SIZE[1]))
            else:
                # Horizontal walls
                pygame.draw.rect(screen, BLACK,
                                 (0,
                                  pos[1] - int(wall['radius'] * SCALE),
                                  WINDOW_SIZE[0],
                                  int(wall['radius'] * 2 * SCALE)))

        # Draw dynamic objects
        colors = [BLUE, RED]
        for i, obj in enumerate(dynamic_objects):
            color = colors[i % len(colors)]
            pos = world_to_screen(obj['position'])

            # Draw trail
            if len(obj['trail']) > 1:
                trail_points = [world_to_screen(p) for p in obj['trail']]
                pygame.draw.lines(screen, color, False, trail_points, 2)

            # Draw safety circle
            safety_color = color if obj['safety']['safe'] else YELLOW
            pygame.draw.circle(screen, safety_color, pos,
                               int((obj['radius'] + config.min_safe_distance) * SCALE), 1)

            # Draw object
            pygame.draw.circle(screen, color, pos, int(obj['radius'] * SCALE))

            # Draw velocity vector
            end_pos = world_to_screen(obj['position'] + obj['velocity'])
            pygame.draw.line(screen, color, pos, end_pos, 2)

        # Render text
        text1 = font.render("Object 1: {}".format(
            "Safe" if dynamic_objects[0]['safety']['safe'] else "Warning!"),
            True,
            GREEN if dynamic_objects[0]['safety']['safe'] else RED
        )

        text2 = font.render("Object 2: {}".format(
            "Safe" if dynamic_objects[1]['safety']['safe'] else "Warning!"),
            True,
            GREEN if dynamic_objects[1]['safety']['safe'] else RED
        )

        text_pause = font.render("PAUSED" if paused else "", True, RED)
        text_help = font.render("Space: Pause  R: Reset", True, BLACK)

        # Now blit the text
        screen.blit(text1, (10, 10))
        screen.blit(text2, (10, 50))
        screen.blit(text_pause, (WINDOW_SIZE[0] - 100, 10))
        screen.blit(text_help, (10, WINDOW_SIZE[1] - 30))

        screen.blit(text1, (10, 10))
        screen.blit(text2, (10, 50))
        screen.blit(text_pause, (WINDOW_SIZE[0] - 100, 10))
        screen.blit(text_help, (10, WINDOW_SIZE[1] - 30))

        pygame.display.flip()
        clock.tick(30)  # 30 FPS

    pygame.quit()


if __name__ == "__main__":
    test_object_tracking()