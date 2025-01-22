import pygame
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from CoreUCS import UnifiedCognitiveSystem, UCSConfig, Encodable


@dataclass
class SwarmConfig(UCSConfig):
    """Configuration for swarm simulation extending UCSConfig"""
    window_width: int = 800
    window_height: int = 600
    num_robots: int = 5
    robot_radius: int = 10
    sensor_range: float = 100.0
    min_separation: float = 25.0
    max_speed: float = 100.0
    cohesion_weight: float = 0.3
    alignment_weight: float = 0.4
    separation_weight: float = 0.5


class Robot(Encodable):
    """Robot agent with UCS-driven behavior"""

    def __init__(self, x: float, y: float, config: SwarmConfig, robot_id: int):
        self.x = x
        self.y = y
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.config = config
        self.color = (50, 150, 250)
        self.id = robot_id  # Use provided ID instead of runtime ID
        self.last_states = []
        self.max_states = 5

    def to_array(self) -> np.ndarray:
        """Convert robot state to array for UCS encoding"""
        return np.array([
            self.x / self.config.window_width,
            self.y / self.config.window_height,
            self.velocity_x / self.config.max_speed,
            self.velocity_y / self.config.max_speed
        ])

    def extract_behavior_vectors(self, processed_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract position and velocity vectors from UCS processed state"""
        temporal_state = processed_state[:4]
        graph_state = processed_state[4:8]

        position = np.array([
            temporal_state[0] * self.config.window_width,
            temporal_state[1] * self.config.window_height
        ])

        velocity = np.array([
            temporal_state[2] * self.config.max_speed,
            temporal_state[3] * self.config.max_speed
        ])

        return position, velocity

    def process_ucs_state(self, processed_state: np.ndarray) -> None:
        """Store and process UCS state information"""
        self.last_states.append(processed_state)
        if len(self.last_states) > self.max_states:
            self.last_states.pop(0)

    def update(self, target_x: float, target_y: float, neighbors: List['Robot'],
               ucs: UnifiedCognitiveSystem, dt: float) -> None:
        """Update robot using UCS-driven behavior"""
        # Process current state through UCS
        current_state = ucs.process(self.to_array(), self.id)
        self.process_ucs_state(current_state)

        # Calculate basic separation force (safety)
        separation_force = np.zeros(2)
        for neighbor in neighbors:
            if neighbor != self:
                dx = self.x - neighbor.x
                dy = self.y - neighbor.y
                dist = np.sqrt(dx * dx + dy * dy)
                if self.config.min_separation > dist > 0:
                    separation_force[0] += dx / dist * (self.config.min_separation - dist)
                    separation_force[1] += dy / dist * (self.config.min_separation - dist)

        # Calculate target attraction
        dx = target_x - self.x
        dy = target_y - self.y
        dist_to_target = np.sqrt(dx * dx + dy * dy)
        target_force = np.zeros(2)
        if dist_to_target > 0:
            target_force = np.array([dx / dist_to_target, dy / dist_to_target]) * self.config.max_speed

        # Calculate final force
        total_force = (
                separation_force * self.config.separation_weight +
                target_force * 0.5  # Reduced target influence
        )

        # Update velocity using combined forces
        self.velocity_x += total_force[0] * dt
        self.velocity_y += total_force[1] * dt

        # Apply velocity limits
        speed = np.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)
        if speed > self.config.max_speed:
            scale = self.config.max_speed / speed
            self.velocity_x *= scale
            self.velocity_y *= scale

        # Update position
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt

        # Boundary conditions
        self.x = np.clip(self.x, 0, self.config.window_width)
        self.y = np.clip(self.y, 0, self.config.window_height)


class SwarmSimulation:
    """Swarm simulation with UCS-driven behavior"""

    def __init__(self, config: SwarmConfig):
        pygame.init()
        self.config = config
        self.screen = pygame.display.set_mode((config.window_width, config.window_height))
        pygame.display.set_caption("UCS Swarm Simulation")

        # Initialize UCS system
        self.ucs = UnifiedCognitiveSystem(config)

        # Initialize robots with sequential IDs
        self.robots = []
        self._initialize_robots()

        self.target_x = config.window_width // 2
        self.target_y = config.window_height // 2
        self.clock = pygame.time.Clock()
        self.running = True

        # Debug visualization
        self.font = pygame.font.Font(None, 24)
        self.show_debug = False

    def _initialize_robots(self) -> None:
        """Initialize robots in a circle formation with sequential IDs"""
        for i in range(self.config.num_robots):
            angle = (2 * np.pi * i) / self.config.num_robots
            radius = 100
            x = self.config.window_width / 2 + radius * np.cos(angle)
            y = self.config.window_height / 2 + radius * np.sin(angle)
            # Use index as robot ID
            self.robots.append(Robot(x, y, self.config, i))

    def update(self, dt: float) -> None:
        """Update simulation state"""
        for robot in self.robots:
            robot.update(self.target_x, self.target_y, self.robots, self.ucs, dt)

    def draw(self) -> None:
        """Render simulation state"""
        self.screen.fill((0, 0, 0))

        # Draw connections between robots
        for i, robot1 in enumerate(self.robots):
            for j, robot2 in enumerate(self.robots[i + 1:], i + 1):
                # Get node indices in UCS graph
                idx1 = self.ucs.graph.nodes.index(robot1.id)
                idx2 = self.ucs.graph.nodes.index(robot2.id)

                # Get weight between robots
                weight = float(self.ucs.graph.weights[idx1, idx2])

                if weight > 0.3:  # Only draw strong connections
                    color = (0, int(255 * weight), 0, int(255 * weight))
                    pygame.draw.line(
                        self.screen,
                        color,
                        (int(robot1.x), int(robot1.y)),
                        (int(robot2.x), int(robot2.y)),
                        1
                    )

        # Draw robots
        for robot in self.robots:
            # Draw sensor range
            pygame.draw.circle(
                self.screen,
                (30, 30, 30),
                (int(robot.x), int(robot.y)),
                int(self.config.sensor_range),
                1
            )

            # Draw robot
            pygame.draw.circle(
                self.screen,
                robot.color,
                (int(robot.x), int(robot.y)),
                self.config.robot_radius
            )

            # Draw velocity vector
            end_x = robot.x + robot.velocity_x * 0.5
            end_y = robot.y + robot.velocity_y * 0.5
            pygame.draw.line(
                self.screen,
                (255, 255, 0),
                (int(robot.x), int(robot.y)),
                (int(end_x), int(end_y)),
                2
            )

            # Draw ID
            if self.show_debug:
                text = self.font.render(str(robot.id), True, (255, 255, 255))
                self.screen.blit(text, (int(robot.x - 10), int(robot.y - 20)))

        # Draw target
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (int(self.target_x), int(self.target_y)),
            15,
            2
        )

        pygame.display.flip()

    def handle_events(self) -> None:
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.target_x, self.target_y = pygame.mouse.get_pos()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    self.show_debug = not self.show_debug

    def run(self) -> None:
        """Main simulation loop"""
        while self.running:
            dt = 1 / 60.0

            self.handle_events()
            self.update(dt)
            self.draw()

            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    # hd_dimension: int = 10000  # Dimension of hypervectors
    # input_dimension: int = 10  # Default input dimension
    # temporal_window: float = 1.0  # Time window for temporal processing
    # decay_rate: float = 0.1  # Temporal decay rate
    # learning_rate: float = 0.1  # Learning rate for graph updates
    # max_weight: float = 1.0  # Maximum weight bound
    # reg_lambda: float = 0.001  # Regularization parameter
    # cache_size: int = 1000  # Size of similarity cache

    # window_width: int = 800
    # window_height: int = 600
    # num_robots: int = 5
    # robot_radius: int = 10
    # sensor_range: float = 100.0
    # min_separation: float = 25.0
    # max_speed: float = 100.0
    # cohesion_weight: float = 0.3
    # alignment_weight: float = 0.4
    # separation_weight: float = 0.5

    # Initialize with UCS-specific configuration
    config = SwarmConfig(
        window_width=800,
        window_height=600,
        num_robots=5,
        input_dimension=4,
        hd_dimension=100,
        temporal_window=2.5,
        decay_rate=0.1,
        learning_rate=0.5,
        max_weight=1.0
    )

    simulation = SwarmSimulation(config)
    simulation.run()
