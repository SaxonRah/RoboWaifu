import pygame
import numpy as np
from dataclasses import dataclass
import math
from typing import List, Tuple


@dataclass
class UCSConfig:
    """Configuration for UCS system"""
    hd_dimension: int = 1000  # Dimension of hypervectors
    temporal_window: float = 1.0  # Time window
    decay_rate: float = 0.1  # Decay constant
    max_weight: float = 1.0  # Maximum weight bound


class HDCEncoder:
    """Hyperdimensional Computing encoder for bird states"""

    def __init__(self, dim: int):
        self.dimension = dim
        # Create projection matrix
        self.projection = np.random.randn(4, dim) / np.sqrt(4)  # 4 features: x, y, dx, dy

    def encode(self, state: np.ndarray) -> np.ndarray:
        """Encode bird state into hypervector"""
        state_norm = state / (np.linalg.norm(state) + 1e-8)
        hd_vector = np.tanh(state_norm @ self.projection)
        return hd_vector


class Bird:
    def __init__(self, x: float, y: float):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.random.randn(2) * 2
        self.acceleration = np.zeros(2)
        self.max_speed = 5.0
        self.max_force = 0.5
        self.history = []  # Store previous positions for trail
        self.max_history = 10

    def apply_force(self, force: np.ndarray):
        self.acceleration += force

    def update(self):
        # Update velocity
        self.velocity += self.acceleration
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        # Update position
        self.position += self.velocity
        # Reset acceleration
        self.acceleration *= 0

        # Update history for trail
        self.history.append(np.copy(self.position))
        if len(self.history) > self.max_history:
            self.history.pop(0)


class FlockingSimulation:
    def __init__(self, width: int, height: int):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("UCS Flocking Simulation")

        # UCS Components
        self.config = UCSConfig()
        self.hdc = HDCEncoder(self.config.hd_dimension)

        # Simulation objects
        self.birds: List[Bird] = []
        self.obstacles: List[Tuple[float, float, float]] = []  # (x, y, radius)

        # Initialize birds
        for _ in range(50):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            self.birds.append(Bird(x, y))

        # Add some obstacles
        self.obstacles.append((width / 2, height / 2, 50))
        self.obstacles.append((width / 4, height / 4, 30))

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()

    def get_state_vector(self, bird: Bird) -> np.ndarray:
        """Get state vector for HDC encoding"""
        return np.array([
            bird.position[0] / self.width,
            bird.position[1] / self.height,
            bird.velocity[0] / bird.max_speed,
            bird.velocity[1] / bird.max_speed
        ])

    def align(self, bird: Bird, neighbors: List[Bird]) -> np.ndarray:
        """Alignment force for flocking behavior"""
        steering = np.zeros(2)
        if not neighbors:
            return steering

        for neighbor in neighbors:
            steering += neighbor.velocity

        steering = steering / len(neighbors)
        steering = (steering / np.linalg.norm(steering) * bird.max_speed
                    if np.linalg.norm(steering) > 0 else steering)
        steering -= bird.velocity
        return np.clip(steering, -bird.max_force, bird.max_force)

    def cohesion(self, bird: Bird, neighbors: List[Bird]) -> np.ndarray:
        """Cohesion force for flocking behavior"""
        steering = np.zeros(2)
        if not neighbors:
            return steering

        center = np.zeros(2)
        for neighbor in neighbors:
            center += neighbor.position
        center = center / len(neighbors)

        desired = center - bird.position
        if np.linalg.norm(desired) > 0:
            desired = desired / np.linalg.norm(desired) * bird.max_speed
        steering = desired - bird.velocity
        return np.clip(steering, -bird.max_force, bird.max_force)

    def separation(self, bird: Bird, neighbors: List[Bird]) -> np.ndarray:
        """Separation force for flocking behavior"""
        steering = np.zeros(2)
        if not neighbors:
            return steering

        for neighbor in neighbors:
            diff = bird.position - neighbor.position
            dist = np.linalg.norm(diff)
            if dist > 0:
                diff = diff / dist / dist  # Weight by inverse square of distance
                steering += diff

        if np.linalg.norm(steering) > 0:
            steering = steering / np.linalg.norm(steering) * bird.max_speed
            steering -= bird.velocity
        return np.clip(steering, -bird.max_force, bird.max_force)

    def avoid_obstacles(self, bird: Bird) -> np.ndarray:
        """Generate force to avoid obstacles"""
        steering = np.zeros(2)
        for ox, oy, radius in self.obstacles:
            to_obstacle = np.array([ox, oy]) - bird.position
            distance = np.linalg.norm(to_obstacle)

            if distance < radius + 30:  # Avoidance radius
                if distance > 0:
                    steering -= to_obstacle / distance * (radius + 30 - distance)

        if np.linalg.norm(steering) > 0:
            steering = steering / np.linalg.norm(steering) * bird.max_speed
            steering -= bird.velocity
        return np.clip(steering, -bird.max_force, bird.max_force)

    def find_neighbors(self, bird: Bird, radius: float) -> List[Bird]:
        """Find neighboring birds within radius"""
        neighbors = []
        for other in self.birds:
            if other != bird:
                distance = np.linalg.norm(bird.position - other.position)
                if distance < radius:
                    neighbors.append(other)
        return neighbors

    def draw_bird(self, bird: Bird):
        """Draw a bird as a triangle with motion trail"""
        # Draw trail
        if len(bird.history) > 1:
            pygame.draw.lines(self.screen, (100, 100, 255), False,
                              [(int(p[0]), int(p[1])) for p in bird.history], 1)

        # Calculate bird triangle points
        angle = math.atan2(bird.velocity[1], bird.velocity[0])
        size = 10
        points = [
            (bird.position[0] + size * math.cos(angle),
             bird.position[1] + size * math.sin(angle)),
            (bird.position[0] + size * math.cos(angle + 2.5),
             bird.position[1] + size * math.sin(angle + 2.5)),
            (bird.position[0] + size * math.cos(angle - 2.5),
             bird.position[1] + size * math.sin(angle - 2.5))
        ]
        pygame.draw.polygon(self.screen, self.WHITE, points)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill(self.BLACK)

            # Draw obstacles
            for ox, oy, radius in self.obstacles:
                pygame.draw.circle(self.screen, self.RED, (int(ox), int(oy)), int(radius))
                # Draw force field
                pygame.draw.circle(self.screen, (100, 0, 0), (int(ox), int(oy)),
                                   int(radius + 30), 1)

            # Update and draw birds
            for bird in self.birds:
                # Encode bird state using HDC
                state_vector = self.get_state_vector(bird)
                hd_state = self.hdc.encode(state_vector)

                # Find neighbors
                neighbors = self.find_neighbors(bird, 50)

                # Calculate forces
                alignment = self.align(bird, neighbors)
                cohesion = self.cohesion(bird, neighbors)
                separation = self.separation(bird, neighbors)
                obstacle_avoidance = self.avoid_obstacles(bird)

                # Apply forces with weights
                bird.apply_force(alignment * 1.0)
                bird.apply_force(cohesion * 1.0)
                bird.apply_force(separation * 1.5)
                bird.apply_force(obstacle_avoidance * 2.0)

                # Update bird position
                bird.update()

                # Wrap around screen edges
                bird.position[0] = bird.position[0] % self.width
                bird.position[1] = bird.position[1] % self.height

                # Draw bird
                self.draw_bird(bird)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    simulation = FlockingSimulation(800, 600)
    simulation.run()