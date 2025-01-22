import pygame
import numpy as np
from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SPRITE_SIZE = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


@dataclass
class Species:
    """Species class with HDC-encoded traits"""
    name: str
    position: Tuple[float, float]
    traits: np.ndarray  # HDC encoded traits
    energy: float
    color: Tuple[int, int, int]
    prey: List[str]
    velocity: Tuple[float, float] = (0.0, 0.0)  # Current velocity for momentum
    max_speed: float = 3.0  # Maximum movement speed


class HDCEncoder:
    """Hyperdimensional Computing encoder for species traits"""

    def __init__(self, dim=1000, num_traits=4):
        self.dim = dim
        self.num_traits = num_traits
        self.projection = np.random.randn(num_traits, dim) / np.sqrt(num_traits)
        self.cache = {}  # Cache for similarity computations

    def encode(self, traits: List[float]) -> np.ndarray:
        """Encode traits into hypervector with input validation"""
        if len(traits) != self.num_traits:
            raise ValueError(f"Expected {self.num_traits} traits, got {len(traits)}")

        traits = np.array(traits)
        if not np.all((traits >= 0) & (traits <= 1)):
            raise ValueError("Traits must be in range [0, 1]")

        # Normalize input
        traits = (traits - traits.mean()) / (traits.std() + 1e-8)
        hd_vector = np.tanh(traits @ self.projection)
        return hd_vector / (np.linalg.norm(hd_vector) + 1e-8)

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two hypervectors with caching"""
        # Create cache key from vector hashes
        key = (hash(vec1.tobytes()), hash(vec2.tobytes()))
        if key in self.cache:
            return self.cache[key]

        # Compute and cache similarity
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
        self.cache[key] = sim

        # Limit cache size
        if len(self.cache) > 1000:
            self.cache.pop(next(iter(self.cache)))

        return sim


class DynamicGraph:
    """Graph representation of ecosystem relationships"""

    def __init__(self, decay_rate=0.1):
        self.weights = {}  # Edge weights between species
        self.decay_rate = decay_rate
        self.update_counter = 0
        self.last_update = {}  # Track when each edge was last updated

    def update_weights(self, species_dict: Dict[str, Species]):
        """Update relationship weights based on species interactions with optimizations"""
        self.update_counter += 1

        # Update weights selectively and apply temporal decay
        updated_weights = {}

        for name1, species1 in species_dict.items():
            for name2, species2 in species_dict.items():
                if name1 >= name2:  # Only compute for unique pairs
                    continue

                edge = (name1, name2)

                # Check if update is needed (every 10 frames)
                if self.update_counter % 10 == 0 or edge not in self.weights:
                    # Compute relationship strength based on traits and spatial proximity
                    trait_similarity = hdc_encoder.similarity(species1.traits, species2.traits)

                    # Spatial component
                    distance = np.sqrt(
                        (species1.position[0] - species2.position[0]) ** 2 +
                        (species1.position[1] - species2.position[1]) ** 2
                    )
                    spatial_factor = np.exp(-distance / 100.0)  # Decay with distance

                    # Combined weight with temporal decay
                    new_weight = 0.7 * trait_similarity + 0.3 * spatial_factor

                    # Apply temporal decay based on last update
                    time_since_update = self.update_counter - self.last_update.get(edge, 0)
                    decay = np.exp(-self.decay_rate * time_since_update)

                    updated_weights[edge] = new_weight * decay
                    updated_weights[(name2, name1)] = updated_weights[edge]  # Symmetric
                    self.last_update[edge] = self.update_counter
                else:
                    # Keep existing weight with decay
                    if edge in self.weights:
                        updated_weights[edge] = self.weights[edge]
                        updated_weights[(name2, name1)] = self.weights[edge]

        self.weights = updated_weights


class EcosystemSimulation:
    def __init__(self):
        try:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Adaptive Ecosystem Simulation")
            self.clock = pygame.time.Clock()

            self.species = {}
            self.graph = DynamicGraph(decay_rate=0.05)
            self.time = 0.0

            # Environmental factors
            self.environment = {
                'temperature': 0.5,  # Normalized [0,1], 0.5 is optimal
                'resources': 1.0,  # Global resource availability
                'carrying_capacity': {
                    'Grass': 50,
                    'Rabbit': 30,
                    'Fox': 15
                }
            }

            # Performance monitoring
            self.last_update_time = pygame.time.get_ticks()
            self.frame_times = []

            # Initialize species
            self.initialize_species()

        except pygame.error as e:
            print(f"Failed to initialize pygame: {e}")
            raise

    def __del__(self):
        """Cleanup resources"""
        try:
            pygame.quit()
        except:
            pass

    def initialize_species(self):
        """Initialize species with random traits"""
        species_configs = [
            ("Rabbit", GREEN, ["Grass"]),
            ("Fox", RED, ["Rabbit"]),
            ("Grass", (0, 150, 0), [])
        ]

        for name, color, prey in species_configs:
            for _ in range(3):  # Create multiple instances of each species
                position = (random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT))
                traits = [
                    random.random(),  # size
                    random.random(),  # speed
                    random.random(),  # strength
                    random.random()  # reproduction rate
                ]

                species_id = f"{name}_{len([s for s in self.species if name in s])}"
                self.species[species_id] = Species(
                    name=name,
                    position=position,
                    traits=hdc_encoder.encode(traits),
                    energy=100.0,
                    color=color,
                    prey=prey
                )

    def update_movement(self, species_name: str, species: Species) -> Tuple[float, float]:
        """Calculate new position based on nearby food/threats and HDC similarity"""
        # Constants for forces
        ATTRACTION_FORCE = 0.5
        REPULSION_FORCE = 0.8
        MOMENTUM_FACTOR = 0.8
        RANDOM_FORCE = 0.1
        BOUNDARY_FORCE = 1.0

        # Initialize force components
        force_x, force_y = 0.0, 0.0

        # Get current position and velocity
        curr_x, curr_y = species.position
        vel_x, vel_y = species.velocity

        # 1. Attraction to prey (food-seeking behavior)
        for other_name, other in self.species.items():
            if other_name != species_name:
                # Calculate distance and direction
                dx = other.position[0] - curr_x
                dy = other.position[1] - curr_y
                distance = np.sqrt(dx * dx + dy * dy) + 1e-5  # Avoid division by zero

                # Get relationship weight from graph
                weight = self.graph.weights.get((species_name, other_name), 0.0)

                # Determine if other species is prey
                is_prey = any(prey_type in other_name for prey_type in species.prey)
                is_predator = any(prey_type in species_name for prey_type in other.prey)

                if is_prey:
                    # Attract towards prey
                    force_x += ATTRACTION_FORCE * weight * dx / distance
                    force_y += ATTRACTION_FORCE * weight * dy / distance
                elif is_predator:
                    # Run away from predators
                    force_x -= REPULSION_FORCE * weight * dx / distance
                    force_y -= REPULSION_FORCE * weight * dy / distance

        # 2. Boundary avoidance (soft boundaries)
        if curr_x < 50:
            force_x += BOUNDARY_FORCE * (50 - curr_x) / 50
        elif curr_x > SCREEN_WIDTH - 50:
            force_x -= BOUNDARY_FORCE * (curr_x - (SCREEN_WIDTH - 50)) / 50

        if curr_y < 50:
            force_y += BOUNDARY_FORCE * (50 - curr_y) / 50
        elif curr_y > SCREEN_HEIGHT - 50:
            force_y -= BOUNDARY_FORCE * (curr_y - (SCREEN_HEIGHT - 50)) / 50

        # 3. Add small random movement (exploration)
        force_x += RANDOM_FORCE * random.uniform(-1, 1)
        force_y += RANDOM_FORCE * random.uniform(-1, 1)

        # 4. Apply momentum (smooth transitions)
        new_vel_x = MOMENTUM_FACTOR * vel_x + (1 - MOMENTUM_FACTOR) * force_x
        new_vel_y = MOMENTUM_FACTOR * vel_y + (1 - MOMENTUM_FACTOR) * force_y

        # 5. Limit maximum speed
        speed = np.sqrt(new_vel_x * new_vel_x + new_vel_y * new_vel_y)
        if speed > species.max_speed:
            new_vel_x = (new_vel_x / speed) * species.max_speed
            new_vel_y = (new_vel_y / speed) * species.max_speed

        # Update velocity
        species.velocity = (new_vel_x, new_vel_y)

        # Calculate new position
        new_x = max(0, min(SCREEN_WIDTH, curr_x + new_vel_x))
        new_y = max(0, min(SCREEN_HEIGHT, curr_y + new_vel_y))

        return (new_x, new_y)

    def update(self):
        """Update ecosystem state"""
        self.time += 0.1

        # Update graph weights
        self.graph.update_weights(self.species)

        # Update species positions and interactions
        to_remove = []
        for name, species in self.species.items():
            # Calculate movement using the movement system
            new_pos = self.update_movement(name, species)
            species.position = new_pos

            # Energy decay
            species.energy -= 0.1

            # Check for nearby prey
            for prey_name, prey in self.species.items():
                if any(prey_type in prey_name for prey_type in species.prey):
                    distance = np.sqrt(
                        (species.position[0] - prey.position[0]) ** 2 +
                        (species.position[1] - prey.position[1]) ** 2
                    )
                    if distance < SPRITE_SIZE * 2:
                        species.energy += 20
                        to_remove.append(prey_name)

            # Remove dead species
            if species.energy <= 0:
                to_remove.append(name)

        # Remove dead species
        for name in to_remove:
            if name in self.species:
                del self.species[name]

        # Add new grass periodically
        if random.random() < 0.02:
            position = (random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT))
            traits = [random.random() for _ in range(4)]
            species_id = f"Grass_{len([s for s in self.species if 'Grass' in s])}"
            self.species[species_id] = Species(
                name="Grass",
                position=position,
                traits=hdc_encoder.encode(traits),
                energy=100.0,
                color=(0, 150, 0),
                prey=[]
            )

    def draw(self):
        """Draw the current state"""
        self.screen.fill(WHITE)

        # Draw relationships (edges)
        for (name1, name2), weight in self.graph.weights.items():
            if name1 in self.species and name2 in self.species:
                start_pos = self.species[name1].position
                end_pos = self.species[name2].position
                if any(prey_type in name2 for prey_type in self.species[name1].prey):
                    # Draw predator-prey relationships
                    # Ensure weight is between 0 and 1, then convert to valid RGB
                    intensity = max(0, min(255, int(255 * abs(weight))))
                    color = (intensity, 0, 0)  # Red with varying intensity
                    pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        # Draw species
        for species in self.species.values():
            pygame.draw.circle(self.screen, species.color,
                               (int(species.position[0]), int(species.position[1])),
                               SPRITE_SIZE // 2)

        pygame.display.flip()

    def run(self):
        """Main simulation loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


# Initialize HDC encoder globally
hdc_encoder = HDCEncoder()

# Run simulation
if __name__ == "__main__":
    simulation = EcosystemSimulation()
    simulation.run()