import pygame
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = (1200, 800)
NODE_RADIUS = 20
LINE_WIDTH = 2
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)


@dataclass
class PowerNode:
    id: int
    pos: Tuple[int, int]
    power_level: float  # Current power level (0 to 1)
    capacity: float  # Maximum power capacity
    is_generator: bool  # True if power source
    connections: List[int]  # List of connected node IDs


class SmartGrid:
    def __init__(self):
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Smart Grid Power Distribution")
        self.clock = pygame.time.Clock()
        self.nodes: Dict[int, PowerNode] = {}
        self.running = True
        self.selected_node = None
        self.time = 0

        # Initialize nodes
        self.setup_grid()

        # HDC encoding dimension
        self.hd_dim = 1000

    def setup_grid(self):
        # Create power generator nodes
        generator_positions = [
            (200, 400),
            (1000, 400)
        ]

        # Create consumer nodes
        consumer_positions = [
            (400, 200), (600, 200), (800, 200),
            (400, 600), (600, 600), (800, 600)
        ]

        # Add generators
        for i, pos in enumerate(generator_positions):
            self.nodes[i] = PowerNode(
                id=i,
                pos=pos,
                power_level=1.0,
                capacity=1.0,
                is_generator=True,
                connections=[]
            )

        # Add consumers
        for i, pos in enumerate(consumer_positions, start=len(generator_positions)):
            self.nodes[i] = PowerNode(
                id=i,
                pos=pos,
                power_level=0.5,
                capacity=1.0,
                is_generator=False,
                connections=[]
            )

        # Create connections
        self.create_grid_connections()

    def create_grid_connections(self):
        # Connect generators to nearby consumers
        for node_id, node in self.nodes.items():
            if node.is_generator:
                # Connect to closest 3 consumer nodes
                consumers = [(n.id, self.distance(node.pos, n.pos))
                             for n in self.nodes.values() if not n.is_generator]
                closest = sorted(consumers, key=lambda x: x[1])[:3]

                for consumer_id, _ in closest:
                    if consumer_id not in node.connections:
                        node.connections.append(consumer_id)
                        self.nodes[consumer_id].connections.append(node_id)

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def encode_node_state(self, node: PowerNode) -> np.ndarray:
        """Neural-HDC encoding of node state"""
        # Create a high-dimensional vector representing node state
        hd_vector = np.random.randn(self.hd_dim)
        # Scale by power level and generator status
        hd_vector *= (node.power_level + 0.1)  # Add small constant to avoid zero vector
        if node.is_generator:
            hd_vector *= 2  # Generators have stronger representation
        return hd_vector / np.linalg.norm(hd_vector)

    def update_power_flow(self):
        """Update power levels based on Hybrid Dynamic Graph"""
        # Simple power distribution simulation
        for node_id, node in self.nodes.items():
            if not node.is_generator:
                # Get connected generators
                generators = [n for n in node.connections if self.nodes[n].is_generator]
                if generators:
                    # Receive power from connected generators
                    power_received = sum(self.nodes[g].power_level for g in generators) / len(generators)
                    # Add some random fluctuation
                    node.power_level = min(1.0, power_received * (0.8 + 0.4 * random.random()))

        self.time += 1

    def draw(self):
        self.screen.fill(WHITE)

        # Draw connections
        for node_id, node in self.nodes.items():
            for conn_id in node.connections:
                conn_node = self.nodes[conn_id]
                # Calculate power flow intensity
                power_flow = (node.power_level + conn_node.power_level) / 2
                # Create animated power flow effect
                flow_offset = (math.sin(self.time / 10) + 1) / 2
                color = (int(255 * (1 - power_flow)), int(255 * power_flow), 0)

                # Draw animated line
                start_pos = node.pos
                end_pos = conn_node.pos
                pygame.draw.line(self.screen, color, start_pos, end_pos, LINE_WIDTH)

                # Draw flow indicators
                mid_x = start_pos[0] + (end_pos[0] - start_pos[0]) * flow_offset
                mid_y = start_pos[1] + (end_pos[1] - start_pos[1]) * flow_offset
                pygame.draw.circle(self.screen, BLUE, (int(mid_x), int(mid_y)), 5)

        # Draw nodes
        for node_id, node in self.nodes.items():
            # Calculate node color based on power level
            if node.is_generator:
                color = YELLOW
            else:
                green = int(255 * node.power_level)
                red = int(255 * (1 - node.power_level))
                color = (red, green, 0)

            pygame.draw.circle(self.screen, color, node.pos, NODE_RADIUS)
            pygame.draw.circle(self.screen, BLACK, node.pos, NODE_RADIUS, 2)

            # Draw power level text
            font = pygame.font.Font(None, 24)
            text = font.render(f"{node.power_level:.2f}", True, BLACK)
            text_rect = text.get_rect(center=node.pos)
            self.screen.blit(text, text_rect)

        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if a node was clicked
                    mouse_pos = pygame.mouse.get_pos()
                    for node_id, node in self.nodes.items():
                        if self.distance(mouse_pos, node.pos) < NODE_RADIUS:
                            self.selected_node = node_id
                            break

            # Update power flow
            self.update_power_flow()

            # Draw everything
            self.draw()

            # Maintain frame rate
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    smart_grid = SmartGrid()
    smart_grid.run()