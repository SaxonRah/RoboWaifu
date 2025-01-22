import pygame
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random
from contextlib import contextmanager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = (1200, 800)
NODE_RADIUS = 20
LINE_WIDTH = 2
FPS = 60
MAX_BUFFER_SIZE = 5
UPDATE_INTERVAL = 30  # frames between connection updates

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
    power_level: float
    capacity: float
    is_generator: bool
    connections: List[int]

    def __post_init__(self):
        # Validate power level
        self.power_level = np.clip(self.power_level, 0.0, self.capacity)


class NeuralHDCEncoder:
    def __init__(self, hd_dimension: int, input_dimension: int = 4):
        self.hd_dimension = hd_dimension
        # Initialize weights once
        self.W1 = np.random.randn(input_dimension, 16) / np.sqrt(input_dimension)
        self.W2 = np.random.randn(16, hd_dimension) / np.sqrt(16)

    def encode(self, node: PowerNode) -> np.ndarray:
        """Neural-HDC encoding with neural preprocessing"""
        try:
            features = np.array([
                node.power_level,
                float(node.is_generator),
                len(node.connections) / 10.0,
                node.capacity
            ])

            # Forward pass with ReLU activation
            hidden = np.maximum(0, features @ self.W1)
            output = hidden @ self.W2

            # HDC encoding with normalization
            hd_vector = np.tanh(output)
            norm = np.linalg.norm(hd_vector)
            if norm > 0:
                return hd_vector / norm
            return np.zeros(self.hd_dimension)

        except Exception as e:
            logger.error(f"Error encoding node {node.id}: {str(e)}")
            return np.zeros(self.hd_dimension)


class NeuralTemporalProcessor:
    def __init__(self, buffer_size: int = MAX_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.temporal_buffer: Dict[int, np.ndarray] = {}

    def initialize_buffer(self, node_ids: List[int]):
        """Initialize temporal buffer for all nodes"""
        for node_id in node_ids:
            self.temporal_buffer[node_id] = np.zeros(self.buffer_size)

    def process(self, node_id: int, power_level: float) -> float:
        """Process temporal information with attention mechanism"""
        try:
            if node_id not in self.temporal_buffer:
                self.temporal_buffer[node_id] = np.zeros(self.buffer_size)

            # Roll buffer and add new value
            buffer = self.temporal_buffer[node_id]
            buffer = np.roll(buffer, 1)
            buffer[0] = power_level
            self.temporal_buffer[node_id] = buffer

            # Compute attention weights
            attention_weights = np.exp(buffer)
            attention_weights /= np.sum(attention_weights) + 1e-10

            return float((attention_weights * buffer).sum().item())

        except Exception as e:
            logger.error(f"Error processing temporal data for node {node_id}: {str(e)}")
            return power_level


class HybridDynamicGraph:
    def __init__(self, hd_dimension: int):
        self.hd_dimension = hd_dimension
        input_dim = hd_dimension * 2
        # Initialize weights once
        self.W1 = np.random.randn(input_dim, 64) / np.sqrt(input_dim)
        self.W2 = np.random.randn(64, 1) / np.sqrt(64)
        self.edge_cache: Dict[Tuple[int, int], Tuple[float, int]] = {}

    def predict_edge(self, node1_features: np.ndarray, node2_features: np.ndarray,
                     current_frame: int) -> float:
        """Predict edge weight using neural network with caching"""
        try:
            combined_features = np.concatenate([node1_features, node2_features])
            hidden = np.maximum(0, combined_features @ self.W1)
            edge_score = float((1 / (1 + np.exp(-(hidden @ self.W2)))).item())
            return np.clip(edge_score, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error predicting edge: {str(e)}")
            return 0.0


class SmartGridVisualizer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.Font(None, 24)

    @contextmanager
    def safe_drawing(self):
        """Context manager for safe pygame drawing"""
        try:
            yield
        except pygame.error as e:
            logger.error(f"Pygame drawing error: {str(e)}")

    def draw_connection(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int],
                        power_flow: float, time: float):
        """Draw power line connection with flow animation"""
        with self.safe_drawing():
            flow_offset = (math.sin(time / 10) + 1) / 2
            color = (int(255 * (1 - power_flow)), int(255 * power_flow), 0)

            pygame.draw.line(self.screen, color, start_pos, end_pos, LINE_WIDTH)

            mid_x = start_pos[0] + (end_pos[0] - start_pos[0]) * flow_offset
            mid_y = start_pos[1] + (end_pos[1] - start_pos[1]) * flow_offset
            pygame.draw.circle(self.screen, BLUE, (int(mid_x), int(mid_y)), 5)

    def draw_node(self, node: PowerNode):
        """Draw power node with status information"""
        with self.safe_drawing():
            color = YELLOW if node.is_generator else (
                int(255 * (1 - node.power_level)),
                int(255 * node.power_level),
                0
            )

            pygame.draw.circle(self.screen, color, node.pos, NODE_RADIUS)
            pygame.draw.circle(self.screen, BLACK, node.pos, NODE_RADIUS, 2)

            text = self.font.render(f"{node.power_level:.2f}", True, BLACK)
            text_rect = text.get_rect(center=node.pos)
            self.screen.blit(text, text_rect)


class SmartGrid:
    def __init__(self):
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Smart Grid Power Distribution")
        self.clock = pygame.time.Clock()

        # Initialize components
        self.hd_dimension = 1000
        self.encoder = NeuralHDCEncoder(self.hd_dimension)
        self.temporal = NeuralTemporalProcessor()
        self.graph = HybridDynamicGraph(self.hd_dimension)
        self.visualizer = SmartGridVisualizer(self.screen)

        # State
        self.nodes: Dict[int, PowerNode] = {}
        self.running = True
        self.selected_node = None
        self.frame_count = 0

        # Initialize nodes and connections
        self.setup_grid()

    def create_grid_connections(self):
        """Create initial grid connections"""
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

    def update_power_flow(self):
        """Update power levels with fixed power distribution"""
        try:
            # Reset generators to full power
            for node_id, node in self.nodes.items():
                if node.is_generator:
                    node.power_level = node.capacity

            # Update consumer nodes
            for node_id, node in self.nodes.items():
                if not node.is_generator:
                    generators = [n for n in node.connections if self.nodes[n].is_generator]
                    if generators:
                        # Fixed power level for stability
                        node.power_level = 0.8
                    else:
                        # Decrease power if disconnected
                        node.power_level = max(0.0, node.power_level * 0.95)

        except Exception as e:
            logger.error(f"Error updating power flow: {str(e)}")

    def setup_grid(self):
        """Initialize grid with generators and consumers"""
        try:
            # Create power generator nodes
            generator_positions = [(200, 400), (1000, 400)]
            for i, pos in enumerate(generator_positions):
                self.nodes[i] = PowerNode(
                    id=i, pos=pos, power_level=1.0,
                    capacity=1.0, is_generator=True, connections=[]
                )

            # Create consumer nodes
            consumer_positions = [
                (400, 200), (600, 200), (800, 200),
                (400, 600), (600, 600), (800, 600)
            ]
            for i, pos in enumerate(consumer_positions, start=len(generator_positions)):
                self.nodes[i] = PowerNode(
                    id=i, pos=pos, power_level=0.5,
                    capacity=1.0, is_generator=False, connections=[]
                )

            # Initialize temporal buffer and create initial connections
            self.temporal.initialize_buffer(list(self.nodes.keys()))
            self.create_grid_connections()
        except Exception as e:
            logger.error(f"Error setting up grid: {str(e)}")
            raise

    def update_connections(self):
        """Update grid connections using neural edge prediction"""
        if self.frame_count % UPDATE_INTERVAL != 0:
            return

        try:
            for node_id, node in self.nodes.items():
                if node.is_generator:
                    node_features = self.encoder.encode(node)
                    for consumer_id, consumer in self.nodes.items():
                        if not consumer.is_generator:
                            consumer_features = self.encoder.encode(consumer)
                            edge_score = self.graph.predict_edge(
                                node_features, consumer_features, self.frame_count
                            )

                            # Update connections based on prediction
                            if edge_score > 0.5:
                                if consumer_id not in node.connections:
                                    node.connections.append(consumer_id)
                                    consumer.connections.append(node_id)
                            elif consumer_id in node.connections:
                                node.connections.remove(consumer_id)
                                consumer.connections.remove(node_id)

        except Exception as e:
            logger.error(f"Error updating connections: {str(e)}")

    def update_power_flow(self):
        """Update power levels using neural temporal processing"""
        try:
            # First, ensure generators maintain full power
            for node_id, node in self.nodes.items():
                if node.is_generator:
                    node.power_level = node.capacity

            # Then update consumer nodes
            for node_id, node in self.nodes.items():
                if not node.is_generator:
                    generators = [n for n in node.connections if self.nodes[n].is_generator]
                    if generators:
                        # Calculate base power received from each generator
                        base_power = 0.8  # Base power transfer rate
                        connected_count = len(generators)
                        power_received = base_power * connected_count

                        # Apply temporal processing for dynamic adjustment
                        temporal_weight = self.temporal.process(node_id, node.power_level)
                        # Scale power received based on temporal pattern (0.5 to 1.5 range)
                        power_scale = 0.5 + temporal_weight

                        # Calculate final power level
                        new_power = power_received * power_scale

                        # Update power level with clipping
                        node.power_level = np.clip(new_power, 0.0, node.capacity)
                    else:
                        # More gradual power decrease when disconnected
                        node.power_level = max(0.0, node.power_level * 0.95)

        except Exception as e:
            logger.error(f"Error updating power flow: {str(e)}")

    def draw(self):
        """Render the current state"""
        try:
            self.screen.fill(WHITE)

            # Draw connections
            for node_id, node in self.nodes.items():
                for conn_id in node.connections:
                    conn_node = self.nodes[conn_id]
                    power_flow = (node.power_level + conn_node.power_level) / 2
                    self.visualizer.draw_connection(
                        node.pos, conn_node.pos, power_flow, self.frame_count
                    )

            # Draw nodes
            for node in self.nodes.values():
                self.visualizer.draw_node(node)

            pygame.display.flip()

        except Exception as e:
            logger.error(f"Error drawing grid: {str(e)}")

    def cleanup(self):
        """Clean up resources"""
        try:
            pygame.quit()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def run(self):
        """Main simulation loop"""
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_pos = pygame.mouse.get_pos()
                        for node_id, node in self.nodes.items():
                            dist = math.sqrt(
                                (mouse_pos[0] - node.pos[0]) ** 2 +
                                (mouse_pos[1] - node.pos[1]) ** 2
                            )
                            if dist < NODE_RADIUS:
                                self.selected_node = node_id
                                break

                # Update power distribution
                self.update_power_flow()
                self.frame_count += 1

                # Draw everything
                self.draw()

                # Cap frame rate
                self.clock.tick(FPS)

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()


if __name__ == "__main__":
    try:
        smart_grid = SmartGrid()
        smart_grid.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        pygame.quit()
