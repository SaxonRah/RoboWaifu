import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import random
from typing import List, Tuple, Dict

# Constants
WINDOW_SIZE = 800
GRID_SIZE = 4
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
ROAD_WIDTH = 40
VEHICLE_SIZE = 20
MAX_VEHICLES = 30
INTERSECTION_SIZE = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Vehicle:
    x: float
    y: float
    direction: int  # 0: right, 1: down, 2: left, 3: up
    speed: float
    destination: Tuple[int, int]
    vehicle_type: int  # 0: car, 1: bus, 2: emergency
    current_intersection: Tuple[int, int]
    waiting: bool = False

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.x / WINDOW_SIZE,
            self.y / WINDOW_SIZE,
            self.direction / 4,
            self.speed / 5,
            self.vehicle_type / 3
        ], dtype=torch.float32, device=DEVICE)


class NeuralHDCEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, hd_dim=10000):
        super().__init__()
        self.input_dim = input_dim
        self.hd_dim = hd_dim

        # Neural preprocessing network
        self.preprocessor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        ).to(DEVICE)

        # HDC projection matrix
        self.projection = (torch.randn(input_dim, hd_dim, device=DEVICE, dtype=torch.float32) /
                           np.sqrt(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Neural preprocessing
        enhanced_x = self.preprocessor(x)
        # HDC encoding
        hd_vector = torch.tanh(enhanced_x @ self.projection)
        return hd_vector

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=self.hd_dim)

    def bundle(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        if not vectors:
            return torch.zeros(self.hd_dim, device=DEVICE)
        return torch.tanh(sum(vectors) / len(vectors))


class NeuralTemporalProcessor(nn.Module):
    def __init__(self, hd_dim=10000, hidden_dim=128, window_size=1.0, decay_rate=0.1):
        super().__init__()
        self.hd_dim = hd_dim
        self.window_size = window_size
        self.decay_rate = decay_rate

        self.attention = nn.Sequential(
            nn.Linear(hd_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(DEVICE)

        self.time_buffer = []

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        # Update buffer
        self.time_buffer.append((t, x))
        cutoff_time = t - self.window_size
        self.time_buffer = [(t_i, x_i) for t_i, x_i in self.time_buffer if t_i > cutoff_time]

        if not self.time_buffer:
            return x

        times, samples = zip(*self.time_buffer)
        samples_tensor = torch.stack(samples)

        # Neural attention
        attention_weights = self.attention(samples_tensor)
        weighted_sum = (samples_tensor * attention_weights).sum(dim=0)

        # Temporal decay
        decay_weights = torch.tensor([np.exp(-self.decay_rate * (t - t_i))
                                      for t_i in times], device=DEVICE, dtype=torch.float32)
        decay_sum = (samples_tensor * decay_weights.unsqueeze(1)).sum(dim=0)

        # Combine attention and decay
        return 0.7 * decay_sum + 0.3 * weighted_sum


class HybridDynamicGraph(nn.Module):
    def __init__(self, num_nodes, hd_dim=10000, hidden_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.hd_dim = hd_dim

        self.edge_predictor = nn.Sequential(
            nn.Linear(hd_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(DEVICE)

        self.weights = torch.zeros((num_nodes, num_nodes), device=DEVICE, dtype=torch.float32)
        self.traffic_lights = torch.zeros((int(np.sqrt(num_nodes)),
                                           int(np.sqrt(num_nodes))), device=DEVICE, dtype=torch.float32)
        self.light_timers = torch.zeros_like(self.traffic_lights)
        self.min_green_time = 60

    def update_weights(self, node_features: Dict[int, torch.Tensor]) -> None:
        with torch.no_grad():
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if i in node_features and j in node_features:
                        # Ensure consistent dtype for concatenated tensors
                        feature_i = node_features[i].to(torch.float32)
                        feature_j = node_features[j].to(torch.float32)
                        combined = torch.cat([feature_i, feature_j])
                        weight = self.edge_predictor(combined.unsqueeze(0)).item()
                        self.weights[i, j] = self.weights[j, i] = weight

    def update_traffic_lights(self) -> None:
        grid_size = int(np.sqrt(self.num_nodes))
        self.light_timers += 1

        for i in range(grid_size):
            for j in range(grid_size):
                if self.light_timers[i, j] >= self.min_green_time:
                    node_idx = i * grid_size + j
                    incoming_flow = self.weights[node_idx].sum()

                    if incoming_flow > 0.5 and self.traffic_lights[i, j] == 0:
                        self.traffic_lights[i, j] = 1
                        self.light_timers[i, j] = 0
                    elif incoming_flow <= 0.5 and self.traffic_lights[i, j] == 1:
                        self.traffic_lights[i, j] = 0
                        self.light_timers[i, j] = 0


class TrafficHybridUCS:
    def __init__(self, grid_size: int, hd_dim: int = 10000):
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.hd_dim = hd_dim

        # Initialize components
        self.encoder = NeuralHDCEncoder(input_dim=5, hd_dim=hd_dim)
        self.temporal = NeuralTemporalProcessor(hd_dim=hd_dim)
        self.graph = HybridDynamicGraph(num_nodes=self.num_nodes, hd_dim=hd_dim)

        # Optimization
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.temporal.parameters()},
            {'params': self.graph.parameters()}
        ])

        self.t = 0.0

    def process(self, vehicles: List[Vehicle]) -> torch.Tensor:
        self.t += 0.01

        if not vehicles:
            return self.graph.traffic_lights

        # 1. Neural-HDC encoding
        vehicle_tensors = [v.to_tensor() for v in vehicles]
        encoded_states = [self.encoder(v) for v in vehicle_tensors]

        # 2. Temporal processing
        temporal_states = [self.temporal(self.t, e) for e in encoded_states]

        # 3. Update graph
        node_features = {}
        for i, state in enumerate(temporal_states):
            grid_x = int(vehicles[i].x // CELL_SIZE)
            grid_y = int(vehicles[i].y // CELL_SIZE)
            node_id = grid_y * self.grid_size + grid_x

            if node_id not in node_features:
                node_features[node_id] = state
            else:
                node_features[node_id] = self.encoder.bundle([node_features[node_id], state])

        # 4. Update graph and traffic lights
        self.graph.update_weights(node_features)
        self.graph.update_traffic_lights()

        return self.graph.traffic_lights


class NeuralTrafficManager:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Neural City Traffic Management")

        self.vehicles = []
        self.ucs = TrafficHybridUCS(grid_size=GRID_SIZE)
        self.clock = pygame.time.Clock()

    def generate_vehicle(self):
        if len(self.vehicles) < MAX_VEHICLES:
            side = random.randint(0, 3)
            if side == 0:  # Top
                x = CELL_SIZE * random.randint(0, GRID_SIZE - 1) + CELL_SIZE // 2
                y = ROAD_WIDTH // 2
                direction = 1
            elif side == 1:  # Right
                x = WINDOW_SIZE - ROAD_WIDTH // 2
                y = CELL_SIZE * random.randint(0, GRID_SIZE - 1) + CELL_SIZE // 2
                direction = 2
            elif side == 2:  # Bottom
                x = CELL_SIZE * random.randint(0, GRID_SIZE - 1) + CELL_SIZE // 2
                y = WINDOW_SIZE - ROAD_WIDTH // 2
                direction = 3
            else:  # Left
                x = ROAD_WIDTH // 2
                y = CELL_SIZE * random.randint(0, GRID_SIZE - 1) + CELL_SIZE // 2
                direction = 0

            vehicle_type = random.randint(0, 2)
            speed = random.uniform(2, 4)
            current_intersection = (int(x // CELL_SIZE), int(y // CELL_SIZE))

            if direction in [0, 2]:  # Moving horizontally
                dest_x = GRID_SIZE - 1 if direction == 0 else 0
                dest_y = random.randint(0, GRID_SIZE - 1)
            else:  # Moving vertically
                dest_x = random.randint(0, GRID_SIZE - 1)
                dest_y = GRID_SIZE - 1 if direction == 1 else 0

            self.vehicles.append(Vehicle(x, y, direction, speed, (dest_x, dest_y),
                                         vehicle_type, current_intersection))

    def can_move(self, vehicle: Vehicle, traffic_lights: torch.Tensor) -> bool:
        grid_x, grid_y = int(vehicle.x // CELL_SIZE), int(vehicle.y // CELL_SIZE)

        # Check if at intersection
        at_intersection = (abs(vehicle.x - (grid_x * CELL_SIZE + CELL_SIZE // 2)) < INTERSECTION_SIZE and
                           abs(vehicle.y - (grid_y * CELL_SIZE + CELL_SIZE // 2)) < INTERSECTION_SIZE)

        if at_intersection and 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
            # Check traffic light
            if vehicle.direction in [0, 2]:  # Moving horizontally
                return traffic_lights[grid_y, grid_x] == 0
            else:  # Moving vertically
                return traffic_lights[grid_y, grid_x] == 1

        return True

    def update_vehicles(self, traffic_lights: torch.Tensor):
        for vehicle in self.vehicles[:]:
            if self.can_move(vehicle, traffic_lights):
                vehicle.waiting = False
                if vehicle.direction == 0:  # Right
                    vehicle.x += vehicle.speed
                elif vehicle.direction == 1:  # Down
                    vehicle.y += vehicle.speed
                elif vehicle.direction == 2:  # Left
                    vehicle.x -= vehicle.speed
                else:  # Up
                    vehicle.y -= vehicle.speed
            else:
                vehicle.waiting = True

            vehicle.current_intersection = (int(vehicle.x // CELL_SIZE),
                                            int(vehicle.y // CELL_SIZE))

            if (vehicle.x < 0 or vehicle.x > WINDOW_SIZE or
                    vehicle.y < 0 or vehicle.y > WINDOW_SIZE):
                self.vehicles.remove(vehicle)

    def draw_roads(self):
        for i in range(GRID_SIZE):
            y = i * CELL_SIZE + CELL_SIZE // 2 - ROAD_WIDTH // 2
            pygame.draw.rect(self.screen, GRAY, (0, y, WINDOW_SIZE, ROAD_WIDTH))
            x = i * CELL_SIZE + CELL_SIZE // 2 - ROAD_WIDTH // 2
            pygame.draw.rect(self.screen, GRAY, (x, 0, ROAD_WIDTH, WINDOW_SIZE))

    def draw_traffic_lights(self, traffic_lights: torch.Tensor):
        traffic_lights = traffic_lights.cpu().numpy()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                center_x = j * CELL_SIZE + CELL_SIZE // 2
                center_y = i * CELL_SIZE + CELL_SIZE // 2

                # Draw NS light
                ns_color = GREEN if traffic_lights[i, j] == 1 else RED
                pygame.draw.circle(self.screen, ns_color,
                                   (center_x - 10, center_y - 10), 5)

                # Draw EW light
                ew_color = GREEN if traffic_lights[i, j] == 0 else RED
                pygame.draw.circle(self.screen, ew_color,
                                   (center_x + 10, center_y - 10), 5)

    def draw(self, traffic_lights: torch.Tensor):
        self.screen.fill(WHITE)
        self.draw_roads()
        self.draw_traffic_lights(traffic_lights)

        # Draw vehicles
        for vehicle in self.vehicles:
            color = BLUE if vehicle.vehicle_type == 0 else YELLOW if vehicle.vehicle_type == 1 else RED
            pygame.draw.rect(self.screen, color,
                             (vehicle.x - VEHICLE_SIZE // 2, vehicle.y - VEHICLE_SIZE // 2,
                              VEHICLE_SIZE, VEHICLE_SIZE))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Generate new vehicles
            if random.random() < 0.02:  # 2% chance each frame
                self.generate_vehicle()

            # Process through Hybrid UCS
            traffic_lights = self.ucs.process(self.vehicles)

            # Update and draw
            self.update_vehicles(traffic_lights)
            self.draw(traffic_lights)

            self.clock.tick(600)

        pygame.quit()


if __name__ == "__main__":
    manager = NeuralTrafficManager()
    manager.run()