import numpy as np
import pygame
from dataclasses import dataclass
from typing import Dict, List, Tuple
import scipy.sparse as sparse
import sys


@dataclass
class WeatherConfig:
    """Configuration for UCS weather system"""
    window_size: Tuple[int, int] = (800, 600)
    grid_size: Tuple[int, int] = (10, 8)
    hd_dimension: int = 1000  # HDC vector dimension
    input_dimension: int = 3  # [temperature, pressure, wind]
    temporal_window: float = 1.0  # Time window
    decay_rate: float = 0.1  # Temporal decay rate
    learning_rate: float = 0.01  # For graph updates


class HDCEncoder:
    """HDC encoder following UCS principles"""

    def __init__(self, config: WeatherConfig):
        # Create normalized projection matrix
        self.projection = np.random.randn(config.input_dimension, config.hd_dimension)
        self.projection /= np.linalg.norm(self.projection, axis=0)

    def encode(self, weather_data: np.ndarray) -> np.ndarray:
        """Encode weather data into hypervector"""
        if weather_data.ndim == 1:
            weather_data = weather_data.reshape(1, -1)
        # Normalize and project to high-dimensional space
        weather_norm = weather_data / (np.linalg.norm(weather_data, axis=1, keepdims=True) + 1e-8)
        hd_vector = np.tanh(weather_norm @ self.projection)
        return hd_vector


class TemporalProcessor:
    """Temporal processing following UCS principles"""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.time_buffer: List[Tuple[float, np.ndarray]] = []

    def process(self, t: float, x: np.ndarray) -> np.ndarray:
        """Process temporal weather data"""
        # Update buffer
        self.time_buffer.append((t, x))
        # Remove old samples
        cutoff_time = t - self.config.temporal_window
        self.time_buffer = [(t_i, x_i) for t_i, x_i in self.time_buffer if t_i > cutoff_time]

        # Compute temporal integral with exponential decay
        result = x.copy()
        for t_i, x_i in self.time_buffer[:-1]:
            weight = np.exp(-self.config.decay_rate * (t - t_i))
            result += weight * (x_i - x) * (t - t_i)
        return result


class DynamicGraph:
    """Dynamic graph following UCS principles"""

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.weights = sparse.lil_matrix((0, 0))
        self.nodes = []
        self.node_features = {}

    def update_weights(self, features: Dict[int, np.ndarray]) -> None:
        """Update graph weights based on feature similarity"""
        n = len(self.nodes)
        if n < 2:
            return

        if self.weights.shape != (n, n):
            self.weights = sparse.lil_matrix((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                f_i = features[self.nodes[i]]
                f_j = features[self.nodes[j]]
                # Compute similarity
                sim = self._bounded_similarity(f_i, f_j)
                # Update weights
                grad = 2 * (self.weights[i, j] - sim)
                new_weight = self.weights[i, j] - self.config.learning_rate * grad
                new_weight = np.clip(new_weight, 0, 1)
                self.weights[i, j] = self.weights[j, i] = new_weight

    def _bounded_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute similarity with proven bounds in [0,1]"""
        x = x.ravel()
        y = y.ravel()
        cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
        return 0.5 * (cos_sim + 1)  # Map to [0,1]


class WeatherUCS:
    """Unified Cognitive System for weather pattern prediction"""

    def __init__(self, config: WeatherConfig):
        print("Initializing WeatherUCS...")
        self.config = config

        # Initialize UCS components
        self.encoder = HDCEncoder(config)
        self.temporal = TemporalProcessor(config)
        self.graph = DynamicGraph(config)
        self.t = 0.0

        # Initialize pygame
        try:
            pygame.init()
            self.screen = pygame.display.set_mode(config.window_size)
            pygame.display.set_caption("Weather UCS")
            print("Display initialized successfully")
        except Exception as e:
            print(f"Failed to initialize pygame: {e}")
            sys.exit(1)

        # Calculate cell dimensions
        self.cell_width = config.window_size[0] // config.grid_size[0]
        self.cell_height = config.window_size[1] // config.grid_size[1]

        # Initialize weather grid
        self.weather_grid = np.zeros((config.grid_size[1], config.grid_size[0], 3))
        for y in range(config.grid_size[1]):
            for x in range(config.grid_size[0]):
                self.weather_grid[y, x] = [0.5, 0.5, 0.5]

        print("Initialization complete")

    def process_cell(self, weather_data: np.ndarray) -> np.ndarray:
        """Process weather data through UCS pipeline"""
        try:
            # Ensure input data is correct shape
            weather_data = np.asarray(weather_data).flatten()[:3]
            if weather_data.size < 3:
                weather_data = np.pad(weather_data, (0, 3 - weather_data.size))

            # 1. HDC encoding
            hd_x = self.encoder.encode(weather_data)

            # 2. Temporal processing
            temporal_x = self.temporal.process(self.t, hd_x.reshape(1, -1))

            # 3. Graph processing
            node_id = len(self.graph.nodes)
            self.graph.nodes.append(node_id)
            self.graph.node_features[node_id] = temporal_x.flatten()
            self.graph.update_weights(self.graph.node_features)

            # 4. Graph embedding
            if self.graph.weights.shape[0] > 1:
                laplacian = sparse.eye(self.graph.weights.shape[0]) - self.graph.weights
                try:
                    if laplacian.shape[0] < 5:
                        eigenvals, eigenvects = np.linalg.eigh(laplacian.toarray())
                    else:
                        k = min(3, laplacian.shape[0] - 1)
                        eigenvals, eigenvects = sparse.linalg.eigsh(laplacian, k=k, which='SM')
                    embedding = np.real(eigenvects[:, 1:])
                except Exception as e:
                    print(f"Embedding error: {e}")
                    embedding = np.zeros((1, 3))
            else:
                embedding = np.zeros((1, 3))

            # 5. Return processed weather data
            # Ensure temporal_x has correct shape
            temporal_features = temporal_x.flatten()[:3]

            # Ensure embedding has correct shape
            if embedding.size > 0:
                embedding_features = embedding.flatten()[:3]
                if embedding_features.size < 3:
                    embedding_features = np.pad(embedding_features, (0, 3 - embedding_features.size))
            else:
                embedding_features = np.zeros(3)

            # Combine features
            result = 0.7 * temporal_features + 0.3 * embedding_features
            return np.clip(result, 0, 1)
        finally:
            ...

    def update_weather(self) -> None:
        """Update weather patterns using UCS processing"""
        try:
            new_grid = self.weather_grid.copy()
            self.t += 0.01

            for y in range(self.config.grid_size[1]):
                for x in range(self.config.grid_size[0]):
                    # Process through UCS pipeline
                    current_weather = self.weather_grid[y, x]
                    processed_weather = self.process_cell(current_weather)

                    # Gradual update
                    new_grid[y, x] = current_weather + 0.1 * (processed_weather - current_weather)
                    new_grid[y, x] = np.clip(new_grid[y, x], 0, 1)

            self.weather_grid = new_grid

        except Exception as e:
            print(f"Error in update_weather: {e}")

    def render(self) -> None:
        """Render weather visualization"""
        try:
            self.screen.fill((30, 30, 30))

            for y in range(self.config.grid_size[1]):
                for x in range(self.config.grid_size[0]):
                    # Convert weather data to colors
                    temp, pressure, wind = self.weather_grid[y, x]
                    color = (
                        int(temp * 255),
                        int(pressure * 255),
                        int(wind * 255)
                    )

                    rect = pygame.Rect(
                        x * self.cell_width + 2,
                        y * self.cell_height + 2,
                        self.cell_width - 4,
                        self.cell_height - 4
                    )
                    pygame.draw.rect(self.screen, color, rect)

            # Draw time
            font = pygame.font.Font(None, 24)
            time_text = font.render(f"Time: {self.t:.1f}", True, (255, 255, 255))
            self.screen.blit(time_text, (10, 10))

            pygame.display.flip()

        except Exception as e:
            print(f"Error in render: {e}")

    def run(self) -> None:
        """Main loop"""
        print("Starting main loop...")
        clock = pygame.time.Clock()
        running = True
        frame_count = 0

        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                self.update_weather()
                self.render()
                frame_count += 1

                if frame_count % 60 == 0:
                    print(f"Frame {frame_count}: t={self.t:.1f}")

                clock.tick(30)

        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            print("Cleaning up...")
            pygame.quit()


def main():
    """Entry point"""
    try:
        config = WeatherConfig()
        weather = WeatherUCS(config)
        weather.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()