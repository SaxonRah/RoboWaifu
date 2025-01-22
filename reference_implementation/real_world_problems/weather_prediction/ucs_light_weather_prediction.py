import numpy as np
import pygame
from dataclasses import dataclass
from typing import List, Tuple
import sys


@dataclass
class WeatherConfig:
    """Lightweight configuration for UCS weather system"""
    window_size: Tuple[int, int] = (800, 600)
    grid_size: Tuple[int, int] = (80, 60)  # Smaller grid
    hd_dimension: int = 64  # Reduced dimension
    temporal_window: int = 10  # Keep only last 5 frames
    decay_rate: float = 0.2


class LightHDCEncoder:
    """Simplified HDC encoder"""

    def __init__(self, input_dim: int, hd_dim: int):
        self.projection = np.random.randn(input_dim, hd_dim)
        self.projection /= np.linalg.norm(self.projection, axis=0)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.projection)


class LightTemporalProcessor:
    """Simplified temporal processor"""

    def __init__(self, window_size: int, decay_rate: float):
        self.buffer: List[np.ndarray] = []
        self.window_size = window_size
        self.decay_rate = decay_rate

    def process(self, x: np.ndarray) -> np.ndarray:
        self.buffer.append(x)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # Compute weighted average based on recency
        weights = [np.exp(-self.decay_rate * i) for i in range(len(self.buffer))]
        weights = np.array(weights) / sum(weights)

        result = np.zeros_like(x)
        for w, b in zip(weights, self.buffer):
            result += w * b
        return result


class WeatherUCS:
    """Lightweight UCS for weather visualization"""

    def __init__(self, config: WeatherConfig):
        print("Initializing WeatherUCS...")
        self.config = config

        # Initialize UCS components
        self.encoder = LightHDCEncoder(3, config.hd_dimension)
        self.temporal = LightTemporalProcessor(config.temporal_window, config.decay_rate)

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(config.window_size)
        pygame.display.set_caption("Weather UCS")

        # Calculate cell dimensions
        self.cell_width = config.window_size[0] // config.grid_size[0]
        self.cell_height = config.window_size[1] // config.grid_size[1]

        # Initialize weather grid with slight variations
        self.weather_grid = np.random.uniform(0.4, 0.6,
                                              (config.grid_size[1],
                                               config.grid_size[0], 3))

        self.t = 0.0
        print("Initialization complete")

    def process_weather(self, current: np.ndarray) -> np.ndarray:
        """Process single weather cell through UCS pipeline"""
        # HDC encoding
        encoded = self.encoder.encode(current)

        # Temporal processing
        processed = self.temporal.process(encoded)

        # Convert back to weather space (simple linear projection)
        result = np.dot(processed[:3], np.eye(3))
        return np.clip(result, 0, 1)

    def update_weather(self) -> None:
        """Update weather patterns"""
        try:
            new_grid = self.weather_grid.copy()
            self.t += 0.01

            # Process each cell
            for y in range(self.config.grid_size[1]):
                for x in range(self.config.grid_size[0]):
                    current = self.weather_grid[y, x]

                    # Add small random variations
                    noise = np.random.normal(0, 0.01, 3)
                    current = np.clip(current + noise, 0, 1)

                    # Process through UCS pipeline
                    processed = self.process_weather(current)

                    # Smooth update
                    new_grid[y, x] = current + 0.1 * (processed - current)
                    new_grid[y, x] = np.clip(new_grid[y, x], 0, 1)

            # Apply simple diffusion between neighboring cells
            for y in range(self.config.grid_size[1]):
                for x in range(self.config.grid_size[0]):
                    neighbors = []
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.config.grid_size[1] and 0 <= nx < self.config.grid_size[0]:
                            neighbors.append(new_grid[ny, nx])
                    if neighbors:
                        new_grid[y, x] += 0.05 * (np.mean(neighbors, axis=0) - new_grid[y, x])

            self.weather_grid = np.clip(new_grid, 0, 1)

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

                    # Draw cell with padding
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