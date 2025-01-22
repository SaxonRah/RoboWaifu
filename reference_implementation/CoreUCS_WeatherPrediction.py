import numpy as np
import pygame
from dataclasses import dataclass
from typing import List, Tuple
import sys
from CoreUCS import UCSConfig, UnifiedCognitiveSystem


@dataclass
class WeatherConfig:
    """Configuration for UCS weather system"""
    window_size: Tuple[int, int] = (800, 600)
    grid_size: Tuple[int, int] = (80, 60)  # Smaller grid for performance
    # UCS configuration will be initialized separately


class WeatherCell:
    """Represents a single weather cell that can be encoded"""

    def __init__(self, temp: float, pressure: float, wind: float):
        self.temp = temp
        self.pressure = pressure
        self.wind = wind

    def to_array(self) -> np.ndarray:
        """Convert weather state to numpy array for encoding"""
        return np.array([self.temp, self.pressure, self.wind])


class WeatherUCS:
    """Lightweight UCS for weather visualization"""

    def __init__(self, config: WeatherConfig):
        print("Initializing WeatherUCS...")
        self.config = config

        # Initialize core UCS with appropriate dimensions
        ucs_config = UCSConfig(
            input_dimension=3,  # temp, pressure, wind
            hd_dimension=64,  # Reduced for performance
            temporal_window=1.0,
            decay_rate=0.2
        )
        self.ucs = UnifiedCognitiveSystem(ucs_config)

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(config.window_size)
        pygame.display.set_caption("Weather UCS")

        # Calculate cell dimensions
        self.cell_width = config.window_size[0] // config.grid_size[0]
        self.cell_height = config.window_size[1] // config.grid_size[1]

        # Initialize weather grid with slight variations
        self.weather_grid = np.random.uniform(
            0.4, 0.6, (config.grid_size[1], config.grid_size[0], 3)
        )

        self.t = 0.0
        print("Initialization complete")

    def process_weather(self, current: np.ndarray, cell_id: Tuple[int, int]) -> np.ndarray:
        """Process single weather cell through UCS pipeline"""
        # Create WeatherCell object
        cell = WeatherCell(current[0], current[1], current[2])

        # Process through UCS pipeline
        result = self.ucs.process(cell.to_array(), object_id=cell_id)

        # Take first three components for weather values
        return np.clip(result[:3], 0, 1)

    def update_weather(self) -> None:
        """Update weather patterns"""
        try:
            new_grid = self.weather_grid.copy()
            self.t += 0.01

            # Process each cell
            for y in range(self.config.grid_size[1]):
                for x in range(self.config.grid_size[0]):
                    current = self.weather_grid[y, x]
                    cell_id = (x, y)  # Use coordinates as cell ID

                    # Add small random variations
                    noise = np.random.normal(0, 0.01, 3)
                    current = np.clip(current + noise, 0, 1)

                    # Process through UCS pipeline
                    processed = self.process_weather(current, cell_id)

                    # Smooth update
                    new_grid[y, x] = current + 0.1 * (processed - current)
                    new_grid[y, x] = np.clip(new_grid[y, x], 0, 1)

            # Simple diffusion between neighboring cells
            for y in range(self.config.grid_size[1]):
                for x in range(self.config.grid_size[0]):
                    neighbors = []
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.config.grid_size[1] and
                                0 <= nx < self.config.grid_size[0]):
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

            # Draw weather cells
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