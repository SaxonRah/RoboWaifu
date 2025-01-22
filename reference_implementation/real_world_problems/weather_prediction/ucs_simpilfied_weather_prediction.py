import numpy as np
import pygame
from dataclasses import dataclass
from typing import Tuple
import sys


@dataclass
class WeatherConfig:
    """Simplified configuration for UCS weather system"""
    window_size: Tuple[int, int] = (800, 600)
    grid_size: Tuple[int, int] = (80, 60)
    hd_dimension: int = 1000


class WeatherUCS:
    """Simplified UCS for weather visualization"""

    def __init__(self, config: WeatherConfig):
        self.config = config
        print("Initializing WeatherUCS...")

        # Initialize pygame
        try:
            pygame.init()
            print("Pygame initialized successfully")
        except Exception as e:
            print(f"Failed to initialize pygame: {e}")
            sys.exit(1)

        # Set up display
        try:
            self.screen = pygame.display.set_mode(config.window_size)
            pygame.display.set_caption("Weather UCS (Debug)")
            print("Display initialized successfully")
        except Exception as e:
            print(f"Failed to create display: {e}")
            pygame.quit()
            sys.exit(1)

        # Calculate cell dimensions
        self.cell_width = config.window_size[0] // config.grid_size[0]
        self.cell_height = config.window_size[1] // config.grid_size[1]

        # Initialize weather grid with simple values
        self.weather_grid = np.zeros((config.grid_size[1], config.grid_size[0], 3))
        # Add some initial variation
        for y in range(config.grid_size[1]):
            for x in range(config.grid_size[0]):
                self.weather_grid[y, x] = [
                    0.5 + 0.1 * np.sin(x / 2),  # Temperature
                    0.5 + 0.1 * np.cos(y / 2),  # Pressure
                    0.5  # Wind
                ]

        self.t = 0.0
        print("Initialization complete")

    def update_weather(self) -> None:
        """Simplified weather update"""
        try:
            # Simple wave-like pattern updates
            self.t += 0.01
            for y in range(self.config.grid_size[1]):
                for x in range(self.config.grid_size[0]):
                    # Create simple wave patterns
                    self.weather_grid[y, x, 0] = 0.5 + 0.3 * np.sin(self.t + x / 2)  # Temperature
                    self.weather_grid[y, x, 1] = 0.5 + 0.3 * np.cos(self.t + y / 2)  # Pressure
                    self.weather_grid[y, x, 2] = 0.5 + 0.3 * np.sin(self.t + (x + y) / 4)  # Wind

            # Ensure values stay in bounds
            self.weather_grid = np.clip(self.weather_grid, 0, 1)

        except Exception as e:
            print(f"Error in update_weather: {e}")
            raise

    def render(self) -> None:
        """Simplified rendering"""
        try:
            # Clear screen
            self.screen.fill((30, 30, 30))

            # Draw cells
            for y in range(self.config.grid_size[1]):
                for x in range(self.config.grid_size[0]):
                    # Get weather data
                    temp, pressure, wind = self.weather_grid[y, x]

                    # Convert to colors (ensure integers)
                    color = (
                        int(temp * 255),
                        int(pressure * 255),
                        int(wind * 255)
                    )

                    # Create rectangle
                    rect = pygame.Rect(
                        x * self.cell_width + 2,
                        y * self.cell_height + 2,
                        self.cell_width - 4,
                        self.cell_height - 4
                    )

                    # Draw rectangle
                    pygame.draw.rect(self.screen, color, rect)

            # Update display
            pygame.display.flip()

        except Exception as e:
            print(f"Error in render: {e}")
            raise

    def run(self) -> None:
        """Main loop with error handling"""
        print("Starting main loop...")
        clock = pygame.time.Clock()
        running = True
        frame_count = 0

        try:
            while running:
                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                # Update and render
                try:
                    self.update_weather()
                    self.render()
                    frame_count += 1

                    # Print debug info every 60 frames
                    if frame_count % 60 == 0:
                        print(f"Frame {frame_count}: Running OK")

                    # Maintain 30 FPS
                    clock.tick(30)

                except Exception as e:
                    print(f"Error in main loop: {e}")
                    running = False

        except Exception as e:
            print(f"Fatal error in main loop: {e}")

        finally:
            print("Cleaning up...")
            pygame.quit()


def main():
    """Entry point with error handling"""
    try:
        print("Starting Weather UCS...")
        config = WeatherConfig()
        weather = WeatherUCS(config)
        weather.run()
    except Exception as e:
        print(f"Fatal error in main: {e}")
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()