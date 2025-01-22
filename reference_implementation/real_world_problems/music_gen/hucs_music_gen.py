import pygame
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import scipy.sparse as sparse
from typing import Dict, List, Tuple, Optional


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 44100, volume: float = 0.5):
    """Generate a sine wave sound using NumPy."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 32767 * volume * np.sin(2 * np.pi * frequency * t)
    waveform = waveform.astype(np.int16)  # Convert to 16-bit PCM format
    stereo_waveform = np.column_stack((waveform, waveform))  # Make it stereo
    return stereo_waveform


@dataclass
class HybridMusicConfig:
    """Configuration for hybrid UCS music system"""
    hd_dimension: int = 10000  # Hypervector dimension
    input_dimension: int = 88  # Piano keys
    temporal_window: float = 2.0  # Time window in seconds
    decay_rate: float = 0.1
    learning_rate: float = 0.01
    max_weight: float = 1.0
    reg_lambda: float = 0.001
    nn_hidden_dim: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Music-specific parameters
    notes_per_bar: int = 8
    num_bars: int = 4
    note_duration: float = 0.5  # seconds


class NeuralHDCMusicEncoder(nn.Module):
    """Enhanced HDC encoder with neural preprocessing for music"""

    def __init__(self, config: HybridMusicConfig):
        super().__init__()
        self.config = config

        # Neural preprocessing for musical features
        self.preprocessor = nn.Sequential(
            nn.Linear(config.input_dimension, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, config.input_dimension)
        ).to(config.device)

        # Initialize projection matrix for HDC
        self.projection = np.random.randn(config.input_dimension, config.hd_dimension) / np.sqrt(config.input_dimension)
        self.projection /= np.linalg.norm(self.projection, axis=0)

    def forward(self, x: torch.Tensor) -> np.ndarray:
        # Neural preprocessing
        enhanced_x = self.preprocessor(x)
        # Convert to numpy for HDC encoding
        enhanced_x_np = enhanced_x.detach().cpu().numpy()
        # HDC encoding with tanh activation
        hd_vector = np.tanh(enhanced_x_np @ self.projection)
        return hd_vector


class NeuralTemporalMusic(nn.Module):
    """Enhanced temporal processor with neural attention for music"""

    def __init__(self, config: HybridMusicConfig):
        super().__init__()
        self.config = config
        self.time_buffer: List[Tuple[float, np.ndarray]] = []

        # Neural attention for temporal weighting
        self.attention = nn.Sequential(
            nn.Linear(config.hd_dimension, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, 1),
            nn.Sigmoid()
        ).to(config.device)

    def process(self, t: float, x: np.ndarray) -> np.ndarray:
        # Update buffer
        self.time_buffer.append((t, x))
        # Remove old samples
        cutoff_time = t - self.config.temporal_window
        self.time_buffer = [(t_i, x_i) for t_i, x_i in self.time_buffer if t_i > cutoff_time]

        # Compute temporal integration
        result = x.copy()
        if self.time_buffer:
            times, samples = zip(*self.time_buffer)
            samples_tensor = torch.FloatTensor(np.array(samples)).to(self.config.device)
            attention_weights = self.attention(samples_tensor)
            weighted_samples = (samples_tensor * attention_weights).mean(dim=0)
            result = weighted_samples.detach().cpu().numpy()

        return result


class HybridMusicGraph:
    """Enhanced dynamic graph with neural edge prediction for music structure"""

    def __init__(self, config: HybridMusicConfig):
        self.config = config
        self.weights = sparse.lil_matrix((0, 0))
        self.nodes = []
        self.node_features = {}
        self.hd_dim = 128  # Match the reduced dimension from projection matrix

        # Neural edge predictor with corrected dimensions
        self.edge_predictor = nn.Sequential(
            nn.Linear(self.hd_dim * 2, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, 1),
            nn.Sigmoid()
        ).to(config.device)

        self.optimizer = torch.optim.Adam(self.edge_predictor.parameters())

    def update_weights(self, features: Dict[int, np.ndarray]) -> None:
        n = len(self.nodes)
        if n == 0:
            return

        if self.weights.shape != (n, n):
            self.weights = sparse.lil_matrix((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                f_i = features[i]
                f_j = features[j]

                # Ensure correct feature dimensionality
                f_i_flat = f_i.flatten()[:self.hd_dim]
                f_j_flat = f_j.flatten()[:self.hd_dim]

                # Pad if necessary
                if len(f_i_flat) < self.hd_dim:
                    f_i_flat = np.pad(f_i_flat, (0, self.hd_dim - len(f_i_flat)))
                if len(f_j_flat) < self.hd_dim:
                    f_j_flat = np.pad(f_j_flat, (0, self.hd_dim - len(f_j_flat)))

                # Concatenate features
                combined = torch.FloatTensor(
                    np.concatenate([f_i_flat, f_j_flat])
                ).to(self.config.device)

                # Predict refined edge weight
                with torch.no_grad():
                    neural_weight = self.edge_predictor(combined.unsqueeze(0)).item()

                # Combine with traditional similarity
                sim = np.dot(f_i_flat, f_j_flat) / (
                        np.linalg.norm(f_i_flat) * np.linalg.norm(f_j_flat) + 1e-8
                )
                hybrid_weight = 0.7 * sim + 0.3 * neural_weight

                self.weights[i, j] = self.weights[j, i] = hybrid_weight


class HybridMusicUCS:
    """Hybrid UCS system for music generation"""

    def __init__(self, config: HybridMusicConfig):
        self.config = config
        self.encoder = NeuralHDCMusicEncoder(config)
        self.temporal = NeuralTemporalMusic(config)
        self.graph = HybridMusicGraph(config)
        self.t = 0.0

        # Initialize Pygame
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Hybrid UCS Music Generation")

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)

        # Staff parameters
        self.staff_start = 200
        self.staff_spacing = 20

        # Memory for optimization
        self.max_graph_size = 10  # Limit graph size
        self.update_interval = 5  # Update graph every N frames
        self.frame_count = 0

        # Precompute projection matrix for HDC
        self.projection = torch.FloatTensor(
            np.random.randn(config.input_dimension, 128) / np.sqrt(config.input_dimension)
        ).to(config.device)  # Reduced HD dimension for performance

        # Note generation parameters
        self.note_duration = 0.2  # Faster note generation
        self.note_speed = 5  # Faster note movement
        self.note_trigger_x = 50

        # Note history and scale parameters
        self.note_history = []  # Store past notes with positions
        self.max_notes = 20  # Maximum notes to display at once

        # Scale now includes all 88 piano keys (MIDI notes 21 to 108)
        self.scale = list(range(21, 109))  # MIDI notes for all piano keys
        #  self.scale = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76]  # C4 to E5
        self.min_y = self.staff_start - 40  # Limit above top staff line
        self.max_y = self.staff_start + 6 * self.staff_spacing  # Limit below bottom staff line
        self.played_notes = set()

        # Musical state
        self.current_time = 0
        self.last_note_time = 0

        # Musical state
        self.current_time = 0
        self.note_duration = 0.5  # seconds
        self.last_note_time = 0

        # Initialize sounds dynamically
        self.sounds = {}
        sample_rate = 44100
        duration = 0.5  # seconds
        for note in self.scale:
            frequency = 440.0 * (2 ** ((note - 69) / 12.0))  # MIDI to frequency
            waveform = generate_sine_wave(frequency, duration, sample_rate)
            self.sounds[note] = pygame.mixer.Sound(waveform)

    def draw_staff(self):
        """Draw musical staff lines with measure bars (optimized)"""
        # Draw the five main staff lines efficiently
        staff_lines = [(50, self.staff_start + i * self.staff_spacing,
                        self.width - 50, self.staff_start + i * self.staff_spacing)
                       for i in range(5)]
        for start_x, start_y, end_x, end_y in staff_lines:
            pygame.draw.line(self.screen, self.BLACK, (start_x, start_y), (end_x, end_y), 1)

        # Draw measure bars (reduced frequency)
        for x in range(150, self.width - 50, 200):
            pygame.draw.line(self.screen, self.BLACK,
                             (x, self.staff_start),
                             (x, self.staff_start + 4 * self.staff_spacing), 1)

    def draw_notes(self):
        """Draw musical notes on the staff with movement, sound, note names, and ledger lines."""
        # Font for note names
        font = pygame.font.Font(None, 24)

        # Update note positions and remove off-screen notes
        updated_history = []
        for x, y, note, color in self.note_history:
            new_x = x - self.note_speed

            # Play sound when note crosses trigger line
            try:
                if (new_x <= self.note_trigger_x and
                        (new_x, note) not in self.played_notes and
                        self.sounds.get(note) is not None):
                    self.sounds[note].play(maxtime=300)
                    self.played_notes.add((new_x, note))
            except Exception as e:
                print(f"Error playing sound: {e}")

            if new_x > 0:  # Keep note if still on screen
                updated_history.append((new_x, y, note, color))

                # Draw ledger lines for notes outside the staff
                if y < self.staff_start - 4 * self.staff_spacing:  # Above the staff
                    ledger_y = self.staff_start - 4 * self.staff_spacing
                    while ledger_y > y:
                        pygame.draw.line(self.screen, self.BLACK, (new_x - 5, ledger_y), (new_x + 20, ledger_y), 1)
                        ledger_y -= self.staff_spacing
                elif y > self.staff_start:  # Below the staff
                    ledger_y = self.staff_start + self.staff_spacing
                    while ledger_y <= y:
                        pygame.draw.line(self.screen, self.BLACK, (new_x - 5, ledger_y), (new_x + 20, ledger_y), 1)
                        ledger_y += self.staff_spacing

                # Draw note stem and head
                pygame.draw.ellipse(self.screen, color, (new_x, y, 15, 10))
                pygame.draw.line(self.screen, color, (new_x + 15, y + 5), (new_x + 15, y - 30), 2)

                # Calculate note name (e.g., C4, D4)
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                octave = (note // 12) - 1
                note_name = f"{note_names[note % 12]}{octave}"

                # Draw note name above the note
                note_text = font.render(note_name, True, self.BLACK)
                self.screen.blit(note_text, (new_x - 5, y - 40))

                # Draw trigger line
                pygame.draw.line(self.screen, (200, 200, 200),
                                 (self.note_trigger_x, 0),
                                 (self.note_trigger_x, self.height), 1)

        self.note_history = updated_history[-self.max_notes:]  # Keep only recent notes

        # Clean up old played notes
        self.played_notes = {(x, note) for x, note in self.played_notes if x > 0}

    def process(self, input_notes: np.ndarray) -> np.ndarray:
        """Process input through hybrid system preserving UCS pipeline"""
        self.frame_count += 1

        # Convert input to tensor
        x_tensor = torch.FloatTensor(input_notes).to(self.config.device)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

        # 1. Enhanced HDC encoding (optimized)
        with torch.no_grad():
            hd_x = torch.tanh(x_tensor @ self.projection).cpu().numpy()

        # 2. Temporal processing (simplified)
        self.t += 0.01
        temporal_x = hd_x  # Simplified for performance

        # 3. Graph processing (only update periodically)
        if self.frame_count % self.update_interval == 0:
            node_id = len(self.graph.nodes)
            if node_id >= self.max_graph_size:
                self.graph.nodes = self.graph.nodes[1:]
                self.graph.node_features = {i: self.graph.node_features[i + 1]
                                            for i in range(len(self.graph.nodes))}
                node_id = len(self.graph.nodes)

            self.graph.nodes.append(node_id)
            self.graph.node_features[node_id] = temporal_x
            self.graph.update_weights(self.graph.node_features)

            # 4. Graph embedding (simplified)
            if self.graph.weights.shape[0] > 0:
                laplacian = sparse.eye(self.graph.weights.shape[0]) - self.graph.weights
                try:
                    dense_lap = laplacian.toarray()
                    eigenvals, eigenvects = np.linalg.eigh(dense_lap)
                    embedding = np.real(eigenvects[:, 1:5])
                except:
                    embedding = np.zeros((1, 4))
            else:
                embedding = np.zeros((1, 4))

        # 5. Combine features (use cached embedding if not updated)
        result = np.concatenate([temporal_x.flatten()[:4],
                                 embedding.flatten()[:4] if 'embedding' in locals()
                                 else np.zeros(4)])
        return result

    def run(self):
        """Main loop for music generation visualization"""
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Generate random notes for demonstration
            input_notes = np.random.binomial(1, 0.1, self.config.input_dimension)
            result = self.process(input_notes)

            # Update musical time
            self.current_time += 1 / 60  # Assuming 60 FPS

            # Process result into musical notes
            if self.current_time - self.last_note_time >= self.note_duration:
                # Generate note with uniform random selection
                if np.random.random() < 0.7:
                    note = np.random.choice(self.scale)

                    # Calculate y position based on MIDI note number
                    note_offset = note - 64  # Offset from E4 (first line of the treble staff)
                    y = self.staff_start - (note_offset * self.staff_spacing / 2)

                    # Add new note with color based on position in scale
                    color_val = int(255 * ((note - 21) / len(self.scale)))  # Normalize color based on position
                    color = (color_val, 0, 255 - color_val)

                    self.note_history.append((self.width - 50, y, note, color))

                self.last_note_time = self.current_time

            # Draw
            self.screen.fill(self.WHITE)
            self.draw_staff()
            self.draw_notes()

            # Draw clef
            font = pygame.font.Font(None, 72)
            text = font.render("𝄞", True, self.BLACK)
            self.screen.blit(text, (20, self.staff_start - 20))

            pygame.display.flip()

            clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    config = HybridMusicConfig()
    music_system = HybridMusicUCS(config)
    music_system.run()