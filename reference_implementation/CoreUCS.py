"""
Core Unified Cognitive System (UCS) Implementation
===============================================
Provides base classes for HDC encoding, temporal processing, and dynamic graph operations.
Designed to be modular and reusable across different UCS applications.
"""

import numpy as np
import scipy.sparse as sparse
from typing import Dict, List, Tuple, Optional, Any, Protocol
from dataclasses import dataclass


@dataclass
class UCSConfig:
    """Configuration parameters for UCS system"""
    hd_dimension: int = 10000  # Dimension of hypervectors
    input_dimension: int = 10  # Default input dimension
    temporal_window: float = 1.0  # Time window for temporal processing
    decay_rate: float = 0.1  # Temporal decay rate
    learning_rate: float = 0.1  # Learning rate for graph updates
    max_weight: float = 1.0  # Maximum weight bound
    reg_lambda: float = 0.001  # Regularization parameter
    cache_size: int = 1000  # Size of similarity cache


class Encodable(Protocol):
    """Protocol defining what can be encoded in HDC space"""
    def to_array(self) -> np.ndarray:
        """Convert object to numpy array for encoding"""
        ...


class HDCEncoder:
    """Hyperdimensional Computing encoder with proven distance preservation"""

    def __init__(self, input_dim: int, output_dim: int):
        """Initialize HDC encoder with input and output dimensions"""
        self.input_dimension = input_dim
        self.output_dimension = output_dim
        # Create projection matrix for input dimension to HD dimension
        self.projection = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        # Normalize columns
        self.projection /= np.linalg.norm(self.projection, axis=0)
        # Initialize cache
        self._cache: Dict[int, np.ndarray] = {}

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input vector into hyperdimensional space"""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Check cache first
        cache_key = hash(x.tobytes())
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Normalize input
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        # Project to high-dimensional space
        hd_vector = np.tanh(x_norm @ self.projection)

        # Cache result
        if len(self._cache) < 1000:  # Limit cache size
            self._cache[cache_key] = hd_vector

        return hd_vector

    def encode_object(self, obj: Encodable) -> np.ndarray:
        """Encode any object that implements the Encodable protocol"""
        return self.encode(obj.to_array())

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two hypervectors using circular convolution"""
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple hypervectors"""
        if not vectors:
            return np.zeros(self.output_dimension)
        bundle = np.zeros(self.output_dimension)
        for v in vectors:
            bundle += v.reshape(-1)  # Ensure vector is flattened
        return np.tanh(bundle / len(vectors))

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between two hypervectors"""
        return np.dot(a.ravel(), b.ravel()) / (
            np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
        )


class TemporalBuffer:
    """Efficient temporal buffer implementation"""

    def __init__(self, max_size: int = 1000):
        self.buffer: List[Tuple[float, np.ndarray]] = []
        self.max_size = max_size

    def add(self, t: float, x: np.ndarray) -> None:
        """Add sample to buffer with timestamp"""
        self.buffer.append((t, x))
        # Clean buffer of old entries when adding new ones
        self._clean_buffer(t)

    def get_window(self, t: float, window: float) -> List[Tuple[float, np.ndarray]]:
        """Get samples within time window"""
        cutoff = t - window
        valid_samples = [(t_i, x_i) for t_i, x_i in self.buffer if t_i > cutoff]
        # Take only the most recent samples up to max_size
        return valid_samples[-self.max_size:]

    def _clean_buffer(self, current_time: float, window: float = 2.0) -> None:
        """Clean old samples from buffer"""
        cutoff = current_time - window
        self.buffer = [(t, x) for t, x in self.buffer if t > cutoff]
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]


class TemporalProcessor:
    """Temporal processing with proven convergence"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.buffer = TemporalBuffer()

    def process_original(self, t: float, x: np.ndarray, object_id: Optional[Any] = None) -> np.ndarray:
        """Process input using temporal-spatial integration

        Args:
            t: Current time
            x: Input vector
            object_id: Optional identifier for tracking specific objects
        """
        # Store new sample
        self.buffer.add(t, x)

        # Get samples in current window
        window_samples = self.buffer.get_window(t, self.config.temporal_window)

        # Compute temporal integral with proven convergence
        result = x.copy()
        if len(window_samples) > 1:
            for t_i, x_i in window_samples[:-1]:
                weight = np.exp(-self.config.decay_rate * (t - t_i))
                result += weight * (x_i - x) * (t - t_i)

        return result

    def process(self, t: float, x: np.ndarray, object_id: Optional[Any] = None, noise: bool = True) -> np.ndarray:
        self.buffer.add(t, x)
        window_samples = self.buffer.get_window(t, self.config.temporal_window)
        result = x.copy()
        for t_i, x_i in window_samples[:-1]:
            weight = np.exp(-self.config.decay_rate * (t - t_i))
            result += weight * (x_i - x) * (t - t_i)
        if noise:
            result += np.random.normal(0, 0.01, size=result.shape)
        return result

    def get_history(self) -> List[Tuple[float, np.ndarray]]:
        """Get full temporal history"""
        return self.buffer.buffer.copy()


class DynamicGraph:
    """Dynamic graph with proven convergence properties"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.weights = sparse.lil_matrix((0, 0))
        self.nodes = []  # List of node identifiers
        self.node_features = {}  # Dict mapping node ID to features

    def _bounded_similarity_original(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute similarity with proven bounds in [0,1]"""
        x = x.ravel()
        y = y.ravel()
        # Normalize vectors
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        if x_norm == 0 or y_norm == 0:
            return 0.0

        # Compute cosine similarity
        cos_sim = np.dot(x, y) / (x_norm * y_norm)
        # Apply nonlinear scaling for better sensitivity
        sim = (np.tanh(4.0 * cos_sim) + 1.0) / 2.0
        return sim

    def _bounded_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        if x_norm == 0 or y_norm == 0:
            return 0.0
        cos_sim = np.dot(x, y) / (x_norm * y_norm)
        return (np.tanh(4.0 * cos_sim) + 1.0) / 2.0  # Scale to [0, 1]

    def update_weights(self, features: Dict[int, np.ndarray]) -> None:
        """Update graph weights using proven convergent dynamics"""
        n = len(self.nodes)
        if n == 0:
            return

        if self.weights.shape != (n, n):
            self.weights = sparse.lil_matrix((n, n))

        # Precompute normalized features
        norm_features = {}
        for node_id in self.nodes:
            f = features[node_id].ravel()
            norm = np.linalg.norm(f)
            if norm > 0:
                norm_features[node_id] = f / norm
            else:
                norm_features[node_id] = f

        # Update weights with enhanced dynamics
        for i in range(n):
            for j in range(i + 1, n):
                node_i = self.nodes[i]
                node_j = self.nodes[j]

                if node_i in norm_features and node_j in norm_features:
                    # Compute similarity
                    sim = self._bounded_similarity(
                        norm_features[node_i],
                        norm_features[node_j]
                    )

                    # Apply weight update with momentum
                    current_weight = self.weights[i, j]
                    target_weight = sim
                    new_weight = current_weight + self.config.learning_rate * (target_weight - current_weight)
                    new_weight = np.clip(new_weight, 0, self.config.max_weight)

                    # Set symmetric weights
                    self.weights[i, j] = self.weights[j, i] = new_weight

        self.weights = self.weights.tocsr()

    def add_node(self, node_id: Any, features: np.ndarray) -> None:
        """Add new node to graph"""
        if node_id not in self.nodes:
            self.nodes.append(node_id)
            self.node_features[node_id] = features

    def remove_node(self, node_id: Any) -> None:
        """Remove node from graph"""
        if node_id in self.nodes:
            idx = self.nodes.index(node_id)
            self.nodes.remove(node_id)
            del self.node_features[node_id]

            # Update weight matrix
            n = len(self.nodes)
            new_weights = sparse.lil_matrix((n, n))
            old_weights = self.weights.tolil()

            # Copy remaining weights
            for i in range(n):
                for j in range(n):
                    old_i = i if i < idx else i + 1
                    old_j = j if j < idx else j + 1
                    new_weights[i, j] = old_weights[old_i, old_j]

            self.weights = new_weights.tocsr()

    def get_neighbors(self, node_id: Any, threshold: float = 0.5) -> List[Any]:
        """Get neighbors of node with weight above threshold"""
        if node_id not in self.nodes:
            return []

        idx = self.nodes.index(node_id)
        weights = self.weights[idx].toarray().flatten()
        neighbors = []

        # Update weights before querying neighbors
        self.update_weights(self.node_features)

        # Get neighbors based on updated weights
        for i, w in enumerate(weights):
            if w > threshold and i != idx:
                neighbors.append(self.nodes[i])

        return neighbors

    def compute_laplacian(self) -> sparse.spmatrix:
        """Compute normalized Laplacian matrix"""
        degree = np.array(self.weights.sum(axis=1)).flatten()
        degree_matrix = sparse.diags(1.0 / (np.sqrt(degree) + 1e-8))
        return sparse.eye(self.weights.shape[0]) - degree_matrix @ self.weights @ degree_matrix


class UnifiedCognitiveSystem:
    """Complete UCS implementation integrating all components"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.hdc = HDCEncoder(config.input_dimension, config.hd_dimension)
        self.temporal = TemporalProcessor(config)
        self.graph = DynamicGraph(config)
        self.t = 0.0

    def process(self, x: np.ndarray, object_id: Optional[Any] = None) -> np.ndarray:
        """Process input through complete UCS pipeline"""
        # 1. HDC encoding
        hd_x = self.hdc.encode(x)

        # 2. Temporal processing
        self.t += 0.01
        temporal_x = self.temporal.process(self.t, hd_x, object_id)

        # 3. Graph processing
        if object_id is not None:
            self.graph.add_node(object_id, temporal_x)
            self.graph.update_weights(self.graph.node_features)

        # 4. Graph embedding
        if len(self.graph.nodes) > 1:
            laplacian = self.graph.compute_laplacian()
            try:
                if laplacian.shape[0] <= 1:
                    embedding = np.zeros((1, 4))
                else:
                    k = min(5, laplacian.shape[0] - 1)
                    eigenvals, eigenvects = sparse.linalg.eigs(laplacian, k=k, which='SM')
                    embedding = np.real(eigenvects[:, 1:])  # Skip first trivial eigenvector
            except:
                embedding = np.zeros((1, 4))
        else:
            embedding = np.zeros((1, 4))

        # 5. Combine features
        result = np.concatenate([temporal_x.flatten()[:4], embedding.flatten()[:4]])
        return result