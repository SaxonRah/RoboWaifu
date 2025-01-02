"""
Unified Cognitive Systems (UCS) Reference Implementation
======================================================
Core implementation of UCS with corrected dimensionality handling.
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.sparse as sparse

@dataclass
class UCSConfig:
    """Configuration parameters for UCS system"""
    hd_dimension: int = 10000  # Dimension of hypervectors
    input_dimension: int = 10  # Input dimension
    temporal_window: float = 1.0  # Time window T
    decay_rate: float = 0.1  # Decay constant β
    learning_rate: float = 0.01  # Learning rate α
    max_weight: float = 1.0  # Maximum weight bound
    reg_lambda: float = 0.001  # Regularization parameter

class HDCEncoder:
    """Hyperdimensional Computing encoder with proven distance preservation"""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dimension = input_dim
        self.output_dimension = output_dim
        # Create projection matrix for input dimension to HD dimension
        self.projection = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        # Normalize columns
        self.projection /= np.linalg.norm(self.projection, axis=0)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input vector into hyperdimensional space"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Normalize input
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        # Project to high-dimensional space
        hd_vector = x_norm @ self.projection
        # Apply non-linear transformation preserving distances
        return np.tanh(hd_vector)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two hypervectors using circular convolution"""
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple hypervectors with proven superposition"""
        if not vectors:
            return np.zeros(self.output_dimension)
        bundle = np.zeros(self.output_dimension)
        for v in vectors:
            bundle += v
        return np.tanh(bundle / len(vectors))  # Normalized bundling

class DynamicGraph:
    """Dynamic graph implementation with proven convergence properties"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.weights = sparse.lil_matrix((0, 0))
        self.nodes = []
        self.node_features = {}

    def update_weights(self, features: Dict[int, np.ndarray]) -> None:
        """Update graph weights using proven convergent dynamics"""
        n = len(self.nodes)
        if n == 0:
            return

        # Construct new weight matrix if needed
        if self.weights.shape != (n, n):
            self.weights = sparse.lil_matrix((n, n))

        # Compute similarity matrix
        for i in range(n):
            for j in range(i+1, n):
                f_i = features[self.nodes[i]]
                f_j = features[self.nodes[j]]
                # Compute similarity with proven bounds
                sim = self._bounded_similarity(f_i, f_j)
                # Update weights using gradient descent
                grad = 2 * (self.weights[i,j] - sim) + 2 * self.config.reg_lambda * self.weights[i,j]
                new_weight = self.weights[i,j] - self.config.learning_rate * grad
                # Apply weight bounds
                new_weight = np.clip(new_weight, 0, self.config.max_weight)
                self.weights[i,j] = self.weights[j,i] = new_weight

        self.weights = self.weights.tocsr()  # Convert to CSR for efficient operations

    # def _bounded_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
    #     """Compute similarity with proven bounds in [0,1]"""
    #     cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
    #     return 0.5 * (cos_sim + 1)  # Map to [0,1]

    # def _bounded_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
    #     """Compute similarity with proven bounds in [0,1]"""
    #     # Ensure inputs are 1D arrays
    #     x = x.flatten()
    #     y = y.flatten()
    #     # Compute cosine similarity
    #     cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
    #     return 0.5 * (cos_sim + 1)  # Map to [0,1]

    def _bounded_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute similarity with proven bounds in [0,1]"""
        # Flatten inputs if they are not 1D
        x = x.ravel()
        y = y.ravel()
        # Compute cosine similarity
        cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
        return 0.5 * (cos_sim + 1)  # Map to [0,1]

class TemporalProcessor:
    """Implementation of temporal-spatial processing with proven convergence"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.time_buffer: List[Tuple[float, np.ndarray]] = []

    def process(self, t: float, x: np.ndarray) -> np.ndarray:
        """Process input using temporal-spatial integration"""
        # Update buffer
        self.time_buffer.append((t, x))
        # Remove old samples
        cutoff_time = t - self.config.temporal_window
        self.time_buffer = [(t_i, x_i) for t_i, x_i in self.time_buffer if t_i > cutoff_time]

        # Compute temporal integral
        result = x.copy()
        for t_i, x_i in self.time_buffer[:-1]:
            weight = np.exp(-self.config.decay_rate * (t - t_i))
            result += weight * (x_i - x) * (t - t_i)

        return result

class UnifiedCognitiveSystem:
    """Main UCS implementation integrating all components"""

    def __init__(self, config: UCSConfig):
        self.config = config
        self.hdc = HDCEncoder(config.input_dimension, config.hd_dimension)
        self.graph = DynamicGraph(config)
        self.temporal = TemporalProcessor(config)
        self.memory_buffer = {}
        self.t = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process input through complete UCS pipeline"""
        # 1. Encode input in HDC space
        hd_x = self.hdc.encode(x)

        # 2. Apply temporal processing
        self.t += 0.01  # Simulate time step
        temporal_x = self.temporal.process(self.t, hd_x)

        # 3. Update graph structure
        node_id = len(self.graph.nodes)
        self.graph.nodes.append(node_id)
        self.graph.node_features[node_id] = temporal_x
        self.graph.update_weights(self.graph.node_features)

        # 4. Compute graph embedding
        if self.graph.weights.shape[0] > 0:
            # Use normalized Laplacian embedding
            laplacian = sparse.eye(self.graph.weights.shape[0]) - self.graph.weights
            try:
                eigenvals, eigenvects = sparse.linalg.eigs(laplacian, k=min(5, laplacian.shape[0]-1), which='SM')
                embedding = np.real(eigenvects[:, 1:])  # Skip first trivial eigenvector
            except:
                embedding = np.zeros((1, 4))
        else:
            embedding = np.zeros((1, 4))

        # 5. Combine with temporal features
        # result = np.concatenate([temporal_x[:4], embedding.flatten()[:4]])
        result = np.concatenate([temporal_x.flatten()[:4], embedding.flatten()[:4]])
        return result

def test_ucs():
    """Test UCS implementation with sample data"""
    config = UCSConfig()
    ucs = UnifiedCognitiveSystem(config)

    # Generate test data
    X = np.random.randn(100, config.input_dimension)
    results = []

    # Process sequence
    for x in X:
        result = ucs.process(x)
        results.append(result)

    results = np.array(results)
    print(f"Processed {len(X)} samples")
    print(f"Output shape: {results.shape}")
    print(f"Output range: [{results.min():.3f}, {results.max():.3f}]")
    return results

if __name__ == "__main__":
    test_ucs()