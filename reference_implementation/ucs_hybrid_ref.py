"""
True Hybrid UCS-Neural Network Implementation
===========================================
Enhances UCS with neural components while preserving core UCS architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import scipy.sparse as sparse

from ucs_ref import HDCEncoder, TemporalProcessor, DynamicGraph


@dataclass
class HybridConfig:
    """Configuration for hybrid UCS-NN system"""
    hd_dimension: int = 10000
    input_dimension: int = 10
    temporal_window: float = 1.0
    decay_rate: float = 0.1
    learning_rate: float = 0.01
    max_weight: float = 1.0
    reg_lambda: float = 0.001
    nn_hidden_dim: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralHDCEncoder(nn.Module):
    """Enhanced HDC encoder with neural preprocessing"""

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        # Neural preprocessing
        self.preprocessor = nn.Sequential(
            nn.Linear(config.input_dimension, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, config.input_dimension)
        ).to(config.device)

        # Traditional HDC encoder
        self.hdc = HDCEncoder(config.input_dimension, config.hd_dimension)

    def forward(self, x: torch.Tensor) -> np.ndarray:
        # Neural preprocessing
        enhanced_x = self.preprocessor(x)
        # Convert to numpy for HDC encoding
        enhanced_x_np = enhanced_x.detach().cpu().numpy()
        # HDC encoding
        return self.hdc.encode(enhanced_x_np)


class NeuralTemporalProcessor(nn.Module):
    """Enhanced temporal processor with neural attention"""

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.temporal_ucs = TemporalProcessor(config)

        # Neural attention for temporal weighting
        self.attention = nn.Sequential(
            nn.Linear(config.hd_dimension, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, 1),
            nn.Sigmoid()
        ).to(config.device)

    def forward(self, t: float, x: np.ndarray, buffer: List[Tuple[float, np.ndarray]]) -> np.ndarray:
        # Get UCS temporal processing
        ucs_result = self.temporal_ucs.process(t, x)

        # Apply neural attention to temporal samples
        if buffer:
            times, samples = zip(*buffer)
            samples_tensor = torch.FloatTensor(np.array(samples)).to(self.config.device)
            attention_weights = self.attention(samples_tensor)
            weighted_samples = (samples_tensor * attention_weights).mean(dim=0)
            attended_result = weighted_samples.detach().cpu().numpy()

            # Combine UCS and neural results
            alpha = 0.7  # Weight for UCS result
            return alpha * ucs_result + (1 - alpha) * attended_result

        return ucs_result


class HybridDynamicGraph:
    """Enhanced dynamic graph with neural edge prediction"""

    def __init__(self, config: HybridConfig):
        self.config = config
        self.graph_ucs = DynamicGraph(config)

        # Neural edge predictor with correct input dimension
        feature_dim = config.hd_dimension
        self.edge_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, config.nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.nn_hidden_dim, 1),
            nn.Sigmoid()
        ).to(config.device)

        self.optimizer = torch.optim.Adam(self.edge_predictor.parameters())

    def update_weights(self, features: Dict[int, np.ndarray]) -> None:
        # Update UCS graph
        self.graph_ucs.update_weights(features)

        # Neural edge weight refinement
        n = len(features)
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    f_i = features[i]
                    f_j = features[j]

                    # Ensure correct feature dimensionality
                    f_i_flat = f_i.flatten()[:self.config.hd_dimension]
                    f_j_flat = f_j.flatten()[:self.config.hd_dimension]

                    # Pad if necessary
                    if len(f_i_flat) < self.config.hd_dimension:
                        f_i_flat = np.pad(f_i_flat, (0, self.config.hd_dimension - len(f_i_flat)))
                    if len(f_j_flat) < self.config.hd_dimension:
                        f_j_flat = np.pad(f_j_flat, (0, self.config.hd_dimension - len(f_j_flat)))

                    # Concatenate features for edge prediction
                    combined = torch.FloatTensor(np.concatenate([f_i_flat, f_j_flat])).to(self.config.device)

                    # Predict refined edge weight
                    with torch.no_grad():
                        neural_weight = self.edge_predictor(combined.unsqueeze(0)).item()

                    # Combine with UCS weight
                    alpha = 0.7  # Weight for UCS result
                    hybrid_weight = alpha * self.graph_ucs.weights[i, j] + (1 - alpha) * neural_weight

                    # Update weight matrix
                    self.graph_ucs.weights[i, j] = self.graph_ucs.weights[j, i] = hybrid_weight


class HybridUCS:
    """True hybrid system preserving UCS architecture with neural enhancements"""

    def __init__(self, config: HybridConfig):
        self.config = config

        # Enhanced components
        self.encoder = NeuralHDCEncoder(config)
        self.temporal = NeuralTemporalProcessor(config)
        self.graph = HybridDynamicGraph(config)

        self.memory_buffer = {}
        self.t = 0.0

        # Optimizers
        self.optimizer = torch.optim.Adam(
            list(self.encoder.preprocessor.parameters()) +
            list(self.temporal.attention.parameters()) +
            list(self.graph.edge_predictor.parameters())
        )

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process input through hybrid system preserving UCS pipeline"""
        # Convert input to tensor
        x_tensor = torch.FloatTensor(x).to(self.config.device)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

        # 1. Enhanced HDC encoding
        hd_x = self.encoder(x_tensor)

        # 2. Enhanced temporal processing
        self.t += 0.01
        temporal_x = self.temporal(self.t, hd_x, self.temporal.temporal_ucs.time_buffer)

        # 3. Enhanced graph processing
        node_id = len(self.graph.graph_ucs.nodes)
        self.graph.graph_ucs.nodes.append(node_id)
        self.graph.graph_ucs.node_features[node_id] = temporal_x
        self.graph.update_weights(self.graph.graph_ucs.node_features)

        # 4. Graph embedding (preserving UCS approach)
        if self.graph.graph_ucs.weights.shape[0] > 0:
            laplacian = sparse.eye(self.graph.graph_ucs.weights.shape[0]) - self.graph.graph_ucs.weights
            try:
                eigenvals, eigenvects = sparse.linalg.eigs(laplacian, k=min(5, laplacian.shape[0] - 1), which='SM')
                embedding = np.real(eigenvects[:, 1:])
            except:
                embedding = np.zeros((1, 4))
        else:
            embedding = np.zeros((1, 4))

        # 5. Combine features (preserving UCS output format)
        result = np.concatenate([temporal_x.flatten()[:4], embedding.flatten()[:4]])
        return result



def test_hybrid_ucs():
    """Test Hybrid UCS implementation with sample data"""
    config = HybridConfig()
    hybrid = HybridUCS(config)

    # Generate test data
    X = np.random.randn(100, config.input_dimension)
    results = []

    # Process sequence
    for x in X:
        result = hybrid.process(x)
        results.append(result)

    results = np.array(results)
    print(f"Processed {len(X)} samples")
    print(f"Output shape: {results.shape}")
    print(f"Output range: [{results.min():.3f}, {results.max():.3f}]")
    return results


def test_hybrid_encoding_similarity():
    """Test hybrid encoding similarity with noise perturbation"""
    config = HybridConfig()
    hybrid = HybridUCS(config)

    # Generate test data
    X = np.random.randn(100, config.input_dimension)
    noise = np.random.normal(0, 0.01, X.shape)
    X_prime = X + noise

    similarities = []
    for x, x_prime in zip(X, X_prime):
        hd_x = hybrid.encoder(torch.FloatTensor(x).to(config.device)).flatten()  # Flatten to 1D
        hd_x_prime = hybrid.encoder(torch.FloatTensor(x_prime).to(config.device)).flatten()  # Flatten to 1D

        # Compute cosine similarity
        cos_sim = np.dot(hd_x, hd_x_prime) / (np.linalg.norm(hd_x) * np.linalg.norm(hd_x_prime) + 1e-8)
        similarities.append(cos_sim)

    print(f"Average Cosine Similarity: {np.mean(similarities):.3f}")


def test_hybrid_temporal_processing():
    """Test hybrid temporal processing with sinusoidal input"""
    config = HybridConfig(temporal_window=2.0, decay_rate=0.5)
    hybrid = HybridUCS(config)

    t = np.linspace(0, 10, 500)  # Time steps
    x = np.sin(2 * np.pi * 0.5 * t)  # Sinusoidal input
    results = []

    for i, val in enumerate(x):
        # Ensure proper shape for the input
        input_tensor = np.array([val]).reshape(1, -1)  # Shape (1, 1)
        # Match hd_dimension
        expanded_input = np.pad(input_tensor, ((0, 0), (0, config.hd_dimension - 1)), constant_values=0)

        result = hybrid.temporal(
            t[i],
            expanded_input,
            hybrid.temporal.temporal_ucs.time_buffer
        )
        results.append(result)

    results = np.array(results)
    print(f"Processed Temporal Dynamics Shape: {results.shape}")


def test_hybrid_graph_clustering():
    """Test hybrid graph clustering capabilities"""
    config = HybridConfig()
    hybrid = HybridUCS(config)

    # Generate clustered data
    X, y = make_blobs(n_samples=100, centers=3, n_features=config.input_dimension, random_state=42)

    embeddings = []
    for x in X:
        result = hybrid.process(x)
        embeddings.append(result)

    embeddings = np.array(embeddings)

    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=y, cmap='viridis')
    plt.title("Hybrid Graph Embeddings Clustering")
    plt.show()


def test_hybrid_profile_efficiency():
    """Profile hybrid system efficiency"""
    import time

    config = HybridConfig()
    hybrid = HybridUCS(config)

    # Generate input data
    X = np.random.randn(100, config.input_dimension)

    # Profile Hybrid UCS
    start_time = time.time()
    for x in X:
        hybrid.process(x)
    hybrid_time = time.time() - start_time

    print(f"Hybrid UCS Inference Time: {hybrid_time:.3f} seconds")


if __name__ == "__main__":
    test_hybrid_ucs()
    # Expected Result: Hybrid UCS processes samples and produces valid outputs.

    test_hybrid_encoding_similarity()
    # Expected Result: High cosine similarity, indicating robustness of the hybrid encoding.

    test_hybrid_temporal_processing()
    # Expected Result: Smooth integration of temporal features, showing proper decay and neural attention.

    test_hybrid_graph_clustering()
    # Expected Result: Data points from the same cluster are close together in the embedding space.

    test_hybrid_profile_efficiency()
    # Expected Result: Reasonable inference time, considering added neural components.
