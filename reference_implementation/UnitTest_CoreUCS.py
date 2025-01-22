"""
Test suite for Core UCS Module
=============================
Comprehensive tests for HDC encoding, temporal processing, and dynamic graph operations.
"""

import unittest
import numpy as np
import scipy.sparse as sparse
from dataclasses import dataclass
from typing import List

# Import core UCS module (assuming it's in core_ucs.py)
from CoreUCS import (
    UCSConfig, HDCEncoder, TemporalProcessor,
    TemporalBuffer, DynamicGraph, UnifiedCognitiveSystem,
    Encodable
)


# Test helper class implementing Encodable protocol
@dataclass
class TestVector(Encodable):
    values: List[float]

    def to_array(self) -> np.ndarray:
        return np.array(self.values)


class TestHDCEncoder(unittest.TestCase):
    """Test HDC encoding functionality"""

    def setUp(self):
        self.input_dim = 4
        self.output_dim = 100
        self.encoder = HDCEncoder(self.input_dim, self.output_dim)

    def test_encoding_dimensions(self):
        """Test output dimensions of encoded vectors"""
        x = np.random.randn(self.input_dim)
        encoded = self.encoder.encode(x)
        self.assertEqual(encoded.shape[-1], self.output_dim)

        # Test batch encoding
        batch = np.random.randn(5, self.input_dim)
        encoded_batch = self.encoder.encode(batch)
        self.assertEqual(encoded_batch.shape[-1], self.output_dim)

    def test_encoding_bounds(self):
        """Test that encoded vectors are properly bounded"""
        x = np.random.randn(self.input_dim)
        encoded = self.encoder.encode(x)
        self.assertTrue(np.all(np.abs(encoded) <= 1.0))

    def test_encoding_stability(self):
        """Test that same input produces same encoding"""
        x = np.random.randn(self.input_dim)
        encoded1 = self.encoder.encode(x)
        encoded2 = self.encoder.encode(x)
        np.testing.assert_array_almost_equal(encoded1, encoded2)

    def test_encoding_similarity(self):
        """Test that similar inputs produce similar encodings"""
        x = np.random.randn(self.input_dim)
        x_noisy = x + 0.1 * np.random.randn(self.input_dim)

        encoded1 = self.encoder.encode(x)
        encoded2 = self.encoder.encode(x_noisy)

        similarity = self.encoder.similarity(encoded1, encoded2)
        self.assertTrue(similarity > 0.9)  # Similar vectors should have high similarity

    def test_bind_operation(self):
        """Test binding operation properties"""
        a = self.encoder.encode(np.random.randn(self.input_dim))
        b = self.encoder.encode(np.random.randn(self.input_dim))

        bound = self.encoder.bind(a, b)
        self.assertEqual(bound.shape, a.shape)
        self.assertTrue(np.all(np.isfinite(bound)))

    def test_bundle_operation(self):
        """Test bundling operation properties"""
        vectors = [
            self.encoder.encode(np.random.randn(self.input_dim))
            for _ in range(5)
        ]

        bundled = self.encoder.bundle(vectors)
        self.assertEqual(bundled.shape[-1], self.output_dim)
        self.assertTrue(np.all(np.abs(bundled) <= 1.0))

    def test_encodable_protocol(self):
        """Test encoding of objects implementing Encodable"""
        test_obj = TestVector([1.0, 2.0, 3.0, 4.0])
        encoded = self.encoder.encode_object(test_obj)
        self.assertEqual(encoded.shape[-1], self.output_dim)


class TestTemporalProcessor(unittest.TestCase):
    """Test temporal processing functionality"""

    def setUp(self):
        self.config = UCSConfig(
            temporal_window=5.0,
            decay_rate=0.05
        )
        self.processor = TemporalProcessor(self.config)

    def test_temporal_buffer(self):
        """Test temporal buffer operations"""
        buffer = TemporalBuffer(max_size=3)

        # Test adding samples
        buffer.add(0.0, np.array([1.0]))
        buffer.add(0.5, np.array([2.0]))
        buffer.add(1.0, np.array([3.0]))

        self.assertEqual(len(buffer.buffer), 3)

        # Test max size enforcement
        buffer.add(1.5, np.array([4.0]))
        self.assertEqual(len(buffer.buffer), 3)
        self.assertEqual(buffer.buffer[0][1][0], 2.0)

    def test_temporal_processing(self):
        """Test temporal processing behavior"""
        x = np.array([1.0, 0.0])

        # Process single sample
        result1 = self.processor.process(0.0, x, noise=False)
        np.testing.assert_array_almost_equal(result1, x)
        # np.testing.assert_allclose(result1, x, atol=1e-2)  # Tolerate deviations up to 0.01

        # Process sequence
        result2 = self.processor.process(0.1, x * 2)
        self.assertTrue(np.any(result2 != x))

    def test_temporal_decay(self):
        """Test temporal decay behavior"""
        x = np.array([1.0])

        # Add sequence of samples
        times = np.linspace(0, 2, 10)
        for t in times:
            result = self.processor.process(t, x)

        # Check that older samples have less influence
        history = self.processor.get_history()
        self.assertLess(len(history), len(times))  # Should have removed old samples


class TestDynamicGraph(unittest.TestCase):
    """Test dynamic graph functionality"""

    def setUp(self):
        self.config = UCSConfig()
        self.graph = DynamicGraph(self.config)

    def test_graph_construction(self):
        """Test basic graph operations"""
        # Add nodes
        features1 = np.array([1.0, 0.0])
        features2 = np.array([0.0, 1.0])

        self.graph.add_node(0, features1)
        self.graph.add_node(1, features2)

        self.assertEqual(len(self.graph.nodes), 2)

    def test_weight_updates(self):
        """Test weight matrix updates"""
        features = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0])
        }

        self.graph.add_node(0, features[0])
        self.graph.add_node(1, features[1])
        self.graph.update_weights(features)

        # Check weight matrix properties
        self.assertEqual(self.graph.weights.shape, (2, 2))
        self.assertTrue(np.all(self.graph.weights.toarray() >= 0))
        self.assertTrue(np.all(self.graph.weights.toarray() <= self.config.max_weight))

    def test_node_removal(self):
        """Test node removal behavior"""
        # Add nodes
        features = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([1.0, 1.0])
        }

        for i in range(3):
            self.graph.add_node(i, features[i])

        self.graph.update_weights(features)

        # Remove middle node
        self.graph.remove_node(1)
        self.assertEqual(len(self.graph.nodes), 2)
        self.assertEqual(self.graph.weights.shape, (2, 2))

    def test_neighbor_query(self):
        """Test neighbor querying"""
        features = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.9, 0.1]),  # Similar to 0
            2: np.array([0.0, 1.0])  # Different
        }

        for i in range(3):
            self.graph.add_node(i, features[i])

        self.graph.update_weights(features)

        # Get neighbors of node 0
        neighbors = self.graph.get_neighbors(0, threshold=0.05)
        self.assertIn(1, neighbors)  # Node 1 should be similar
        self.assertNotIn(2, neighbors)  # Node 2 should be different


class TestUnifiedCognitiveSystem(unittest.TestCase):
    """Test complete UCS functionality"""

    def setUp(self):
        self.config = UCSConfig(
            input_dimension=4,
            hd_dimension=100
        )
        self.ucs = UnifiedCognitiveSystem(self.config)

    def test_complete_pipeline(self):
        """Test full UCS processing pipeline"""
        # Create input sequence
        x = np.random.randn(self.config.input_dimension)

        # Process without object ID
        result1 = self.ucs.process(x)
        self.assertTrue(np.all(np.isfinite(result1)))

        # Process with object ID
        result2 = self.ucs.process(x, object_id=0)
        self.assertTrue(np.all(np.isfinite(result2)))

    def test_temporal_consistency(self):
        x = np.random.randn(self.config.input_dimension)

        # Process input over a longer duration with larger intervals
        results = [self.ucs.process(x, object_id=0) for _ in range(50)]

        # Debugging output
        print("Temporal Processing Results:", results)

        # Ensure at least one result differs due to temporal effects
        self.assertTrue(any(not np.array_equal(r1, r2) for r1, r2 in zip(results[:-1], results[1:])))

        # self.assertTrue(
        #     any(
        #         np.linalg.norm(r1 - r2) > 1e-5  # Check for small but significant changes
        #         for r1, r2 in zip(results[:-1], results[1:])
        #     )
        # )

    def test_multiple_objects(self):
        """Test processing of multiple objects"""
        # Create two distinct inputs
        x1 = np.array([1.0, 0.0, 0.0, 0.0])
        x2 = np.array([0.0, 1.0, 0.0, 0.0])

        # Process both objects
        result1 = self.ucs.process(x1, object_id=0)
        result2 = self.ucs.process(x2, object_id=1)

        # Results should be different
        self.assertTrue(not np.array_equal(result1, result2))


if __name__ == '__main__':
    unittest.main()
