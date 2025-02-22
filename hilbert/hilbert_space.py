import numpy as np
from typing import Callable, Any, List, Tuple


class ReducedHilbertSpace:
    def __init__(self,
                 problem_size: int,
                 reduction_ratio: float = 0.5,
                 min_dimension: int = 2):
        """
        Initialize a reduced Hilbert space.

        Args:
            problem_size: Original dimension of the problem
            reduction_ratio: Ratio for dimension reduction (0 < ratio <= 1)
            min_dimension: Minimum dimension of reduced space
        """
        self.original_dim = problem_size
        self.reduced_dim = max(min_dimension,
                               int(problem_size * reduction_ratio))
        self.state = np.zeros(self.reduced_dim, dtype=np.complex128)

    def embed(self,
              data: Any,
              embedding_function: Callable[[Any, int], np.ndarray]) -> np.ndarray:
        """
        Embed data into reduced Hilbert space using custom embedding function.

        Args:
            data: Data to embed
            embedding_function: Function that defines how to embed data

        Returns:
            Embedded vector in reduced space
        """
        embedded = embedding_function(data, self.reduced_dim)
        return self.normalize(embedded)

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector while preserving phase information"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def evolve_state(self,
                     current_state: np.ndarray,
                     evolution_function: Callable[[np.ndarray], np.ndarray],
                     steps: int = 1) -> np.ndarray:
        """
        Evolve the quantum state using custom evolution function.

        Args:
            current_state: Current state vector
            evolution_function: Function defining state evolution
            steps: Number of evolution steps

        Returns:
            Evolved state vector
        """
        state = current_state.copy()
        for _ in range(steps):
            state = evolution_function(state)
            state = self.normalize(state)
        return state

    def extract_solution(self,
                         state: np.ndarray,
                         extraction_function: Callable[[np.ndarray], Any]) -> Any:
        """
        Extract solution from quantum state using custom extraction function.

        Args:
            state: Current state vector
            extraction_function: Function defining how to extract solution

        Returns:
            Extracted solution in original problem space
        """
        return extraction_function(state)


# Example usage with custom functions
def example():
    # Example embedding function for numerical vectors
    def custom_embed(data: np.ndarray, reduced_dim: int) -> np.ndarray:
        # Simple modular mapping with phase encoding
        embedded = np.zeros(reduced_dim, dtype=np.complex128)
        for i, val in enumerate(data):
            idx = i % reduced_dim
            embedded[idx] += np.exp(2j * np.pi * val)
        return embedded

    # Example evolution function
    def custom_evolve(state: np.ndarray) -> np.ndarray:
        # Simple quantum walk-inspired evolution
        shifted = np.roll(state, 1)
        return state + 0.1 * shifted

    # Example extraction function
    def custom_extract(state: np.ndarray) -> List[float]:
        # Extract phases and convert to original space
        phases = np.angle(state) / (2 * np.pi)
        return phases.tolist()

    # Create instance
    reducer = ReducedHilbertSpace(problem_size=100, reduction_ratio=0.2)

    # Example data
    original_data = np.random.random(100)

    # Perform reduction and evolution
    embedded = reducer.embed(original_data, custom_embed)
    evolved = reducer.evolve_state(embedded, custom_evolve, steps=10)
    solution = reducer.extract_solution(evolved, custom_extract)

    return solution


example()
