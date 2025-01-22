import random
import numpy as np
from enum import Enum

"""
Introduce Degrees of Randomness and Determinism
    1. Use pseudo-random generators or quantum randomness for decisions.
    2. Combine randomness with rule-based systems to add a layer of structured unpredictability.
    3. Decision-making could weigh deterministic processes against stochastic ones,
        using thresholds or probability distributions to favor one over the other.
"""


class DecisionType(Enum):
    DETERMINISTIC = "deterministic"
    RANDOM = "random"
    HYBRID = "hybrid"


class DecisionMaker:
    def __init__(self, random_weight=0.5, random_seed=None):
        """
        Initialize the decision maker with configurable randomness weight.

        Args:
            random_weight (float): Weight given to random decisions (0 to 1)
            random_seed (int): Optional seed for reproducibility
        """
        self.random_weight = random_weight
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    @staticmethod
    def _deterministic_decision(inputs):
        """
        Make a decision based on deterministic rules with safe normalization.

        Args:
            inputs (dict): Dictionary of input parameters
        Returns:
            float: Decision score between 0 and 1
        """
        if not inputs:
            return 0.5  # Default score for empty inputs

        # Find max value safely
        max_value = max(abs(v) for v in inputs.values()) if inputs else 1.0

        # Avoid division by zero
        if max_value == 0:
            normalized_inputs = {k: 0.0 for k in inputs}
        else:
            normalized_inputs = {k: abs(v) / max_value for k, v in inputs.items()}

        # Weighted sum based on input values (support for positive and negative influences)
        weighted_sum = sum(v * (1 if inputs[k] > 0 else -1) for k, v in normalized_inputs.items())

        # Normalize to a 0-1 range
        return max(0, min(1, 0.5 + weighted_sum / 2))

    @staticmethod
    def _random_decision():
        """
        Make a purely random decision.
        Returns:
            float: Random value between 0 and 1
        """
        return random.random()

    def _hybrid_decision(self, inputs):
        """
        Combine random and deterministic decisions.

        Args:
            inputs (dict): Dictionary of input parameters
        Returns:
            float: Combined decision score between 0 and 1
        """
        deterministic_score = self._deterministic_decision(inputs)
        random_score = self._random_decision()

        return (self.random_weight * random_score +
                (1 - self.random_weight) * deterministic_score)

    def make_decision(self, inputs, decision_type=DecisionType.HYBRID, threshold=0.5):
        """
        Make a decision based on the specified type.

        Args:
            inputs (dict): Dictionary of input parameters
            decision_type (DecisionType): Type of decision to make
            threshold (float): Threshold for binary decisions
        Returns:
            tuple: (decision score, binary decision)
        """
        if decision_type == DecisionType.DETERMINISTIC:
            score = self._deterministic_decision(inputs)
        elif decision_type == DecisionType.RANDOM:
            score = self._random_decision()
        else:  # HYBRID
            score = self._hybrid_decision(inputs)

        return score, score >= threshold
    @staticmethod
    def quantum_random_decision():
        """
        Simulate quantum randomness by leveraging numpy's advanced random functions.
        Returns:
            float: Quantum-like random value between 0 and 1
        """
        return np.random.uniform(0, 1)

    def make_quantum_decision(self, inputs, threshold=0.5):
        """
        Make a decision using quantum-like randomness blended with deterministic rules.

        Args:
            inputs (dict): Dictionary of input parameters
            threshold (float): Threshold for binary decisions
        Returns:
            tuple: (decision score, binary decision)
        """
        deterministic_score = self._deterministic_decision(inputs)
        quantum_score = self.quantum_random_decision()

        score = (self.random_weight * quantum_score +
                 (1 - self.random_weight) * deterministic_score)

        return score, score >= threshold


# Example usage
def main():
    # Create a decision maker with 30% randomness
    decision_maker = DecisionMaker(random_weight=0.3, random_seed=42)

    # Sample input parameters
    test_inputs = {
        'parameter1': 75,
        'parameter2': -30,
        'parameter3': 90
    }

    # Test different decision types
    for decision_type in DecisionType:
        score, decision = decision_maker.make_decision(
            test_inputs,
            decision_type=decision_type
        )
        print(f"\n{decision_type.value.title()} Decision:")
        print(f"Score: {score:.3f}")
        print(f"Binary Decision: {decision}")

    # Test quantum decision-making
    quantum_score, quantum_decision = decision_maker.make_quantum_decision(test_inputs)
    print("\nQuantum Decision:")
    print(f"Score: {quantum_score:.3f}")
    print(f"Binary Decision: {quantum_decision}")


if __name__ == "__main__":
    main()
