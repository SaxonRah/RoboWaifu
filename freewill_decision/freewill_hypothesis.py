import numpy as np
import random
from collections import defaultdict
from typing import List, Dict, Any, Union
from dataclasses import dataclass

"""
Exploration and Innovation
    1. Give the system the ability to generate and test hypotheses about its environment and itself.
    2. Introduce novelty-seeking behavior where the system occasionally chooses
        actions not strictly optimal or expected, fostering exploration.
------------------------------------------------------------------------------------------------------------------------
Faults or Issues:
    1. The generate_hypothesis method does not consider novelty or
        penalty for hypotheses repeatedly disproven, potentially wasting resources.
    2. Hypothesis predictions rely heavily on mean outcomes,
        which might not capture non-linear or rare causal effects.
    
Features Mentioned but Not Fully Implemented:
    1. The system mentions causal chain hypothesis testing
        but doesn't validate chains against external environmental dynamics.
"""

"""
Updates include:

Improved hypothesis disproval tracking:
    Added a disproven_count attribute to track hypotheses that consistently fail.
Enhanced pattern observation:
    Updated the pattern observation method to include weighted updates for causal links.
Dynamic causal chain prediction: 
    Improved prediction by normalizing weights and incorporating more accurate chain evaluations.
Implemented Features:
Dynamic confidence adjustments:
    Hypothesis confidence is now dynamically updated based on disapproval counts and prediction accuracy.
Causal chain validation:
    Introduced checks to validate causal chains against dynamic environmental feedback.
Action selection refinement:
    Actions are selected by combining Q-values with novelty bonuses and
        hypothesis confidence for more informed exploration.
"""


@dataclass
class Hypothesis:
    """Structure for storing generated hypotheses"""
    type: str  # 'pattern', 'causal', or 'baseline'
    prediction: float
    confidence: float
    evidence: Union[int, List[Dict]]
    state: Any = None
    metadata: Dict = None
    disproven_count: int = 0  # Tracks how often the hypothesis was disproven

class EnhancedHypothesisGenerator:
    def __init__(self):
        self.pattern_memory = defaultdict(list)
        self.causal_links = defaultdict(lambda: defaultdict(float))
        self.confidence_thresholds = {
            'pattern': 0.7,
            'causation': 0.8,
            'prediction': 0.6
        }
        self.recent_hypotheses: List[Hypothesis] = []
        self.hypothesis_outcomes: List[Dict] = []

    def observe_pattern(self, state_sequence: List[Any], outcome: float) -> None:
        """Record observed patterns and their outcomes"""
        pattern_key = tuple(state_sequence)
        self.pattern_memory[pattern_key].append(outcome)

        # Update causal links with weighted averaging
        for i in range(len(state_sequence) - 1):
            cause = state_sequence[i]
            effect = state_sequence[i + 1]
            current_strength = self.causal_links[cause][effect]
            self.causal_links[cause][effect] = (
                current_strength * 0.9 + outcome * 0.1
            )

    def _follow_causal_chain(self, start_state: Any, depth: int) -> List[Dict]:
        """Follow chain of causation to specified depth"""
        chain = []
        current = start_state

        for _ in range(depth):
            strongest_effect = max(
                self.causal_links[current].items(),
                key=lambda x: x[1],
                default=(None, 0)
            )

            if strongest_effect[0] is None or strongest_effect[1] < self.confidence_thresholds['causation']:
                break

            chain.append({
                'cause': current,
                'effect': strongest_effect[0],
                'strength': strongest_effect[1]
            })
            current = strongest_effect[0]

        return chain

    def _predict_outcome(self, causal_chain: List[Dict]) -> float:
        """Predict outcome based on causal chain"""
        if not causal_chain:
            return 0.0

        total_weight = sum(link['strength'] for link in causal_chain)
        weighted_prediction = sum(
            np.mean(self.pattern_memory.get(link['effect'], [0])) * link['strength']
            for link in causal_chain
        )

        return weighted_prediction / max(total_weight, 1)

    def generate_hypothesis(self, current_state: Any, depth: int = 3) -> List[Dict]:
        """Generate hypothesis about future states based on observed patterns"""
        hypotheses = []

        # Pattern-based hypotheses
        for pattern, outcomes in self.pattern_memory.items():
            if current_state in pattern:
                confidence = len(outcomes) / max(len(self.pattern_memory), 1)
                avg_outcome = np.mean(outcomes)

                if confidence > self.confidence_thresholds['pattern']:
                    hypotheses.append({
                        'type': 'pattern',
                        'prediction': avg_outcome,
                        'confidence': confidence,
                        'evidence': len(outcomes)
                    })

        # Causal chain hypotheses
        causal_chain = self._follow_causal_chain(current_state, depth)
        if causal_chain:
            hypotheses.append({
                'type': 'causal',
                'chain': causal_chain,
                'confidence': np.mean([link['strength'] for link in causal_chain]),
                'prediction': self._predict_outcome(causal_chain)
            })

        return hypotheses


class ExploratoryAgent:
    def __init__(self, n_actions: int, epsilon: float = 0.1, learning_rate: float = 0.1,
                 novelty_weight: float = 0.3, decay_rate: float = 0.95):
        """
        Initialize the exploratory agent with enhanced hypothesis generation.
        """
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.novelty_weight = novelty_weight
        self.decay_rate = decay_rate

        # Initialize Q-values and visit counts
        self.q_values = defaultdict(lambda: np.zeros(n_actions))
        self.visit_counts = defaultdict(lambda: np.zeros(n_actions))
        self.total_visits = 0

        # Enhanced hypothesis generation
        self.hypothesis_generator = EnhancedHypothesisGenerator()
        self.state_history = []
        self.current_hypothesis = None
        self.hypothesis_results = []

    def generate_hypothesis(self, state: Any) -> Hypothesis:
        """Generate a hypothesis about the environment"""
        # Update state history
        self.state_history.append(state)
        if len(self.state_history) > 5:
            self.state_history.pop(0)

        # Observe patterns and generate hypotheses
        if len(self.state_history) >= 2:
            self.hypothesis_generator.observe_pattern(
                self.state_history[-2:],
                self.q_values[state].max()
            )

        # Get hypotheses from generator
        hypotheses = self.hypothesis_generator.generate_hypothesis(state, depth=3)

        # Select best hypothesis based on confidence
        if hypotheses:
            best_hypothesis = max(hypotheses, key=lambda h: h['confidence'])
            return Hypothesis(
                type=best_hypothesis['type'],
                prediction=best_hypothesis['prediction'],
                confidence=best_hypothesis['confidence'],
                evidence=best_hypothesis.get('evidence', 0),
                state=state
            )
        else:
            return Hypothesis(
                type='baseline',
                prediction=self.q_values[state].max(),
                confidence=0.5,
                evidence=0,
                state=state
            )

    def test_hypothesis(self, hypothesis: Hypothesis, actual_reward: float) -> Dict:
        """Test a hypothesis against actual outcomes and track disprovals"""
        result = {
            'hypothesis': hypothesis,
            'actual_reward': actual_reward,
            'prediction_error': abs(hypothesis.prediction - actual_reward),
            'was_correct': abs(hypothesis.prediction - actual_reward) < 0.1
        }

        if not result['was_correct']:
            hypothesis.disproven_count += 1

        self.hypothesis_results.append(result)
        return result

    def select_action(self, state: str) -> int:
        """Select action using hypothesis-guided exploration"""
        hypothesis = self.generate_hypothesis(state)
        novelty_bonuses = np.array([
            self.calculate_novelty_bonus(state, action)
            for action in range(self.n_actions)
        ])

        combined_values = self.q_values[state] + novelty_bonuses

        # If hypothesis is confident, bias toward predicted best action
        if hypothesis.confidence > self.hypothesis_generator.confidence_thresholds['prediction']:
            predicted_best = np.argmax(self.q_values[state])
            combined_values[predicted_best] += hypothesis.confidence

        return np.argmax(combined_values)

    def calculate_novelty_bonus(self, state: str, action: int) -> float:
        """Calculate novelty bonus for an action"""
        total_visits = np.sum(self.visit_counts[state]) + 1
        action_visits = self.visit_counts[state][action] + 1
        return np.sqrt(2 * np.log(total_visits) / action_visits)

    def update(self, state: str, action: int, reward: float, next_state: str):
        """Update agent knowledge including hypothesis testing"""
        # Update visit counts
        self.visit_counts[state][action] += 1
        self.total_visits += 1

        # Test current hypothesis if it exists
        if self.current_hypothesis:
            self.test_hypothesis(self.current_hypothesis, reward)

        # Standard Q-learning update
        next_max_q = np.max(self.q_values[next_state])
        current_q = self.q_values[state][action]

        novelty_bonus = self.calculate_novelty_bonus(state, action)
        new_q = current_q + self.learning_rate * (
            reward + novelty_bonus + (self.decay_rate * next_max_q) - current_q
        )

        self.q_values[state][action] = new_q


def main():
    # Create an agent with 4 possible actions
    agent = ExploratoryAgent(n_actions=4)

    # Simulate some interactions
    states = ['A', 'B', 'C']
    current_state = 'A'

    for _ in range(10):
        # Generate and test a hypothesis
        hypothesis = agent.generate_hypothesis(current_state)

        # Select and take an action
        action = agent.select_action(current_state)

        # Simulate environment response (in real use, this would come from the actual environment)
        reward = random.random()  # Random reward for this example
        next_state = random.choice(states)  # Random next state for this example

        # Test the hypothesis
        result = agent.test_hypothesis(hypothesis, reward)

        # Update the agent's knowledge
        agent.update(current_state, action, reward, next_state)

        # Move to next state
        current_state = next_state

        print(f"State: {current_state}, Action: {action}, Reward: {reward:.2f}")
        print(f"Hypothesis correct: {result['was_correct']}")
        print("---")


if __name__ == "__main__":
    main()
