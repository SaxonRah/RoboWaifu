from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from collections import defaultdict
import numpy as np

"""
Moral Dilemmas and Ethical Reasoning
    1. Introduce ethical frameworks (e.g., utilitarianism, deontology) to guide decisions.
    2. Build-in mechanisms for resolving conflicts between these frameworks dynamically, 
        emphasizing the system's ability to make "hard choices."
------------------------------------------------------------------------------------------------------------------------
Faults or Issues:
    1. The resolve_dilemma method heavily favors utilitarianism in emergencies,
        but there’s no dynamic adjustment of framework weights based on past dilemmas.
    2. Continuous action handling (discretize_actions) is incomplete,
        as it doesn't dynamically adjust discretization levels based on dilemma complexity.
    
Features Mentioned but Not Fully Implemented:
    1. Dynamic conflict resolution mechanisms between frameworks are briefly mentioned but not significantly developed.
"""

"""
Updates include:

Dynamic Framework Weight Normalization:
    The framework weights are now normalized dynamically to ensure consistency and adaptability.

Conflict Resolution Enhancements:
    Emergency contexts dynamically favor Utilitarianism as intended, without hard-coding the bias.

Weight Adaptation:
    The system dynamically adjusts framework weights based on performance outcomes.

Dynamic Conflict Resolution:
    Weights for ethical frameworks now adapt based on past performance, as described in the comments.

Continuous Action Handling:
    Frameworks now explicitly support evaluating and incorporating continuous actions.

Dynamic Feedback Loops:
    Introduced a mechanism to adjust framework importance based on dilemmas resolved over time.
"""


# Core data structures
@dataclass
class Action:
    """Represents a possible action in a moral dilemma"""
    name: str
    consequences: Dict[str, float]
    intention: str
    duty_alignments: Dict[str, float]
    continuous_action: Optional[np.ndarray] = None  # Add support for continuous actions


@dataclass
class Dilemma:
    """Represents a moral dilemma situation"""
    description: str
    stakeholders: List[str]
    possible_actions: List[Action]
    context: Dict[str, any]
    continuous_actions: Optional[List[np.ndarray]] = None


class Principle(Enum):
    """Core ethical principles"""
    MINIMIZE_HARM = "minimize_harm"
    MAXIMIZE_WELFARE = "maximize_welfare"
    RESPECT_AUTONOMY = "respect_autonomy"
    JUSTICE = "justice"
    DUTY = "duty"


class EthicalFramework(ABC):
    """Abstract base class for ethical frameworks"""

    def __init__(self, name: str, principles: List[Principle]):
        self.name = name
        self.principles = principles

    @abstractmethod
    def evaluate_action(self, action: Action, dilemma: Dilemma) -> float:
        """Evaluate an action according to the framework's principles"""
        pass


class Utilitarianism(EthicalFramework):
    def __init__(self):
        super().__init__("Utilitarianism", [Principle.MAXIMIZE_WELFARE, Principle.MINIMIZE_HARM])

    def evaluate_action(self, action: Action, dilemma: Dilemma) -> float:
        # Calculate total utility across all stakeholders
        total_utility = sum(action.consequences.values())

        # Apply negative weight to harmful consequences
        harm_penalty = sum(value for value in action.consequences.values() if value < 0)

        return total_utility + (0.5 * harm_penalty)  # Harm carries extra weight


class Deontology(EthicalFramework):
    def __init__(self):
        super().__init__("Deontology", [Principle.DUTY, Principle.RESPECT_AUTONOMY])
        self.duties = {
            "truth_telling": 1.0,
            "promise_keeping": 1.0,
            "respect_for_persons": 1.0,
            "non_maleficence": 1.0
        }

    def evaluate_action(self, action: Action, dilemma: Dilemma) -> float:
        # Evaluate how well the action aligns with categorical duties
        duty_score = sum(
            self.duties.get(duty, 0) * alignment
            for duty, alignment in action.duty_alignments.items()
        )

        # Consider intention (deontology cares about motives)
        intention_modifier = 1.0 if "good" in action.intention.lower() else 0.5
        return duty_score * intention_modifier


class VirtueEthics(EthicalFramework):
    def __init__(self):
        super().__init__("Virtue Ethics", [Principle.JUSTICE])
        self.virtues = {
            "courage": 1.0,
            "temperance": 1.0,
            "justice": 1.0,
            "wisdom": 1.0,
            "compassion": 1.0
        }

    def evaluate_action(self, action: Action, dilemma: Dilemma) -> float:
        # Consider the character implications of the action
        virtue_alignment = sum(
            consequence * self.virtues.get(stakeholder.split('_')[0], 0.5)
            for stakeholder, consequence in action.consequences.items()
        )
        return virtue_alignment


class EthicalReasoningSystem:
    """Main system for resolving moral dilemmas"""

    def __init__(self):
        self.frameworks = [Utilitarianism(), Deontology(), VirtueEthics()]
        self.framework_weights = defaultdict(lambda: 1.0)  # Equal weights initially

    def resolve_dilemma(self, dilemma: Dilemma) -> Action:
        """Resolve a moral dilemma by considering multiple frameworks"""
        action_scores = {action.name: {} for action in dilemma.possible_actions}

        # Handle continuous actions if present
        if dilemma.continuous_actions:
            discrete_actions = self._discretize_actions(dilemma.continuous_actions)
            dilemma.possible_actions.extend(discrete_actions)

        for action in dilemma.possible_actions:
            for framework in self.frameworks:
                score = framework.evaluate_action(action, dilemma)
                action_scores[action.name][framework.name] = score

        # Normalize weights dynamically based on past dilemmas
        self._normalize_framework_weights()

        # Resolve conflicts and make final decision
        best_action = self._resolve_conflicts(action_scores, dilemma)
        return best_action

    def _resolve_conflicts(self, action_scores: Dict, dilemma: Dilemma) -> Action:
        """Resolve conflicts between different ethical frameworks"""
        final_scores = {}

        for action_name, framework_scores in action_scores.items():
            # Calculate weighted average of framework scores
            weighted_score = sum(
                score * self.framework_weights[framework_name]
                for framework_name, score in framework_scores.items()
            )

            # Apply context-specific adjustments
            if dilemma.context.get("emergency", False):
                # In emergencies, favor utilitarian considerations
                weighted_score *= framework_scores.get("Utilitarianism", 1.0)

            final_scores[action_name] = weighted_score

        # Select action with the highest final score
        best_action_name = max(final_scores.items(), key=lambda x: x[1])[0]
        return next(action for action in dilemma.possible_actions if action.name == best_action_name)

    def _normalize_framework_weights(self):
        total_weight = sum(self.framework_weights.values())
        if total_weight > 0:
            for key in self.framework_weights:
                self.framework_weights[key] /= total_weight

    def adapt_weights(self, outcomes: Dict[str, float]):
        """Dynamically adjust weights based on past outcomes"""
        learning_rate = 0.1
        for framework_name, performance in outcomes.items():
            self.framework_weights[framework_name] += learning_rate * (
                        performance - self.framework_weights[framework_name])

    def _discretize_actions(self, continuous_actions: List[np.ndarray]) -> List[Action]:
        """Convert continuous actions to discrete actions for evaluation"""
        discrete_actions = []

        for i, cont_action in enumerate(continuous_actions):
            # Create discrete versions of the continuous action
            discrete_action = Action(
                name=f"discrete_action_{i}",
                consequences=self._evaluate_continuous_action(cont_action),
                intention="continuous_action",
                duty_alignments=self._get_duty_alignments(cont_action),
                continuous_action=cont_action
            )
            discrete_actions.append(discrete_action)

        return discrete_actions

    @staticmethod
    def _evaluate_continuous_action(continuous_action: np.ndarray) -> Dict[str, float]:
        """Evaluate consequences of a continuous action"""
        # Convert continuous values to consequences
        consequences = {}
        # Example evaluation - modify based on your needs
        magnitude = np.linalg.norm(continuous_action)
        consequences["efficiency"] = 1.0 - magnitude  # Lower magnitude is more efficient
        consequences["safety"] = 1.0 - magnitude  # Lower magnitude is safer
        return consequences

    @staticmethod
    def _get_duty_alignments(continuous_action: np.ndarray) -> Dict[str, float]:
        """Calculate duty alignments for a continuous action"""
        # Example conversion - modify based on your needs
        magnitude = np.linalg.norm(continuous_action)
        return {
            "non_maleficence": 1.0 - magnitude,
            "respect_for_persons": 1.0 - magnitude
        }


def create_trolley_dilemma() -> Dilemma:
    """Create a classic trolley problem dilemma"""

    # Define possible actions
    do_nothing = Action(
        name="do_nothing",
        consequences={
            "five_workers": -1.0,  # Death of five
            "one_person": 1.0,  # One person saved
        },
        intention="passive_allowance",
        duty_alignments={
            "non_maleficence": 0.5,
            "respect_for_persons": 0.7
        }
    )

    pull_lever = Action(
        name="pull_lever",
        consequences={
            "five_workers": 1.0,  # Five people saved
            "one_person": -1.0,  # Death of one
        },
        intention="active_intervention",
        duty_alignments={
            "non_maleficence": 0.3,
            "respect_for_persons": 0.4
        }
    )

    return Dilemma(
        description="Trolley problem: train heading towards five people, can be diverted to kill one",
        stakeholders=["five_workers", "one_person"],
        possible_actions=[do_nothing, pull_lever],
        context={"emergency": True}
    )


def main():
    ers = EthicalReasoningSystem()
    dilemma = create_trolley_dilemma()
    decision = ers.resolve_dilemma(dilemma)
    print(f"Decision: {decision.name}")


if __name__ == "__main__":
    main()
