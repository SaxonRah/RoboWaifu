from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import time
from collections import defaultdict

"""
Multi-Layered Value Systems
    1. Encode a hierarchy of values or goals into the system.
        These values can conflict (e.g., short-term vs. long-term priorities),
            forcing the system to "choose" based on importance, urgency, or other factors.
    2. Allow the system to redefine or prioritize values dynamically based on experience or external inputs.
------------------------------------------------------------------------------------------------------------------------
Faults or Issues:
    1. The resolve_conflict method uses static context multipliers,
        which might not adequately reflect dynamic environmental changes.
    2. Dependencies between values are mentioned but not actively considered in conflict resolution.

Features Mentioned but Not Fully Implemented:
    1. "Redefine or prioritize values dynamically" is implemented through manual context updates,
        not automated processes or learning.
"""


class ValueCategory(Enum):
    ETHICAL = "ethical"
    ECONOMIC = "economic"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"


class TimeFrame(Enum):
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


@dataclass
class Value:
    name: str
    category: ValueCategory
    base_importance: float  # 0 to 1
    time_frame: TimeFrame
    description: str
    dependencies: List[str] = None  # Names of other values this depends on

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class ValueSystem:
    def __init__(self):
        self.values: Dict[str, Value] = {}
        self.experience_log: List[dict] = []
        self.value_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.context_multipliers: Dict[str, float] = defaultdict(lambda: 1.0)

    def add_value(self, value: Value):
        """Add a new value to the system."""
        self.values[value.name] = value

    def log_experience(self, value_name: str, outcome: float, context: dict):
        """Log an experience related to a particular value."""
        self.experience_log.append({
            'value': value_name,
            'outcome': outcome,
            'context': context,
            'timestamp': time.time()
        })
        self._update_weights(value_name, outcome, context)

    def _update_weights(self, value_name: str, outcome: float, context: dict):
        """Update weights based on experience."""
        # Adjust the base weight for the value
        learning_rate = 0.1
        self.value_weights[value_name] += learning_rate * (outcome - self.value_weights[value_name])

        # Update context multipliers
        for context_key, context_value in context.items():
            context_id = f"{context_key}:{context_value}"
            self.context_multipliers[context_id] += learning_rate * (outcome - self.context_multipliers[context_id])

    def get_value_priority(self, value_name: str, current_context: dict) -> float:
        """Calculate the current priority of a value based on its weight and context."""
        if value_name not in self.values:
            raise ValueError(f"Value {value_name} not found in system")

        value = self.values[value_name]
        base_priority = value.base_importance * self.value_weights[value_name]

        # Apply context multipliers
        context_modifier = 1.0
        for context_key, context_value in current_context.items():
            context_id = f"{context_key}:{context_value}"
            context_modifier *= self.context_multipliers[context_id]

        # Consider time frame
        time_frame_multiplier = {
            TimeFrame.SHORT_TERM: 1.5,  # Immediate priority
            TimeFrame.MEDIUM_TERM: 1.0,  # Neutral priority
            TimeFrame.LONG_TERM: 0.7  # Lower immediate priority but still important
        }

        return base_priority * context_modifier * time_frame_multiplier[value.time_frame]

    def resolve_conflict(self, value_names: List[str], current_context: dict) -> str:
        """Resolve conflict between multiple values based on current priorities."""
        priorities = {
            name: self.get_value_priority(name, current_context)
            for name in value_names
        }
        return max(priorities.items(), key=lambda x: x[1])[0]


# Example usage
def main():
    # Initialize the value system
    system = ValueSystem()

    # Add some example values
    system.add_value(Value(
        name="profit",
        category=ValueCategory.ECONOMIC,
        base_importance=0.5,
        time_frame=TimeFrame.SHORT_TERM,
        description="Maximize immediate financial gains"
    ))

    system.add_value(Value(
        name="sustainability",
        category=ValueCategory.ENVIRONMENTAL,
        base_importance=0.7,
        time_frame=TimeFrame.LONG_TERM,
        description="Ensure long-term environmental sustainability",
        dependencies=["profit"]
    ))

    system.add_value(Value(
        name="social_impact",
        category=ValueCategory.SOCIAL,
        base_importance=0.6,
        time_frame=TimeFrame.MEDIUM_TERM,
        description="Positive impact on society"
    ))

    # Example context
    current_context = {
        "resource_scarcity": "high",
        "public_attention": "high",
        "financial_pressure": "medium"
    }

    # Log some experiences
    system.log_experience("profit", 0.5, {"resource_scarcity": "high"})
    system.log_experience("sustainability", 0.9, {"public_attention": "high"})

    # Resolve a conflict
    conflict_result = system.resolve_conflict(
        ["profit", "sustainability", "social_impact"],
        current_context
    )

    print(f"In current context, prioritize: {conflict_result}")

    # Get individual priorities
    for value_name in system.values:
        priority = system.get_value_priority(value_name, current_context)
        print(f"{value_name}: priority = {priority:.2f}")


if __name__ == "__main__":
    main()
