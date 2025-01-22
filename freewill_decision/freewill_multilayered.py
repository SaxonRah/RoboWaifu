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

"""
Updates include:

Dynamic Context Multipliers:
    Adjusted based on trends in feedback for more accurate prioritization.
    
Dependency Consideration:
    Dependencies between values actively influence priority calculations.
    
Dynamic Redefinition of Values:
    Values evolve over time, adjusting their importance based on cumulative experience.
    
Dynamic Time Frame Adjustments:
    Enhanced priority calculation by considering dynamic time frame importance.
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
        """Update weights and context multipliers based on experience."""
        learning_rate = 0.1
        self.value_weights[value_name] += learning_rate * (outcome - self.value_weights[value_name])

        # Dynamically adjust context multipliers based on trends in feedback
        for context_key, context_value in context.items():
            context_id = f"{context_key}:{context_value}"
            past_feedback = [
                log['outcome']
                for log in self.experience_log
                if log['context'].get(context_key) == context_value
            ]
            trend = sum(past_feedback) / len(past_feedback) if past_feedback else 0
            self.context_multipliers[context_id] += learning_rate * (trend - self.context_multipliers[context_id])

    def get_value_priority(self, value_name: str, current_context: dict) -> float:
        """Calculate the current priority of a value based on its weight and context."""
        if value_name not in self.values:
            raise ValueError(f"Value {value_name} not found in system")

        value = self.values[value_name]
        base_priority = value.base_importance * self.value_weights[value_name]

        # Apply dynamic context multipliers
        context_modifier = 1.0
        for context_key, context_value in current_context.items():
            context_id = f"{context_key}:{context_value}"
            context_modifier *= self.context_multipliers[context_id]

        # Consider time frame dynamically based on changing importance
        time_frame_multiplier = {
            TimeFrame.SHORT_TERM: 1.5,
            TimeFrame.MEDIUM_TERM: 1.0,
            TimeFrame.LONG_TERM: 0.7
        }
        dynamic_time_frame_multiplier = time_frame_multiplier[value.time_frame] * self.value_weights[value_name]

        # Consider dependencies and their priorities
        dependency_modifier = 1.0
        for dep in value.dependencies:
            if dep in self.values:
                dependency_modifier += self.get_value_priority(dep, current_context) * 0.1

        return base_priority * context_modifier * dynamic_time_frame_multiplier * dependency_modifier

    def resolve_conflict(self, value_names: List[str], current_context: dict) -> str:
        """Resolve conflict between multiple values based on current priorities."""
        priorities = {
            name: self.get_value_priority(name, current_context)
            for name in value_names
        }
        return max(priorities.items(), key=lambda x: x[1])[0]

    def dynamically_redefine_values(self):
        """Redefine or re-prioritize values based on cumulative feedback."""
        for value_name, value in self.values.items():
            past_feedback = [
                log['outcome']
                for log in self.experience_log
                if log['value'] == value_name
            ]
            if past_feedback:
                avg_feedback = sum(past_feedback) / len(past_feedback)
                value.base_importance = max(0.1, min(1.0, value.base_importance + avg_feedback * 0.05))


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

    # Dynamically redefine values
    system.dynamically_redefine_values()

    # Get individual priorities
    for value_name in system.values:
        priority = system.get_value_priority(value_name, current_context)
        print(f"{value_name}: priority = {priority:.2f}")


if __name__ == "__main__":
    main()
