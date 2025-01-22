from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
import json

"""
Meta-Cognition: Self-Reflection and Learning
    Implement a meta-cognition layer that monitors and evaluates past decisions.

Allow the system to adapt its behavior by:
    1. Updating its value system.
    2. Changing its decision-making rules.
    
This requires the system to analyze its own processes and outcomes, akin to self-reflection.
"""


class MetaCognitionSystem:
    def __init__(self):
        # Initialize core components
        self.value_system = {
            "accuracy": 0.7,
            "efficiency": 0.5,
            "safety": 0.8,
            "innovation": 0.4
        }

        self.decision_rules = {
            "risk_threshold": 0.6,
            "confidence_threshold": 0.7,
            "exploration_rate": 0.2
        }

        # Memory for storing past decisions and outcomes
        self.decision_history: List[Dict] = []
        self.reflection_log: List[Dict] = []
        self.performance_trends: List[float] = []

    def make_decision(self, situation: Dict) -> Tuple[str, float]:
        """
        Make a decision based on current values and rules
        Returns decision and confidence score
        """
        timestamp = datetime.now()

        # Simulate decision-making process using current rules
        confidence = np.random.random() * self.decision_rules["confidence_threshold"]
        should_explore = np.random.random() < self.decision_rules["exploration_rate"]

        # Apply risk threshold
        risk_score = np.random.random()
        decision_safe = risk_score < self.decision_rules["risk_threshold"]

        if should_explore:
            decision = "explore_new_approach"
            confidence *= self.value_system["innovation"]
        elif not decision_safe:
            decision = "conservative_approach"
            confidence *= self.value_system["safety"]
        else:
            decision = "standard_approach"
            confidence *= self.value_system["accuracy"]

        # Record the decision
        self.decision_history.append({
            "timestamp": timestamp,
            "situation": situation,
            "decision": decision,
            "confidence": confidence,
            "values_state": self.value_system.copy(),
            "rules_state": self.decision_rules.copy()
        })

        return decision, confidence

    def record_outcome(self, decision_index: int, outcome: Dict):
        """Record the outcome of a past decision"""
        if 0 <= decision_index < len(self.decision_history):
            self.decision_history[decision_index]["outcome"] = outcome
            self._reflect_on_outcome(decision_index)

    def _reflect_on_outcome(self, decision_index: int):
        """Analyze a decision-outcome pair and update system accordingly"""
        decision_data = self.decision_history[decision_index]
        outcome = decision_data["outcome"]

        # Calculate success metrics
        success_score = outcome.get("success_score", 0)
        efficiency_score = outcome.get("efficiency_score", 0)
        safety_score = outcome.get("safety_score", 0)

        # Update performance trends
        self.performance_trends.append(success_score)
        if len(self.performance_trends) > 50:
            self.performance_trends.pop(0)

        # Generate insights
        insights = {
            "success_vs_confidence": success_score - decision_data["confidence"],
            "value_alignment": self._calculate_value_alignment(outcome),
            "rule_effectiveness": self._evaluate_rule_effectiveness(decision_data)
        }

        # Update value system based on outcomes
        self._update_values(insights)

        # Update decision rules based on performance
        self._update_rules(insights)

        # Log reflection
        self.reflection_log.append({
            "timestamp": datetime.now(),
            "decision_index": decision_index,
            "insights": insights,
            "value_updates": self.value_system.copy(),
            "rule_updates": self.decision_rules.copy()
        })

    def _calculate_value_alignment(self, outcome: Dict) -> float:
        """Calculate how well the outcome aligned with current values"""
        alignment_scores = []

        for key, value in self.value_system.items():
            if key in outcome:
                alignment_scores.append(abs(outcome[key] - value))

        return np.mean(alignment_scores) if alignment_scores else 0.0

    @staticmethod
    def _evaluate_rule_effectiveness(decision_data: Dict) -> float:
        """Evaluate how effective the decision rules were"""
        if "outcome" not in decision_data:
            return 0.0

        outcome = decision_data["outcome"]
        success_score = outcome.get("success_score", 0)

        # Compare outcome to thresholds
        threshold_effectiveness = success_score - decision_data["rules_state"]["confidence_threshold"]

        return threshold_effectiveness

    def _update_values(self, insights: Dict):
        """Update value system based on insights"""
        learning_rate = 0.1

        # Adjust values based on outcome alignment
        alignment_delta = -insights["value_alignment"]  # Lower misalignment improves system
        for value in self.value_system:
            self.value_system[value] += learning_rate * alignment_delta
            # Ensure values stay in valid range
            self.value_system[value] = max(0.1, min(1.0, self.value_system[value]))

    def _update_rules(self, insights: Dict):
        """Update decision rules based on insights"""
        learning_rate = 0.05

        # Adjust confidence threshold based on performance
        confidence_delta = insights["success_vs_confidence"]
        self.decision_rules["confidence_threshold"] += learning_rate * confidence_delta

        # Adjust exploration rate based on rule effectiveness
        exploration_adjustment = insights["rule_effectiveness"]
        self.decision_rules["exploration_rate"] -= learning_rate * exploration_adjustment

        # Ensure rules stay in valid ranges
        for rule in self.decision_rules:
            self.decision_rules[rule] = max(0.1, min(0.9, self.decision_rules[rule]))

    def get_insights(self) -> Dict:
        """Generate insights about system performance and learning"""
        if not self.reflection_log:
            return {"message": "No reflection data available yet"}

        # Analyze trends in value system changes
        value_trends = {}
        for value in self.value_system:
            initial = self.reflection_log[0]["value_updates"].get(value, 0.0)
            current = self.value_system[value]
            value_trends[value] = {
                "initial": initial,
                "current": current,
                "change": current - initial
            }

        # Analyze decision rule evolution
        rule_trends = {}
        for rule in self.decision_rules:
            initial = self.reflection_log[0]["rule_updates"].get(rule, 0.0)
            current = self.decision_rules[rule]
            rule_trends[rule] = {
                "initial": initial,
                "current": current,
                "change": current - initial
            }

        # Analyze performance trends
        avg_performance = np.mean(self.performance_trends) if self.performance_trends else 0.0

        return {
            "value_evolution": value_trends,
            "rule_evolution": rule_trends,
            "average_performance": avg_performance,
            "total_decisions": len(self.decision_history),
            "total_reflections": len(self.reflection_log)
        }


# Example usage
def main():
    # Example usage
    system = MetaCognitionSystem()

    # Simulate some decisions and outcomes
    for i in range(5):
        situation = {
            "complexity": np.random.random(),
            "urgency": np.random.random(),
            "risk_level": np.random.random()
        }

        decision, confidence = system.make_decision(situation)

        # Simulate outcome
        outcome = {
            "success_score": np.random.random(),
            "efficiency_score": np.random.random(),
            "safety_score": np.random.random(),
            "accuracy": np.random.random(),
            "efficiency": np.random.random()
        }

        system.record_outcome(i, outcome)

    # Get insights about system learning
    insights = system.get_insights()
    print(json.dumps(insights, indent=2))

if __name__ == "__main__":
    main()
