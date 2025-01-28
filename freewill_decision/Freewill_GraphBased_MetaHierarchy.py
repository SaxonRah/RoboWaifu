import networkx as nx
import numpy as np
from freewill_environmental import Environment, AdaptiveAgent
from freewill_ethical import EthicalReasoningSystem, Dilemma, Action
from freewill_goals import GoalEvolutionSystem
from freewill_hypothesis import EnhancedHypothesisGenerator, Hypothesis
from freewill_multilayered import ValueSystem, Value, ValueCategory, TimeFrame
from freewill_random import DecisionMaker, DecisionType
from freewill_reinforcement import MetaCognitionSystem


# Class encapsulating the Freewill framework with a graph-based meta-hierarchy
class FreewillGraphSystem:
    def __init__(self):
        # Initialize Freewill components
        self.environment = Environment(size=10)
        self.adaptive_agent = AdaptiveAgent(learning_rate=0.2)
        self.ethical_system = EthicalReasoningSystem()
        self.goal_system = GoalEvolutionSystem()
        self.hypothesis_generator = EnhancedHypothesisGenerator()
        self.value_system = ValueSystem()
        self.decision_maker = DecisionMaker(random_weight=0.5)
        self.meta_cognition = MetaCognitionSystem()

        # Initialize graph-based meta-hierarchy
        self.meta_hierarchy = nx.DiGraph()
        self._initialize_graph()

    def _initialize_graph(self):
        # Add nodes for each Freewill area
        nodes = [
            "Environmental Interaction",
            "Ethical Decision-Making",
            "Goal Evolution",
            "Hypothesis Generation",
            "Multi-Layered Values",
            "Randomness and Determinism",
            "Meta-Cognition"
        ]
        self.meta_hierarchy.add_nodes_from(nodes)

        # Add edges representing interdependencies
        edges = [
            ("Environmental Interaction", "Goal Evolution"),
            ("Goal Evolution", "Ethical Decision-Making"),
            ("Ethical Decision-Making", "Multi-Layered Values"),
            ("Hypothesis Generation", "Goal Evolution"),
            ("Meta-Cognition", "Hypothesis Generation"),
            ("Multi-Layered Values", "Meta-Cognition"),
            ("Randomness and Determinism", "Goal Evolution"),
            ("Goal Evolution", "Meta-Cognition"),
            ("Ethical Decision-Making", "Environmental Interaction")
        ]
        self.meta_hierarchy.add_edges_from(edges)

        # Initialize edge weights dynamically
        for edge in self.meta_hierarchy.edges:
            self.meta_hierarchy.edges[edge]["weight"] = np.random.uniform(0.1, 1.0)

    def update_edge_weights(self, feedback):
        """Dynamically update edge weights based on feedback."""
        for edge in self.meta_hierarchy.edges:
            influence = feedback.get(edge, 0)  # Get feedback signal for the edge
            current_weight = self.meta_hierarchy.edges[edge]["weight"]
            self.meta_hierarchy.edges[edge]["weight"] = max(0.1, min(1.0, current_weight + 0.1 * influence))

    def simulate_iteration(self):
        """Simulate interactions among Freewill areas."""
        feedback = {}

        # Environmental Interaction updates Goal Evolution
        sensor_data = self.environment.get_sensor_data()
        sensor_key = sensor_data.tobytes().hex()  # Convert to a consistent hashable type
        resource_direction = self.adaptive_agent.choose_action(sensor_key)  # Pass as a string
        self.environment.update(resource_direction)
        feedback[("Environmental Interaction", "Goal Evolution")] = 0.1  # Example feedback

        # Goal Evolution updates Ethical Decision-Making
        self.goal_system.update_context({"resource_scarcity": 0.7})
        active_goals = self.goal_system.select_active_goals()
        feedback[("Goal Evolution", "Ethical Decision-Making")] = 0.2

        # Ethical Decision-Making evaluates a dilemma
        dilemma = self.ethical_system.resolve_dilemma(
            Dilemma(
                description="Resource collection vs environmental harm",
                stakeholders=["system", "environment"],
                possible_actions=[
                    Action(
                        name="collect_resource",
                        consequences={"system": 1.0, "environment": -0.5},
                        intention="resource_collection",
                        duty_alignments={"survival": 1.0}
                    ),
                    Action(
                        name="avoid_harm",
                        consequences={"system": -0.5, "environment": 1.0},
                        intention="avoid_damage",
                        duty_alignments={"sustainability": 1.0}
                    )
                ],
                context={"emergency": False}
            )
        )
        feedback[("Ethical Decision-Making", "Multi-Layered Values")] = 0.3

        # Multi-Layered Values influence Meta-Cognition
        self.value_system.log_experience("sustainability", 0.8, {"context": "ethical_dilemma"})
        feedback[("Multi-Layered Values", "Meta-Cognition")] = 0.1

        # Meta-Cognition refines Hypothesis Generation
        hypothesis = self.hypothesis_generator.generate_hypothesis(sensor_key)  # Pass as a string
        feedback[("Meta-Cognition", "Hypothesis Generation")] = 0.2

        # Hypothesis Generation feeds back to Goal Evolution
        self.hypothesis_generator.observe_pattern([sensor_key], 0.6)  # Pass as a string
        feedback[("Hypothesis Generation", "Goal Evolution")] = 0.4

        # Update edge weights based on feedback
        self.update_edge_weights(feedback)

        return feedback

    def run_simulation(self, iterations=10):
        """Run the simulation for a specified number of iterations."""
        for _ in range(iterations):
            feedback = self.simulate_iteration()
            print(f"Feedback after iteration: {feedback}")

    def visualize_graph(self):
        """Visualize the meta-hierarchy graph."""
        import matplotlib.pyplot as plt

        pos = nx.spring_layout(self.meta_hierarchy)
        edge_labels = nx.get_edge_attributes(self.meta_hierarchy, 'weight')
        nx.draw(self.meta_hierarchy, pos, with_labels=True, node_color='lightblue', node_size=2000)
        nx.draw_networkx_edge_labels(self.meta_hierarchy, pos,
                                     edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
        plt.title("Meta-Hierarchy of Freewill System")
        plt.show()


# Initialize and run the system
if __name__ == "__main__":
    freewill_system = FreewillGraphSystem()
    freewill_system.run_simulation(iterations=10)
    freewill_system.visualize_graph()
