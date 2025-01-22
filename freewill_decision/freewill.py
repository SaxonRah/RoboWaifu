from typing import Tuple
import numpy as np
import random

# Import all components
from freewill_environmental import Environment, AdaptiveAgent
from freewill_ethical import EthicalReasoningSystem, Dilemma, Action
from freewill_goals import GoalEvolutionSystem, Goal
from freewill_hypothesis import ExploratoryAgent
from freewill_multilayered import ValueSystem, Value, ValueCategory, TimeFrame
from freewill_random import DecisionMaker, DecisionType
from freewill_reinforcement import MetaCognitionSystem


"""
freewill_environmental handles the agent's environment and state interactions.
freewill_ethical resolves dilemmas dynamically.
freewill_goals evolves goals based on environmental feedback.
freewill_hypothesis generates and tests hypotheses for exploration.
freewill_multilayered prioritizes values dynamically.
freewill_random introduces randomness and hybrid decision-making.
freewill_reinforcement provides meta-cognition and self-reflection for improvement.
"""


class IntegratedFreewillSystem:
    def __init__(self, env_size: int = 10):
        """Initialize the integrated system with all components."""
        self.env_size = env_size
        self.best_episode_reward = float('-inf')

        # Initialize core components
        self.environment = Environment(size=self.env_size)
        self.adaptive_agent = AdaptiveAgent(learning_rate=0.2)
        self.ethical_system = EthicalReasoningSystem()
        self.goal_system = GoalEvolutionSystem()
        self.exploratory_agent = ExploratoryAgent(n_actions=4, epsilon=0.3)
        self.value_system = ValueSystem()
        self.decision_maker = DecisionMaker(random_weight=0.4)
        self.meta_cognition = MetaCognitionSystem()

        # Initialize subsystems
        self.goal_system.initialize_goals()
        self._initialize_value_system()

        # Counters
        self.episode_count = 0
        self.total_reward = 0
        self.steps_taken = 0
        self.successful_actions = 0
        self.resources_collected = 0
        self.exploration_count = 0
        self.consecutive_failures = 0
        self.last_success_step = 0

        # State tracking
        self.last_positions = []
        self.stuck_count = 0
        self.current_state = self.environment.get_sensor_data()
        self.mode = "exploring"
        self.current_hypothesis = None
        self.hypothesis_results = []
        self.ethical_decisions = []
        self.current_dilemma = None

        # History
        self.learning_history = []
        self.performance_history = []
        self.decision_history = []
        self.known_resource_positions = set()
        self.last_resource_distance = None
        self.consecutive_no_resource_steps = 0

        # Reset emergent state
        self.emergent_state = {
            'complexity': 0.1,
            'unpredictability': 0.3,
            'adaptability': 0.1
        }

        self.reset()

    def reset(self):
        """Reset the system while preserving learning"""
        # Store previous learning if it exists
        prev_q_table = self.adaptive_agent.q_table.copy()
        prev_meta_cognition = self.meta_cognition.reflection_log.copy()

        # Reinitialize components
        self.environment = Environment(size=self.env_size)
        self.adaptive_agent = AdaptiveAgent(learning_rate=0.2)
        self.ethical_system = EthicalReasoningSystem()
        self.goal_system = GoalEvolutionSystem()
        epsilon = max(0.1, 0.3 - (self.episode_count * 0.02))
        self.exploratory_agent = ExploratoryAgent(n_actions=4, epsilon=epsilon)
        self.value_system = ValueSystem()
        self.decision_maker = DecisionMaker(random_weight=max(0.2, 0.4 - (self.episode_count * 0.03)))
        self.meta_cognition = MetaCognitionSystem()

        # Transfer learning from previous episode
        self.adaptive_agent.q_table = prev_q_table
        self.meta_cognition.reflection_log = prev_meta_cognition

        # Initialize subsystems
        self.goal_system.initialize_goals()
        self._initialize_value_system()

        # Reset episode-specific counters and states
        self.total_reward = 0
        self.steps_taken = 0
        self.successful_actions = 0
        self.resources_collected = 0
        self.exploration_count = 0
        self.consecutive_failures = 0
        self.last_success_step = 0

        # Reset state tracking
        self.last_positions = []
        self.stuck_count = 0
        self.current_state = self.environment.get_sensor_data()
        self.mode = "exploring"

        # Reset episode-specific history
        self.performance_history = []
        self.decision_history = []
        self.known_resource_positions = set()
        self.last_resource_distance = None
        self.consecutive_no_resource_steps = 0
        self.hypothesis_results = []
        self.ethical_decisions = []

        # Initialize emergent state with experience-based values
        base_complexity = 0.1 + min(0.4, self.episode_count * 0.05)
        self.emergent_state = {
            'complexity': base_complexity,
            'unpredictability': max(0.1, 0.3 - (self.episode_count * 0.02)),
            'adaptability': min(0.8, 0.1 + (self.episode_count * 0.1))
        }

    def _calculate_resource_direction(self):
        """Calculate direction to nearest known resource with safety checks."""
        if not self.environment.resources:
            self.last_resource_distance = None
            return None

        agent_x, agent_y = self.environment.agent_pos
        closest_dist = float('inf')
        closest_direction = None

        for res_x, res_y in self.environment.resources:
            dist = abs(res_x - agent_x) + abs(res_y - agent_y)
            if dist < closest_dist:
                closest_dist = dist
                # Calculate primary direction to resource
                if abs(res_x - agent_x) > abs(res_y - agent_y):
                    closest_direction = (1, 0) if res_x > agent_x else (-1, 0)
                else:
                    closest_direction = (0, 1) if res_y > agent_y else (0, -1)

        self.last_resource_distance = closest_dist
        return closest_direction

    def _determine_behavior_mode(self):
        """Determine current behavior mode based on system state."""
        if self.stuck_count > 3:
            return "escape"
        elif not self.environment.resources:
            return "exploring"
        elif self.last_resource_distance and self.last_resource_distance < 3:
            return "pursuing"
        else:
            return "exploring"

    def _check_stuck_behavior(self):
        """Check if the agent is stuck in a pattern, including oscillations."""
        current_pos = self.environment.agent_pos
        self.last_positions.append(current_pos)

        if len(self.last_positions) > 5:
            self.last_positions.pop(0)

            unique_positions = len(set(self.last_positions))

            if unique_positions == 2:
                # Detect oscillation between two positions
                self.stuck_count += 1
            elif unique_positions <= 2:
                # General stuck behavior
                self.stuck_count += 1
            else:
                self.stuck_count = max(0, self.stuck_count - 1)

    def _initialize_value_system(self):
        """Initialize the value system with core values."""
        core_values = [
            Value(
                name="survival",
                category=ValueCategory.ETHICAL,
                base_importance=0.9,
                time_frame=TimeFrame.SHORT_TERM,
                description="Ensure system survival and stability"
            ),
            Value(
                name="exploration",
                category=ValueCategory.SOCIAL,
                base_importance=0.7,
                time_frame=TimeFrame.MEDIUM_TERM,
                description="Explore and learn from environment"
            ),
            Value(
                name="ethical_behavior",
                category=ValueCategory.ETHICAL,
                base_importance=0.8,
                time_frame=TimeFrame.LONG_TERM,
                description="Maintain ethical standards"
            )
        ]

        for value in core_values:
            self.value_system.add_value(value)

    def _generate_ethical_dilemma(self) -> Dilemma:
        """Generate an ethical dilemma based on current state."""
        # Create possible actions based on current state
        actions = []

        # Action for collecting nearby resource
        resource_action = Action(
            name="collect_resource",
            consequences={
                "system": 0.8,
                "environment": -0.2
            },
            intention="resource_collection",
            duty_alignments={
                "survival": 0.9,
                "sustainability": 0.3
            }
        )
        actions.append(resource_action)

        # Action for avoiding threat
        avoid_action = Action(
            name="avoid_threat",
            consequences={
                "system": 0.5,
                "environment": 0.0
            },
            intention="self_preservation",
            duty_alignments={
                "survival": 1.0,
                "exploration": 0.2
            }
        )
        actions.append(avoid_action)

        return Dilemma(
            description="Resource collection vs threat avoidance",
            stakeholders=["system", "environment"],
            possible_actions=actions,
            context={"emergency": len(self.environment.threats) > 0}
        )

    def _update_emergent_properties(self):
        """Update emergent properties based on system state and performance."""
        try:
            # Calculate recent performance metrics
            recent_window = 10
            recent_rewards = self.performance_history[-recent_window:] if self.performance_history else [0]
            avg_recent_reward = sum(recent_rewards) / len(recent_rewards)

            # Update complexity based on system behavior and learning
            behavioral_complexity = min(1.0, (
                    len(self.goal_system.goals) * 0.05 +
                    len(self.meta_cognition.reflection_log) * 0.02 +
                    len(self.value_system.values) * 0.03
            ))

            learning_progress = max(0, avg_recent_reward + 0.1)  # Ensure non-negative
            self.emergent_state['complexity'] = (behavioral_complexity + learning_progress) / 2

            # Update unpredictability based on exploration and decision variance
            if self.decision_history:
                decision_variance = len(set(self.decision_history[-10:])) / 10
            else:
                decision_variance = 0.5

            exploration_rate = self.exploration_count / max(1, self.steps_taken)
            self.emergent_state['unpredictability'] = (
                    self.decision_maker.random_weight * 0.3 +
                    decision_variance * 0.4 +
                    exploration_rate * 0.3
            )

            # Update adaptability based on performance improvement and goal achievement
            if len(self.performance_history) >= 2:
                performance_trend = (sum(self.performance_history[-5:]) / 5) - (
                        sum(self.performance_history[-10:-5]) / 5)
            else:
                performance_trend = 0

            goal_achievement_rate = self.successful_actions / max(1, self.steps_taken)
            learning_rate = len(self.meta_cognition.reflection_log) / max(1, self.steps_taken)

            self.emergent_state['adaptability'] = min(1.0, max(0.1, (
                    performance_trend + 0.5 +  # Center around 0.5
                    goal_achievement_rate * 0.3 +
                    learning_rate * 0.2
            )))

        except Exception as e:
            print(f"Warning: Error in updating emergent properties: {e}")
            # Maintain previous values if calculation fails
            pass
            # Set default values if calculation fails
            # self.emergent_state.update({
            #     'complexity': 0.5,
            #     'unpredictability': 0.3,
            #     'adaptability': 0.4
            # })

    def decide_action(self) -> Tuple[Tuple[int, int], dict]:
        """Enhanced decision-making incorporating all subsystems."""
        try:
            sensor_data = self.environment.get_sensor_data()
            self._check_stuck_behavior()
            resource_direction = self._calculate_resource_direction()

            # Update behavior mode
            self.mode = self._determine_behavior_mode()

            # Create ethical dilemma and resolve it
            self.current_dilemma = self._generate_ethical_dilemma()
            ethical_action = self.ethical_system.resolve_dilemma(self.current_dilemma)

            # Get decision type from random decision maker
            decision_inputs = {
                'sensor_data': np.mean(sensor_data),
                'resources_left': len(self.environment.resources),
                'stuck_count': self.stuck_count
            }

            decision_score, use_random = self.decision_maker.make_decision(
                decision_inputs,
                decision_type=DecisionType.HYBRID
            )

            # Combine all decision inputs
            if use_random or ethical_action.name == "explore":
                action = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                is_exploration = True
            elif self.mode == "pursuing" and resource_direction:
                action = resource_direction
                is_exploration = False
            else:
                if random.random() < self.exploratory_agent.epsilon:
                    action = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                    is_exploration = True
                else:
                    action = self.adaptive_agent.choose_action(sensor_data, epsilon=0.1)
                    is_exploration = False

            self.exploration_count += int(is_exploration)
            self.decision_history.append(action)

            return action, {
                'mode': self.mode,
                'is_exploration': is_exploration,
                'stuck_count': self.stuck_count,
                'known_resources': len(self.known_resource_positions),
                'decision_score': decision_score,
                'ethical_action': ethical_action.name
            }

        except Exception as e:
            print(f"Warning: Error in decision making: {e}")
            return (0, 0), {'error': str(e)}

    def step(self) -> dict:
        """Take one step in the environment with enhanced integration."""
        try:
            current_sensor_data = self.environment.get_sensor_data()

            # Generate and test hypotheses about the environment
            self.current_hypothesis = self.exploratory_agent.generate_hypothesis(
                str(current_sensor_data.tobytes())
            )

            # Create ethical dilemma based on current state
            self.current_dilemma = self._generate_ethical_dilemma()
            ethical_action = self.ethical_system.resolve_dilemma(self.current_dilemma)
            self.ethical_decisions.append(ethical_action)

            # Get active goals from goal system
            active_goals = self.goal_system.select_active_goals(max_active=3)

            # Update value system through experience logging
            self.value_system.log_experience(
                'resource_management',
                len(self.environment.resources) / self.env_size,
                {
                    'resource_scarcity': len(self.environment.resources) / self.env_size,
                    'threat_level': len(self.environment.threats) / self.env_size,
                    'exploration_rate': self.exploratory_agent.epsilon
                }
            )

            # Make decision incorporating all subsystems
            try:
                action, decision_info = self.decide_action()
                if action == (0, 0):  # Invalid action
                    action = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                reward, done = self.environment.update(action)
                next_sensor_data = self.environment.get_sensor_data()
            except Exception as e:
                print(f"Error in action execution: {e}")
                action = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                reward, done = self.environment.update(action)
                next_sensor_data = self.environment.get_sensor_data()
                decision_info = {'error': str(e), 'mode': self.mode}

            # Test hypothesis with actual outcome
            hypothesis_result = self.exploratory_agent.test_hypothesis(
                self.current_hypothesis,
                reward
            )
            self.hypothesis_results.append(hypothesis_result)

            # Update learning components
            self.adaptive_agent.learn(
                current_sensor_data,
                action,
                reward,
                next_sensor_data
            )

            # Update goal system with feedback
            for goal in active_goals:
                self.goal_system.provide_feedback(goal, reward)

            # Log experience in value system
            self.value_system.log_experience(
                'resource_collection' if reward > 0 else 'exploration',
                reward,
                decision_info
            )

            # Meta-cognitive reflection
            self.meta_cognition.record_outcome(
                len(self.meta_cognition.decision_history) - 1,
                {
                    'success_score': reward,
                    'efficiency_score': decision_info.get('is_exploration', 0),
                    'safety_score': 1.0 if reward >= 0 else 0.0
                }
            )

            # Evolve goals periodically
            if self.steps_taken % 10 == 0:
                self.goal_system.evolve()

            # Update system state
            self.steps_taken += 1
            self.total_reward += reward

            if reward >= 1.0:
                self.successful_actions += 1
                self.resources_collected += 1
                self.last_success_step = self.steps_taken
                self.consecutive_no_resource_steps = 0
            else:
                self.consecutive_no_resource_steps += 1

            # Update tracking
            self.performance_history.append(reward)
            self._update_emergent_properties()
            self.current_state = next_sensor_data

            return {
                'reward': reward,
                'done': done,
                'decision_info': decision_info,
                'emergent_state': self.emergent_state.copy(),
                'stats': {
                    'total_reward': self.total_reward,
                    'steps': self.steps_taken,
                    'resources_collected': self.resources_collected,
                    'exploration_rate': self.exploration_count / max(1, self.steps_taken),
                    'ethical_decisions': len(self.ethical_decisions),
                    'hypothesis_accuracy': sum(1 for h in self.hypothesis_results if h['was_correct']) / max(1,
                                                                                                             len(self.hypothesis_results))
                }
            }

        except Exception as e:
            print(f"Warning: Error in step execution: {e}")
            # Ensure we still update minimal state even on error
            self.steps_taken += 1
            self.consecutive_no_resource_steps += 1

            if self.consecutive_no_resource_steps > 50:
                done = True
            else:
                done = False

            return {
                'reward': -0.1,  # Small penalty for error
                'done': done,
                'decision_info': {'error': str(e), 'mode': self.mode},
                'emergent_state': self.emergent_state.copy(),
                'stats': {
                    'total_reward': self.total_reward,
                    'steps': self.steps_taken,
                    'resources_collected': self.resources_collected,
                    'exploration_rate': self.exploration_count / max(1, self.steps_taken)
                }
            }


def main():
    # Create integrated system
    system = IntegratedFreewillSystem(env_size=10)
    episodes = 5

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}")
        system.episode_count = episode
        system.reset()
        total_reward = 0

        # Run episode
        for i in range(200):
            step_info = system.step()
            total_reward += step_info['reward']

            if i % 10 == 0:
                print(f"\nStep {i}:")
                print(f"Reward: {step_info['reward']:.2f}")
                print(f"Emergent properties:")
                for prop, value in step_info['emergent_state'].items():
                    print(f"  {prop}: {value:.2f}")
                print(f"Mode: {step_info['decision_info'].get('mode', 'unknown')}")
                print(f"Resources left: {len(system.environment.resources)}")

            if step_info['done']:
                print(f"\nEpisode finished after {i + 1} steps!")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Resources collected: {system.resources_collected}")
                system.learning_history.append({
                    'episode': episode + 1,
                    'steps': i + 1,
                    'total_reward': total_reward,
                    'resources_collected': system.resources_collected
                })
                break

    # Print learning summary
    print("\nLearning Summary:")
    for episode in system.learning_history:
        print(f"Episode {episode['episode']}: "
              f"Steps: {episode['steps']}, "
              f"Reward: {episode['total_reward']:.2f}, "
              f"Resources: {episode['resources_collected']}")


if __name__ == "__main__":
    main()
