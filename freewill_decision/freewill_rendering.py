import pygame
from typing import Optional
from freewill import IntegratedFreewillSystem


class SystemVisualizer:
    def __init__(self, window_size: int = 1024):
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Freewill System Visualization")

        self.COLORS = {
            'background': (30, 30, 30),
            'grid': (50, 50, 50),
            'agent': (0, 255, 0),
            'resource': (0, 191, 255),
            'threat': (255, 0, 0),
            'text': (255, 255, 255),
            'highlight': (255, 255, 0),
            'value_bar': (100, 149, 237),
            'goal_bar': (255, 140, 0),
            'decision_bar': (0, 128, 128),
            'hypothesis_bar': (255, 105, 180)
        }

        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)

    def draw_environment(self, env, cell_size: Optional[int] = None):
        if cell_size is None:
            cell_size = self.window_size // (env.size + 2)

        # Grid
        for i in range(env.size):
            for j in range(env.size):
                rect = pygame.Rect(
                    i * cell_size + cell_size,
                    j * cell_size + cell_size,
                    cell_size - 1,
                    cell_size - 1
                )
                pygame.draw.rect(self.screen, self.COLORS['grid'], rect, 1)

        # Resources
        for res_pos in env.resources:
            x, y = res_pos
            pygame.draw.circle(
                self.screen,
                self.COLORS['resource'],
                (int((x + 1.5) * cell_size), int((y + 1.5) * cell_size)),
                cell_size // 3
            )

        # Threats
        for threat_pos in env.threats:
            x, y = threat_pos
            points = [
                ((x + 1) * cell_size + cell_size // 2, (y + 1) * cell_size),
                ((x + 2) * cell_size, (y + 2) * cell_size),
                ((x + 1) * cell_size, (y + 2) * cell_size)
            ]
            pygame.draw.polygon(self.screen, self.COLORS['threat'], points)

        # Agent
        x, y = env.agent_pos
        pygame.draw.circle(
            self.screen,
            self.COLORS['agent'],
            (int((x + 1.5) * cell_size), int((y + 1.5) * cell_size)),
            cell_size // 2
        )

    def draw_value_system(self, system, x: int, y: int, width: int = 200):
        title = self.font.render("Value System", True, self.COLORS['text'])
        self.screen.blit(title, (x, y))

        y += 30
        bar_height = 20

        for value in system.value_system.values.values():
            text = self.small_font.render(value.name, True, self.COLORS['text'])
            self.screen.blit(text, (x, y))

            bar_width = int(value.base_importance * width)
            bar_rect = pygame.Rect(x + 100, y, bar_width, bar_height)
            pygame.draw.rect(self.screen, self.COLORS['value_bar'], bar_rect)
            pygame.draw.rect(self.screen, self.COLORS['text'], bar_rect, 1)

            y += bar_height + 5

    def draw_goals(self, system, x: int, y: int, width: int = 200):
        title = self.font.render("Active Goals", True, self.COLORS['text'])
        self.screen.blit(title, (x, y))

        y += 30
        bar_height = 20
        active_goals = system.goal_system.select_active_goals(max_active=3)

        for goal in active_goals:
            text = self.small_font.render(goal.name, True, self.COLORS['text'])
            self.screen.blit(text, (x, y))

            bar_width = int(goal.priority * width)
            bar_rect = pygame.Rect(x + 100, y, bar_width, bar_height)
            pygame.draw.rect(self.screen, self.COLORS['goal_bar'], bar_rect)
            pygame.draw.rect(self.screen, self.COLORS['text'], bar_rect, 1)

            y += bar_height + 5

    def draw_ethical_state(self, system, x: int, y: int):
        title = self.font.render("Ethical Framework Weights", True, self.COLORS['text'])
        self.screen.blit(title, (x, y))

        y += 30
        for framework in system.ethical_system.frameworks:
            weight = system.ethical_system.framework_weights[framework.name]
            text = self.small_font.render(
                f"{framework.name}: {weight:.2f}",
                True,
                self.COLORS['text']
            )
            self.screen.blit(text, (x, y))
            y += 25

    def draw_hypotheses(self, system, x: int, y: int):
        title = self.font.render("Hypotheses", True, self.COLORS['text'])
        self.screen.blit(title, (x, y))

        y += 30
        hypotheses = system.hypothesis_results[-5:]  # Show the last 5 hypotheses

        for hypothesis in hypotheses:
            text = self.small_font.render(
                f"{hypothesis['hypothesis'].type}: "
                f"Conf {hypothesis['hypothesis'].confidence:.2f}, "
                f"Pred {hypothesis['hypothesis'].prediction:.2f}",
                True,
                self.COLORS['text']
            )
            self.screen.blit(text, (x, y))
            y += 20

    def draw_decision_info(self, system, x: int, y: int):
        title = self.font.render("Decision Info", True, self.COLORS['text'])
        self.screen.blit(title, (x, y))

        y += 30
        # decision_info = system.decide_action()[1]  # Existing decision info

        # Add recent decisions and trends
        recent_decisions = system.meta_cognition.decision_history[-5:]  # Last 5 decisions
        for decision in recent_decisions:
            text = self.small_font.render(f"Dec: {decision['decision']} | Conf: {decision['confidence']:.2f}", True,
                                          self.COLORS['text'])
            self.screen.blit(text, (x, y))
            y += 20

        # Additional trends (if available)
        if hasattr(system.meta_cognition, "performance_trends"):
            trend = system.meta_cognition.performance_trends[-5:]  # Last 5 performance metrics
            trend_text = f"Trend: {', '.join(f'{t:.2f}' for t in trend)}"
            trend_surface = self.small_font.render(trend_text, True, self.COLORS['text'])
            self.screen.blit(trend_surface, (x, y))
            y += 20

    def draw_meta_cognition(self, system, x: int, y: int):
        title = self.font.render("Meta-Cognition Insights", True, self.COLORS['text'])
        self.screen.blit(title, (x, y))

        y += 30
        insights = system.meta_cognition.get_insights()
        if "average_performance" in insights:
            text = self.small_font.render(
                f"Avg Performance: {insights['average_performance']:.2f}",
                True,
                self.COLORS['text']
            )
            self.screen.blit(text, (x, y))

        y += 20
        if "value_evolution" in insights:
            for value, data in insights["value_evolution"].items():
                text = self.small_font.render(
                    f"{value}: {data['current']:.2f} (Change: {data['change']:.2f})",
                    True,
                    self.COLORS['text']
                )
                self.screen.blit(text, (x, y))
                y += 20

    def render(self, system: IntegratedFreewillSystem) -> None:
        self.screen.fill(self.COLORS['background'])
        self.draw_environment(system.environment)
        self.draw_value_system(system, 10, 300)
        self.draw_goals(system, 400, 300)
        self.draw_ethical_state(system, 10, 500)
        self.draw_hypotheses(system, 400, 500)
        self.draw_decision_info(system, 10, 700)
        self.draw_meta_cognition(system, 400, 700)
        pygame.display.flip()

    @staticmethod
    def handle_events() -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True


def main():
    freewill_system = IntegratedFreewillSystem(env_size=100)
    visualizer = SystemVisualizer()

    running = True
    clock = pygame.time.Clock()

    while running:
        step_info = freewill_system.step()
        # print(step_info)
        visualizer.render(freewill_system)
        running = visualizer.handle_events()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()
