import re
import json
from dataclasses import dataclass
from typing import List, Optional
import datetime


@dataclass
class MotorConfig:
    poles: int
    coils: int
    turns: int
    torque: float
    resistance: float
    efficiency: float
    dimensions: tuple  # (diameter, thickness)


class MotorAnalyzer:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.configs = []

    def parse_output(self, text: str) -> List[MotorConfig]:
        """Parse the motor calculator output text and extract configurations."""
        # Regular expression to match configuration blocks
        config_pattern = r"Configuration \d+:\nPoles: (\d+)\nCoils: (\d+)\nTurns per coil: (\d+)\nEstimated torque: ([\d.]+) Nm\nTotal resistance: ([\d.]+) Ω\nEfficiency: ([\d.]+)%\nDimensions: ([\d.]+)mm diameter, ([\d.]+)mm thick"

        matches = re.finditer(config_pattern, text)
        configs = []

        for match in matches:
            config = MotorConfig(
                poles=int(match.group(1)),
                coils=int(match.group(2)),
                turns=int(match.group(3)),
                torque=float(match.group(4)),
                resistance=float(match.group(5)),
                efficiency=float(match.group(6)),
                dimensions=(float(match.group(7)), float(match.group(8)))
            )
            configs.append(config)

        self.configs = configs
        return configs

    def generate_html(self) -> str:
        """Generate HTML report with interactive visualizations."""
        # Convert configs to JSON for JavaScript
        config_data = []
        for config in self.configs:
            config_data.append({
                'config': f"{config.poles}P/{config.coils}C",
                'poles': config.poles,
                'coils': config.coils,
                'turns': config.turns,
                'torque': round(config.torque, 3),
                'resistance': round(config.resistance, 2),
                'efficiency': round(config.efficiency, 1)
            })

        # Find best configurations in different categories
        best_efficiency = max(self.configs, key=lambda x: x.efficiency)
        best_torque = max(self.configs, key=lambda x: x.torque)
        simplest = min(self.configs, key=lambda x: x.poles + x.coils)

        html = f"""
<!DOCTYPE html>
<html lang="en">
    <meta charset="UTF-8">
<head>
    <title>Motor Configuration Analysis</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.0/chart.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .chart-container {{
            position: relative;
            height: 300px;
            width: 100%;
            margin-bottom: 2rem;
        }}
    </style>
</head>
<body class="bg-gray-100 p-8">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-3xl font-bold mb-6">Motor Configuration Analysis</h1>
        <p class="text-gray-600 mb-4">Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <!-- Best Efficiency Card -->
            <div class="bg-white p-4 rounded-lg shadow">
                <h3 class="text-lg font-semibold mb-2">Best Efficiency</h3>
                <p class="text-2xl font-bold text-green-600 mb-2">{best_efficiency.efficiency:.1f}%</p>
                <p class="text-gray-600">
                    {best_efficiency.poles}P/{best_efficiency.coils}C<br>
                    {best_efficiency.turns} turns<br>
                    {best_efficiency.torque:.3f} Nm
                </p>
            </div>

            <!-- Best Torque Card -->
            <div class="bg-white p-4 rounded-lg shadow">
                <h3 class="text-lg font-semibold mb-2">Highest Torque</h3>
                <p class="text-2xl font-bold text-blue-600 mb-2">{best_torque.torque:.3f} Nm</p>
                <p class="text-gray-600">
                    {best_torque.poles}P/{best_torque.coils}C<br>
                    {best_torque.turns} turns<br>
                    {best_torque.efficiency:.1f}% efficiency
                </p>
            </div>

            <!-- Simplest Design Card -->
            <div class="bg-white p-4 rounded-lg shadow">
                <h3 class="text-lg font-semibold mb-2">Simplest Design</h3>
                <p class="text-2xl font-bold text-purple-600 mb-2">{simplest.poles}P/{simplest.coils}C</p>
                <p class="text-gray-600">
                    {simplest.turns} turns<br>
                    {simplest.efficiency:.1f}% efficiency<br>
                    {simplest.torque:.3f} Nm
                </p>
            </div>
        </div>

        <!-- Charts -->
        <div class="bg-white p-6 rounded-lg shadow mb-8">
            <h2 class="text-xl font-bold mb-4">Efficiency Comparison</h2>
            <div class="chart-container">
                <canvas id="efficiencyChart"></canvas>
            </div>
        </div>

        <div class="bg-white p-6 rounded-lg shadow mb-8">
            <h2 class="text-xl font-bold mb-4">Torque vs Resistance</h2>
            <div class="chart-container">
                <canvas id="torqueResistanceChart"></canvas>
            </div>
        </div>

        <!-- Data Table -->
        <div class="bg-white p-6 rounded-lg shadow">
            <h2 class="text-xl font-bold mb-4">All Configurations</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full table-auto">
                    <thead>
                        <tr class="bg-gray-100">
                            <th class="px-4 py-2">Configuration</th>
                            <th class="px-4 py-2">Turns</th>
                            <th class="px-4 py-2">Torque (Nm)</th>
                            <th class="px-4 py-2">Efficiency (%)</th>
                            <th class="px-4 py-2">Resistance (&Omega;)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([
            f'<tr class="border-t">'
            f'<td class="px-4 py-2">{c.poles}P/{c.coils}C</td>'
            f'<td class="px-4 py-2">{c.turns}</td>'
            f'<td class="px-4 py-2">{c.torque:.3f}</td>'
            f'<td class="px-4 py-2">{c.efficiency:.1f}</td>'
            f'<td class="px-4 py-2">{c.resistance:.2f}</td>'
            f'</tr>'
            for c in self.configs
        ])}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Chart configuration and data
        const configData = {json.dumps(config_data)};

        // Helper function to create charts
        function createChart(elementId, datasets, maxY = null) {{
            const config = {{
                type: 'bar',
                data: {{
                    labels: configData.map(c => c.config),
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ...(maxY ? {{max: maxY}} : {{}})
                        }}
                    }}
                }}
            }};
            new Chart(document.getElementById(elementId), config);
        }}

        // Create Efficiency Chart
        createChart('efficiencyChart', [
            {{
                label: 'Efficiency (%)',
                data: configData.map(c => c.efficiency),
                backgroundColor: '#4CAF50'
            }}
        ], 100);

        // Create Torque vs Resistance Chart
        createChart('torqueResistanceChart', [
            {{
                label: 'Torque (Nm)',
                data: configData.map(c => c.torque),
                backgroundColor: '#2196F3'
            }},
            {{
                label: 'Resistance (Ω)',
                data: configData.map(c => c.resistance),
                backgroundColor: '#FF9800'
            }}
        ]);
    </script>
</body>
</html>
        """

        return html

    def save_report(self, reported_output_text: str):
        """Parse the output and save the HTML report."""
        self.parse_output(reported_output_text)
        html = self.generate_html()

        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"Report saved to {self.output_file}")


if __name__ == "__main__":
    with open('motor_output.txt', 'r') as f:
        motor_output_text = f.read()

    analyzer = MotorAnalyzer('motor_analysis.html')
    analyzer.save_report(motor_output_text)
