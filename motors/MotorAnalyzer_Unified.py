import re
import json
import datetime
from typing import List, Dict, Any
from UnifiedMotorParameters import UnifiedMotorParameters


class MotorAnalyzer:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.configs: List[UnifiedMotorParameters] = []

    @staticmethod
    def _generate_head() -> str:
        """Generate HTML head section with styles and scripts."""
        return """
<head>
    <meta charset="UTF-8">
    <title>Motor Configuration Analysis</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.0/chart.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .chart-container {
            position: relative;
            height: 300px;
            width: 100%;
            margin-bottom: 2rem;
        }
        .highlight {
            background-color: rgba(255, 215, 0, 0.1);
        }
        .stat-label {
            color: #6B7280;
            font-size: 0.875rem;
        }
        .stat-value {
            font-size: 1.25rem;
            font-weight: 600;
        }
    </style>
</head>"""

    @staticmethod
    def _generate_header() -> str:
        """Generate page header with title and timestamp."""
        return f"""
<div class="max-w-6xl mx-auto">
    <h1 class="text-3xl font-bold mb-6">Motor Configuration Analysis</h1>
    <p class="text-gray-600 mb-4">Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>"""

    @staticmethod
    def _generate_stat_cards(best_efficiency: UnifiedMotorParameters,
                             best_torque: UnifiedMotorParameters,
                             simplest: UnifiedMotorParameters,
                             most_compact: UnifiedMotorParameters) -> str:
        """Generate the statistics cards section."""
        return f"""
<div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
    <div class="bg-white p-4 rounded-lg shadow">
        <h3 class="text-lg font-semibold mb-2">Best Efficiency</h3>
        <p class="text-2xl font-bold text-green-600 mb-2">{best_efficiency.efficiency:.1f}%</p>
        <p class="text-gray-600">
            {best_efficiency.poles}P/{best_efficiency.coils}C<br>
            {best_efficiency.turns_per_coil} turns<br>
            {best_efficiency.estimated_torque:.3f} Nm
        </p>
    </div>

    <div class="bg-white p-4 rounded-lg shadow">
        <h3 class="text-lg font-semibold mb-2">Highest Torque</h3>
        <p class="text-2xl font-bold text-blue-600 mb-2">{best_torque.estimated_torque:.3f} Nm</p>
        <p class="text-gray-600">
            {best_torque.poles}P/{best_torque.coils}C<br>
            {best_torque.turns_per_coil} turns<br>
            {best_torque.efficiency:.1f}% efficiency
        </p>
    </div>

    <div class="bg-white p-4 rounded-lg shadow">
        <h3 class="text-lg font-semibold mb-2">Simplest Design</h3>
        <p class="text-2xl font-bold text-purple-600 mb-2">{simplest.poles}P/{simplest.coils}C</p>
        <p class="text-gray-600">
            {simplest.turns_per_coil} turns<br>
            {simplest.efficiency:.1f}% efficiency<br>
            {simplest.estimated_torque:.3f} Nm
        </p>
    </div>

    <div class="bg-white p-4 rounded-lg shadow">
        <h3 class="text-lg font-semibold mb-2">Most Compact</h3>
        <p class="text-2xl font-bold text-yellow-600 mb-2">{most_compact.outer_radius * 2:.1f}mm</p>
        <p class="text-gray-600">
            {most_compact.poles}P/{most_compact.coils}C<br>
            {most_compact.efficiency:.1f}% efficiency<br>
            {most_compact.estimated_torque:.3f} Nm
        </p>
    </div>
</div>"""

    @staticmethod
    def _generate_chart_containers() -> str:
        """Generate the chart container sections."""
        return """
<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
    <div class="bg-white p-6 rounded-lg shadow">
        <h2 class="text-xl font-bold mb-4">Efficiency vs Torque</h2>
        <div class="chart-container">
            <canvas id="efficiencyTorqueChart"></canvas>
        </div>
    </div>

    <div class="bg-white p-6 rounded-lg shadow">
        <h2 class="text-xl font-bold mb-4">Size Comparison</h2>
        <div class="chart-container">
            <canvas id="sizeChart"></canvas>
        </div>
    </div>

    <div class="bg-white p-6 rounded-lg shadow">
        <h2 class="text-xl font-bold mb-4">Current vs Resistance</h2>
        <div class="chart-container">
            <canvas id="currentResistanceChart"></canvas>
        </div>
    </div>

    <div class="bg-white p-6 rounded-lg shadow">
        <h2 class="text-xl font-bold mb-4">Poles/Coils Distribution</h2>
        <div class="chart-container">
            <canvas id="polesCoilsChart"></canvas>
        </div>
    </div>

    <div class="bg-white p-6 rounded-lg shadow">
        <h2 class="text-xl font-bold mb-4">Turns vs Efficiency</h2>
        <div class="chart-container">
            <canvas id="turnsEfficiencyChart"></canvas>
        </div>
    </div>

    <div class="bg-white p-6 rounded-lg shadow">
        <h2 class="text-xl font-bold mb-4">Power Analysis</h2>
        <div class="chart-container">
            <canvas id="powerAnalysisChart"></canvas>
        </div>
    </div>
</div>"""

    def _generate_data_table(self) -> str:
        """Generate the data table section."""
        rows = ""
        for config in self.configs:
            rows += f"""
            <tr class="border-t">
                <td class="px-4 py-2">{config.poles}P/{config.coils}C</td>
                <td class="px-4 py-2">{config.turns_per_coil}</td>
                <td class="px-4 py-2">{config.estimated_torque:.3f}</td>
                <td class="px-4 py-2">{config.current:.1f}</td>
                <td class="px-4 py-2">{config.efficiency:.1f}</td>
                <td class="px-4 py-2">∅{config.outer_radius * 2:.1f}×{config.stator_thickness:.1f}</td>
            </tr>"""

        return f"""
<div class="bg-white p-6 rounded-lg shadow">
    <h2 class="text-xl font-bold mb-4">All Configurations</h2>
    <div class="overflow-x-auto">
        <table class="min-w-full table-auto">
            <thead>
                <tr class="bg-gray-100">
                    <th class="px-4 py-2">Configuration</th>
                    <th class="px-4 py-2">Turns</th>
                    <th class="px-4 py-2">Torque (Nm)</th>
                    <th class="px-4 py-2">Current (A)</th>
                    <th class="px-4 py-2">Efficiency (%)</th>
                    <th class="px-4 py-2">Size (mm)</th>
                </tr>
            </thead>
            <tbody>{rows}
            </tbody>
        </table>
    </div>
</div>"""

    @staticmethod
    def _generate_chart_scripts(config_data: List[Dict[str, Any]]) -> str:
        """Generate the JavaScript for charts."""
        return f"""
<script>
    const configData = {json.dumps(config_data)};

    const commonTooltip = (context) => {{
        const data = context.raw;
        return [
            `Config: ${{configData[context.dataIndex].config}}`,
            `Torque: ${{data.torque?.toFixed(3) || data.x?.toFixed(3)}} Nm`,
            `Efficiency: ${{data.efficiency?.toFixed(1) || data.y?.toFixed(1)}}%`,
            `Current: ${{configData[context.dataIndex].current.toFixed(2)}}A`,
            `Turns: ${{configData[context.dataIndex].turns}}`,
            `Size: ∅${{configData[context.dataIndex].diameter}}×${{configData[context.dataIndex].thickness}}mm`
        ];
    }};

    const commonOptions = {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            tooltip: {{
                callbacks: {{
                    label: commonTooltip
                }}
            }}
        }}
    }};

    // Create Efficiency vs Torque Chart
    new Chart(document.getElementById('efficiencyTorqueChart'), {{
        type: 'scatter',
        data: {{
            datasets: [{{
                label: 'Configurations',
                data: configData.map(c => ({{
                    x: c.torque,
                    y: c.efficiency,
                    torque: c.torque,
                    efficiency: c.efficiency
                }})),
                backgroundColor: '#4CAF50'
            }}]
        }},
        options: {{
            ...commonOptions,
            scales: {{
                x: {{
                    title: {{
                        display: true,
                        text: 'Torque (Nm)'
                    }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: 'Efficiency (%)'
                    }},
                    min: 0,
                    max: 100
                }}
            }}
        }}
    }});

    // Create Size Comparison Chart
    new Chart(document.getElementById('sizeChart'), {{
        type: 'bubble',
        data: {{
            datasets: [{{
                label: 'Motor Sizes',
                data: configData.map(c => ({{
                    x: c.diameter,
                    y: c.thickness,
                    r: c.torque * 10,
                    torque: c.torque,
                    efficiency: c.efficiency
                }})),
                backgroundColor: 'rgba(54, 162, 235, 0.5)'
            }}]
        }},
        options: {{
            ...commonOptions,
            scales: {{
                x: {{
                    title: {{
                        display: true,
                        text: 'Diameter (mm)'
                    }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: 'Thickness (mm)'
                    }}
                }}
            }}
        }}
    }});

    // Create Current vs Resistance Chart
    new Chart(document.getElementById('currentResistanceChart'), {{
        type: 'scatter',
        data: {{
            datasets: [{{
                label: 'Operating Points',
                data: configData.map(c => ({{
                    x: c.resistance,
                    y: c.current,
                    torque: c.torque,
                    efficiency: c.efficiency
                }})),
                backgroundColor: '#FF9800'
            }}]
        }},
        options: {{
            ...commonOptions,
            scales: {{
                x: {{
                    title: {{
                        display: true,
                        text: 'Resistance (Ω)'
                    }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: 'Current (A)'
                    }}
                }}
            }}
        }}
    }});

    // Create Poles/Coils Distribution Chart
    new Chart(document.getElementById('polesCoilsChart'), {{
        type: 'scatter',
        data: {{
            datasets: [{{
                label: 'Configurations',
                data: configData.map(c => ({{
                    x: c.poles,
                    y: c.coils,
                    torque: c.torque,
                    efficiency: c.efficiency
                }})),
                backgroundColor: '#9C27B0'
            }}]
        }},
        options: {{
            ...commonOptions,
            scales: {{
                x: {{
                    title: {{
                        display: true,
                        text: 'Number of Poles'
                    }},
                    ticks: {{
                        stepSize: 2
                    }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: 'Number of Coils'
                    }},
                    ticks: {{
                        stepSize: 2
                    }}
                }}
            }}
        }}
    }});

    // Create Turns vs Efficiency Chart
    new Chart(document.getElementById('turnsEfficiencyChart'), {{
        type: 'scatter',
        data: {{
            datasets: [{{
                label: 'Configurations',
                data: configData.map(c => ({{
                    x: c.turns,
                    y: c.efficiency,
                    torque: c.torque,
                    efficiency: c.efficiency
                }})),
                backgroundColor: '#2196F3'
            }}]
        }},
        options: {{
            ...commonOptions,
            scales: {{
                x: {{
                    title: {{
                        display: true,
                        text: 'Turns per Coil'
                    }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: 'Efficiency (%)'
                    }},
                    min: 0,
                    max: 100
                }}
            }}
        }}
    }});

    // Create Power Analysis Chart
    new Chart(document.getElementById('powerAnalysisChart'), {{
        type: 'scatter',
        data: {{
            datasets: [{{
                label: 'Power Output vs Input',
                data: configData.map(c => ({{
                    x: c.current * 12, // Input power (V*I)
                    y: c.torque * (2 * Math.PI * 3000 / 60), // Mechanical power (T*ω)
                    torque: c.torque,
                    efficiency: c.efficiency
                }})),
                backgroundColor: '#F44336'
            }}]
        }},
        options: {{
            ...commonOptions,
            scales: {{
                x: {{
                    title: {{
                        display: true,
                        text: 'Input Power (W)'
                    }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: 'Output Power (W)'
                    }}
                }}
            }}
        }}
    }});
</script>"""

    def generate_html(self) -> str:
        """Generate complete HTML report with interactive visualizations."""
        # Convert configs to JSON for JavaScript
        config_data = [{
            'config': f"{c.poles}P/{c.coils}C",
            'poles': c.poles,
            'coils': c.coils,
            'turns': c.turns_per_coil,
            'torque': round(c.estimated_torque, 3),
            'resistance': round(c.resistance, 2),
            'current': round(c.current, 2),
            'efficiency': round(c.efficiency, 1),
            'diameter': round(c.outer_radius * 2, 1),
            'thickness': round(c.stator_thickness, 1)
        } for c in self.configs]

        # Find the best configurations
        best_efficiency = max(self.configs, key=lambda x: x.efficiency)
        best_torque = max(self.configs, key=lambda x: x.estimated_torque)
        simplest = min(self.configs, key=lambda x: x.poles + x.coils)
        most_compact = min(self.configs,
                           key=lambda x: x.outer_radius * 2 * x.stator_thickness)

        # Build HTML components
        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            self._generate_head(),
            '<body class="bg-gray-100 p-8">',
            self._generate_header(),
            self._generate_stat_cards(best_efficiency, best_torque, simplest, most_compact),
            self._generate_chart_containers(),
            self._generate_data_table(),
            self._generate_chart_scripts(config_data),
            "</body>",
            "</html>"
        ]

        return "\n".join(html_parts)

    def parse_output(self, text: str) -> List[UnifiedMotorParameters]:
        """Parse the motor calculator output text and extract configurations."""
        # More forgiving regex pattern that matches the actual output format
        config_pattern = r"Configuration \d+:\s*\n" + \
                         r"- Poles: (\d+)\s*\n" + \
                         r"- Coils: (\d+)\s*\n" + \
                         r"- Turns per coil: (\d+)\s*\n" + \
                         r"- Estimated torque: ([\d.]+) Nm\s*\n" + \
                         r"- Total resistance: ([\d.]+) Ω\s*\n" + \
                         r"- Operating current: ([\d.]+) A\s*\n" + \
                         r"- Efficiency: ([-\d.]+)%\s*\n" + \
                         r"- Dimensions: ([\d.]+)mm diameter, ([\d.]+)mm thick"

        # Extract parameters from header section with error handling
        try:
            voltage_match = re.search(r"- Voltage: ([\d.]+) V", text)
            max_current_match = re.search(r"- Max current: ([\d.]+) A", text)
            wire_diameter_match = re.search(r"- Wire diameter: ([\d.]+) mm", text)
            target_torque_match = re.search(r"- Torque range: ([\d.]+) to ([\d.]+) Nm", text)

            voltage = float(voltage_match.group(1)) if voltage_match else 24.0
            max_current = float(max_current_match.group(1)) if max_current_match else 10.0
            wire_diameter = float(wire_diameter_match.group(1)) if wire_diameter_match else 0.65
            target_torque = float(target_torque_match.group(2)) if target_torque_match else 0.1

        except (AttributeError, ValueError) as e:
            print(f"Warning: Error parsing header parameters: {e}")
            print("Using default values")
            voltage = 24.0
            max_current = 10.0
            wire_diameter = 0.65
            target_torque = 0.1

        # Find all configurations
        matches = list(re.finditer(config_pattern, text, re.MULTILINE))

        if not matches:
            print("Warning: No configurations found! Debugging information:")
            print(f"Text length: {len(text)}")
            print("First 200 characters of text:")
            print(text[:200])
            print("\nRegex pattern used:")
            print(config_pattern)
            return []

        print(f"Found {len(matches)} configurations")
        self.configs = []

        for match in matches:
            try:
                # Extract dimensions
                diameter = float(match.group(8))
                thickness = float(match.group(9))

                # Create configuration with extracted parameters
                config = UnifiedMotorParameters(
                    poles=int(match.group(1)),
                    coils=int(match.group(2)),
                    turns_per_coil=int(match.group(3)),
                    wire_diameter=wire_diameter,
                    voltage=voltage,
                    max_current=max_current,
                    magnet_type="circle",  # Default value
                    magnet_width=10.0,  # Default value
                    magnet_length=10.0,  # Default value
                    magnet_thickness=3.0,  # Default value
                    magnet_br=1.2,  # Default value for N42 NdFeB
                    outer_radius=diameter / 2,
                    inner_radius=(diameter / 2) * 0.3,  # Using typical ratio
                    target_diameter=diameter,
                    air_gap=1.0,  # Default value
                    stator_thickness=thickness,
                    rotor_thickness=5.0,  # Default value
                    torque=target_torque,
                    tolerance=0.2,  # Default value
                    target_torque=target_torque,
                    estimated_torque=float(match.group(4)),
                    efficiency=float(match.group(7)),
                    resistance=float(match.group(5)),
                    current=float(match.group(6))
                )
                self.configs.append(config)

            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing configuration: {e}")
                print(f"Problematic match: {match.groups()}")
                continue

        return self.configs

    def save_report(self, output_text: str):
        """Parse the output and save the HTML report."""
        print("Parsing configurations...")
        self.parse_output(output_text)

        if not self.configs:
            print("Error: No valid configurations found to generate report")
            return

        print(f"Generating HTML report with {len(self.configs)} configurations...")
        html = self.generate_html()

        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"Report saved to {self.output_file}")


def analyze_motor(motor_output: str = 'motor_output_unified.txt',
                  motor_analysis: str = 'motor_analysis_unified.html'):
    """Analyze motor configurations and generate HTML report."""
    with open(motor_output, 'r') as f:
        motor_output_text = f.read()

    analyzer = MotorAnalyzer(motor_analysis)
    analyzer.save_report(motor_output_text)


if __name__ == "__main__":
    analyze_motor()
