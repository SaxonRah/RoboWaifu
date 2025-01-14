import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from MotorParameters import MotorParameters
from AxialMotorCalculator import AxialMotorCalculator


class MotorAnalysisPlotter:
    def __init__(self, calculator: AxialMotorCalculator):
        self.calculator = calculator
        plt.style.use('dark_background')

    def plot_all_analyses(self) -> None:
        """Generate comprehensive motor analysis dashboard"""
        print("Generating comprehensive motor analysis dashboard...")

        # Get all required data
        magnetic_params = self.calculator.calculate_magnetic_circuit()
        coil_params = self.calculator.calculate_coil_parameters()
        performance = self.calculator.calculate_performance()
        report = self.calculator.generate_design_report()

        # Create a single figure with subplots
        fig = plt.figure(figsize=(20, 25))
        fig.suptitle('Advanced Motor Analysis Dashboard', fontsize=16, y=0.95)

        # Create a 5x2 grid of subplots
        gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)

        # 1. Magnetic Circuit Analysis (Row 1)
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.linspace(0, self.calculator.params.outer_radius, 100)
        flux_density = magnetic_params['flux_density'] * np.exp(-x / self.calculator.params.outer_radius)
        ax1.plot(x, flux_density, 'b-', label='Radial Distribution')
        ax1.set_title('Magnetic Flux Density Distribution')
        ax1.set_xlabel('Radius (mm)')
        ax1.set_ylabel('Flux Density (T)')
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        mmf_components = ['Magnet MMF', 'Air Gap MMF', 'Total MMF']
        mmf_values = [
            magnetic_params['mmf'] * 0.7,  # Magnet
            magnetic_params['mmf'] * 0.3,  # Air gap
            magnetic_params['mmf']  # Total
        ]
        ax2.bar(mmf_components, mmf_values)
        ax2.set_title('Magnetomotive Force Distribution')
        ax2.set_ylabel('MMF (A-turns)')
        ax2.grid(True)

        # 2. Coil and Winding Analysis (Row 2)
        ax3 = fig.add_subplot(gs[1, 0])
        coil_metrics = ['Width', 'Height', 'Wire Length (m)']
        coil_values = [
            coil_params['coil_width'],
            coil_params['coil_height'],
            coil_params['wire_length'] / 1000  # Convert to meters
        ]
        bars3 = ax3.bar(coil_metrics, coil_values)
        ax3.set_title('Coil Geometry and Wire Usage')
        ax3.set_ylabel('Dimensions (mm) / Length (m)')
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}', ha='center', va='bottom')
        ax3.grid(True)

        ax4 = fig.add_subplot(gs[1, 1])
        winding_params = [
            self.calculator.params.turns_per_coil,
            coil_params['turns_per_layer'],
            coil_params['num_layers']
        ]
        winding_labels = ['Total Turns', 'Turns/Layer', 'Num Layers']
        bars4 = ax4.bar(winding_labels, winding_params)
        ax4.set_title('Winding Configuration')
        ax4.set_ylabel('Count')
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.0f}', ha='center', va='bottom')
        ax4.grid(True)

        # 3. Performance and Efficiency (Row 3)
        ax5 = fig.add_subplot(gs[2, 0])
        power_metrics = ['Input', 'Output', 'Copper Loss', 'Iron Loss']
        power_values = [
            performance['input_power'],
            performance['output_power'],
            performance['copper_loss'],
            performance['iron_loss']
        ]
        bars5 = ax5.bar(power_metrics, power_values,
                        color=['blue', 'green', 'red', 'orange'])
        ax5.set_title('Power Flow Analysis')
        ax5.set_ylabel('Power (W)')
        for bar in bars5:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}W', ha='center', va='bottom')
        ax5.grid(True)

        ax6 = fig.add_subplot(gs[2, 1])
        # Create theoretical efficiency curve
        currents = np.linspace(0, self.calculator.params.max_current, 100)
        efficiencies = [self.calculator.calculate_efficiency(i, coil_params['resistance'])
                        for i in currents]
        ax6.plot(currents, efficiencies, 'g-', label='Efficiency Curve')
        ax6.scatter([self.calculator.params.current], [performance['efficiency']],
                    color='red', s=100, label='Operating Point')
        ax6.set_title('Efficiency vs Current')
        ax6.set_xlabel('Current (A)')
        ax6.set_ylabel('Efficiency (%)')
        ax6.grid(True)
        ax6.legend()

        # 4. Thermal and Electrical Analysis (Row 4)
        ax7 = fig.add_subplot(gs[3, 0])
        electrical_metrics = ['Voltage (V)', 'Current (A)', 'Resistance (Ω)']
        electrical_values = [
            self.calculator.params.voltage,
            self.calculator.params.current,
            coil_params['resistance']
        ]
        bars7 = ax7.bar(electrical_metrics, electrical_values,
                        color=['yellow', 'cyan', 'magenta'])
        ax7.set_title('Electrical Parameters')
        ax7.set_ylabel('Value')
        for bar in bars7:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom')
        ax7.grid(True)

        ax8 = fig.add_subplot(gs[3, 1])
        # Calculate current density and expected temperature rise
        wire_area = np.pi * (self.calculator.params.wire_diameter / 2) ** 2
        current_density = self.calculator.params.current / wire_area
        temp_rise = current_density * 0.8  # Simplified temperature rise model
        thermal_metrics = ['Current Density (A/mm²)', 'Est. Temp Rise (°C)']
        thermal_values = [current_density, temp_rise]
        bars8 = ax8.bar(thermal_metrics, thermal_values, color=['red', 'orange'])
        ax8.set_title('Thermal Characteristics')
        ax8.set_ylabel('Value')
        for bar in bars8:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}', ha='center', va='bottom')
        ax8.grid(True)

        # 5. Mechanical and Torque Analysis (Row 5)
        ax9 = fig.add_subplot(gs[4, 0])
        mechanical_dims = [
            report['geometry']['outer_diameter'],
            report['geometry']['inner_diameter'],
            report['geometry']['total_height'],
            report['geometry']['air_gap']
        ]
        dim_labels = ['Outer Dia', 'Inner Dia', 'Height', 'Air Gap']
        bars9 = ax9.bar(dim_labels, mechanical_dims)
        ax9.set_title('Mechanical Dimensions')
        ax9.set_ylabel('mm')
        for bar in bars9:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}', ha='center', va='bottom')
        ax9.grid(True)

        ax10 = fig.add_subplot(gs[4, 1])
        # Calculate torque characteristics
        speeds = np.linspace(0, 3000, 100)  # RPM
        torques = [self.calculator.params.estimated_torque * (1 - s / 3000)
                   for s in speeds]  # Simple speed-torque curve
        ax10.plot(speeds, torques, 'y-', label='Torque Curve')
        ax10.scatter([1500], [self.calculator.params.estimated_torque],
                     color='red', s=100, label='Design Point')
        ax10.set_title('Speed-Torque Characteristics')
        ax10.set_xlabel('Speed (RPM)')
        ax10.set_ylabel('Torque (Nm)')
        ax10.grid(True)
        ax10.legend()

        plt.show()
        print("Advanced analysis dashboard generated successfully!")


def analyze_motor_with_plots(given_params: MotorParameters) -> None:
    """Main function to analyze motor and generate plots"""
    calculator = AxialMotorCalculator(given_params)
    plotter = MotorAnalysisPlotter(calculator)

    # Print detailed analysis
    calculator.analyze_motor_design()

    # Generate all plots in a single dashboard
    plotter.plot_all_analyses()


if __name__ == "__main__":
    # Create default parameters and run analysis
    from MotorParameters import create_default_parameters

    params = create_default_parameters()
    analyze_motor_with_plots(params)