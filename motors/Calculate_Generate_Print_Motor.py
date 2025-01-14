from pathlib import Path
import sys
from typing import List

from AxialMotorCalculator import AxialMotorCalculator
from MotorAnalyzer import MotorAnalyzer
from GeneratePrintableMotor import generate_printable_parts
from MotorParameters import MotorParameters


def validate_printable_parameters(params: MotorParameters) -> bool:
    """Validate parameters required for generating printable parts."""
    required_params = [
        ('outer_radius', params.outer_radius),
        ('inner_radius', params.inner_radius),
        ('stator_thickness', params.stator_thickness),
        ('rotor_thickness', params.rotor_thickness),
        ('air_gap', params.air_gap),
        ('wire_diameter', params.wire_diameter),
        ('magnet_width', params.magnet_width),
        ('magnet_length', params.magnet_length),
        ('magnet_thickness', params.magnet_thickness)
    ]

    for name, value in required_params:
        if value is None or value <= 0:
            print(f"Error: {name} must be a positive number")
            return False

    if params.inner_radius >= params.outer_radius:
        print("Error: inner_radius must be less than outer_radius")
        return False

    return True


def create_base_parameters(target_torque: float = 0.1) -> MotorParameters:
    """Create default parameters with specified target torque."""
    return MotorParameters(
        poles=4,  # Will be varied in calculations
        coils=6,  # Will be varied in calculations
        turns_per_coil=100,  # Will be varied in calculations
        wire_diameter=0.65,  # mm
        voltage=12.0,  # V
        max_current=10.0,  # A
        magnet_type="circle",
        magnet_width=10.0,  # mm
        magnet_length=10.0,  # mm
        magnet_thickness=3.0,  # mm
        magnet_br=1.2,  # Tesla (N42 NdFeB)
        outer_radius=50.0,  # mm
        inner_radius=10.0,  # mm
        air_gap=1.0,  # mm
        stator_thickness=15.0,  # mm
        rotor_thickness=5.0,  # mm
        target_diameter=50,  # mm
        torque=0,  # Nm
        target_torque=target_torque,  # Nm
        estimated_torque=0.0,
        tolerance=0.2,  # ±20%
        efficiency=0.0,
        resistance=0.0,
        current=0.0,
        coil_width=None,
        coil_height=None,
        total_height=None
    )


def find_optimal_configuration(configs: List[MotorParameters],
                               target_torque: float,
                               tolerance: float = 0.2) -> MotorParameters:
    """
    Find the optimal configuration based on efficiency and target torque.

    Args:
        configs: List of valid motor configurations
        target_torque: Desired torque in Nm
        tolerance: Acceptable torque deviation (±%)

    Returns:
        Best configuration based on efficiency within torque constraints
    """
    min_torque = target_torque * (1 - tolerance)
    max_torque = target_torque * (1 + tolerance)

    valid_configs = [
        config for config in configs
        if min_torque <= config.estimated_torque <= max_torque
    ]

    if not valid_configs:
        print("No configurations found within torque constraints.")
        print("Using configuration closest to target torque...")
        return min(configs, key=lambda x: abs(x.estimated_torque - target_torque))

    # Find configuration with the highest efficiency
    best_config = max(valid_configs, key=lambda x: x.efficiency)
    return best_config


def generate_motor(target_torque: float = 0.1,
                   output_dir: str = "motor_output",
                   should_analyze_only: bool = False) -> None:
    """
    Generate motor design files based on target torque.

    Args:
        target_torque: Desired torque in Nm
        output_dir: Directory for output files
        should_analyze_only: If True, only perform analysis without generating printable files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize with base parameters
    base_params = create_base_parameters(target_torque)

    # Initialize calculator with all required parameters
    calculator = AxialMotorCalculator(
        given_params=base_params
    )

    # Generate configurations
    config_output_file = output_path / "motor_configurations.txt"
    calculator.write_motor_output(config_output_file.as_posix())

    report_output_file = output_path / "motor_analysis.html"

    # Analyze configurations
    analyzer = MotorAnalyzer(report_output_file.as_posix())
    with open(config_output_file, 'r') as f:
        analyzer.save_report(f.read())

    # Get configurations and find optimal one
    configs = calculator.find_viable_configurations()
    if not configs:
        print("No viable configurations found!")
        sys.exit(1)

    best_config = find_optimal_configuration(configs, target_torque)

    print("\nSelected Configuration:")
    print(f"Poles: {best_config.poles}")
    print(f"Coils: {best_config.coils}")
    print(f"Turns per coil: {best_config.turns_per_coil}")
    print(f"Estimated torque: {best_config.estimated_torque:.3f} Nm")
    print(f"Efficiency: {best_config.efficiency:.1f}%")

    if should_analyze_only:
        return

    # Generate printable parts
    print("\nGenerating printable parts...")
    print_settings = {
        'wall_thickness': 2.0,
        'tolerance': 0.2,
        'layer_height': 0.2,
        'segments': 100,
        'coil_orientation': 'axial',
        'cutaway': True
    }

    # Ensure all required parameters are set
    # Add missing dimensional parameters if they're None
    if best_config.coil_width is None:
        best_config.coil_width = best_config.outer_radius * 0.2  # 20% of outer radius
    if best_config.coil_height is None:
        best_config.coil_height = best_config.stator_thickness * 0.8  # 80% of stator thickness
    if best_config.total_height is None:
        best_config.total_height = (best_config.stator_thickness +
                                    2 * best_config.rotor_thickness +
                                    2 * best_config.air_gap)

    print("\nPrintable Parameters:")
    print(f"Coil Width: {best_config.coil_width:.2f} mm")
    print(f"Coil Height: {best_config.coil_height:.2f} mm")
    print(f"Total Height: {best_config.total_height:.2f} mm")

    parts = generate_printable_parts(
        best_config,
        output_prefix=f"motor_{best_config.poles}p_{best_config.coils}c",
        output_location=f"{output_path.as_posix()}",
        print_settings=print_settings
    )

    print(f"\nFiles generated in {output_path}:")
    for name in parts.keys():
        print(f"- {name}")


if __name__ == "__main__":
    # Example usage with hardcoded values
    torque = 0.2
    output = "generated_motors"
    analyze_only = False

    try:
        generate_motor(torque, output, analyze_only)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    """
    # Generate motor with default 0.1 Nm torque
    python generate_printable_motor.py

    # Generate motor with custom torque
    python generate_printable_motor.py --torque 0.2

    # Only analyze configurations without generating printable files
    python generate_printable_motor.py --analyze-only

    # Specify custom output directory
    python generate_printable_motor.py --output my_motor_files
    """

    #  python generate_printable_motor.py --output my_motor_files --torque 0.2

    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Generate printable axial motor files')
    # parser.add_argument('--torque', type=float, default=0.1,
    #                     help='Target torque in Nm (default: 0.1)')
    # parser.add_argument('--output', type=str, default='motor_output',
    #                     help='Output directory (default: motor_output)')
    # parser.add_argument('--analyze-only', action='store_true',
    #                     help='Only perform analysis without generating printable files')
    #
    # args = parser.parse_args()
    #
    # try:
    #     generate_motor(args.torque, args.output, args.analyze_only)
    # except Exception as e:
    #     print(f"Error: {e}")
    #     sys.exit(1)
