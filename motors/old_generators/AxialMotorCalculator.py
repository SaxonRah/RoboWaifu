import math
import itertools
from dataclasses import dataclass
from typing import List, Optional, Union

from MotorAnalyzer import MotorAnalyzer


@dataclass
class RectangleMagnetSpec:
    width: float  # mm
    length: float  # mm
    thickness: float  # mm
    br: float  # Tesla (residual magnetization)


@dataclass
class CircleMagnetSpec:
    diameter: float  # mm
    thickness: float  # mm
    br: float  # Tesla (residual magnetization)


@dataclass
class WireSpec:
    diameter: float  # mm
    resistance_per_m: float  # ohms/meter at 20°C


@dataclass
class MotorConfiguration:
    outer_radius: float  # mm
    inner_radius: float  # mm
    num_poles: int
    num_coils: int
    turns_per_coil: int
    estimated_torque: float  # Nm
    total_resistance: float  # ohms
    efficiency: float  # percentage
    coil_width: float  # mm
    stator_thickness: float  # mm


class AxialMotorCalculator:
    def __init__(self,
                 target_diameter: float,  # mm
                 target_torque: float,  # Nm
                 magnet_spec: Union[CircleMagnetSpec, RectangleMagnetSpec],  # Accepts both types
                 wire_spec: WireSpec,
                 tolerance: float = 0.2,  # ±20% by default
                 voltage: float = 12.0,  # V
                 max_current: float = 10.0):  # A

        self.target_diameter = target_diameter
        self.target_torque = target_torque
        self.magnet_spec = magnet_spec
        self.wire_spec = wire_spec
        self.tolerance = tolerance
        self.voltage = voltage
        self.max_current = max_current

        # Physical constants
        self.mu0 = 4 * math.pi * 1e-7  # H/m (vacuum permeability)
        self.copper_resistivity = 1.68e-8  # Ω⋅m at 20°C

        # Design constraints
        self.min_air_gap = 1.0  # mm
        self.min_wall_thickness = 2.0  # mm

    def calculate_magnetic_field_with_thickness(self, air_gap: float) -> float:
        """Calculate magnetic field strength in the air gap."""
        # Simplified magnetic circuit calculation
        if isinstance(self.magnet_spec, CircleMagnetSpec):
            thickness = self.magnet_spec.thickness
        elif isinstance(self.magnet_spec, RectangleMagnetSpec):
            thickness = self.magnet_spec.thickness
        else:
            raise TypeError("Unsupported magnet specification")

        return (self.magnet_spec.br * thickness) / (
                thickness + (self.mu0 * air_gap / self.magnet_spec.br))

    def calculate_magnetic_field_with_area(self, air_gap: float) -> float:
        """Calculate magnetic field using area and flux."""
        # Convert air_gap to meters
        air_gap_m = air_gap * 1e-3

        if isinstance(self.magnet_spec, CircleMagnetSpec):
            area = math.pi * (self.magnet_spec.diameter / 2000) ** 2  # mm² to m²
        elif isinstance(self.magnet_spec, RectangleMagnetSpec):
            area = (self.magnet_spec.width * self.magnet_spec.length) * 1e-6  # mm² to m²
        else:
            raise TypeError("Unsupported magnet specification")

        # Convert magnet thickness to meters
        thickness_m = self.magnet_spec.thickness * 1e-3

        # Calculate reluctance of air gap
        reluctance_gap = air_gap_m / (self.mu0 * area)

        # Calculate reluctance of magnet
        reluctance_magnet = thickness_m / (self.mu0 * self.magnet_spec.br * area)

        # Calculate effective magnetic field
        return self.magnet_spec.br / (1 + reluctance_gap / reluctance_magnet)

    @staticmethod
    def calculate_torque2(mean_radius: float, num_poles: int, num_coils: int,
                         turns_per_coil: int, current: float, b_field: float) -> float:
        """Calculate theoretical torque for given configuration"""
        # Convert mean_radius to meters
        mean_radius_m = mean_radius * 1e-3

        # Active length of conductor in field (per coil)
        active_length = 2 * math.pi * mean_radius_m / num_poles

        # Total force for all coils
        total_force = b_field * current * active_length * turns_per_coil * num_coils

        # Torque = Force * radius
        return total_force * mean_radius_m  # Already in Nm since we converted to meters

    def calculate_torque(self, mean_radius: float, num_poles: int, num_coils: int,
                         turns_per_coil: int, current: float, b_field: float) -> float:
        """Calculate theoretical torque for given configuration"""
        # Convert mean_radius to meters
        mean_radius_m = mean_radius * 1e-3

        # Calculate active length for a single coil side
        active_length = mean_radius_m * math.pi / num_poles

        # Each coil has two active sides
        total_force = 2 * b_field * current * active_length * turns_per_coil * num_coils

        # Torque = Force * radius
        return total_force * mean_radius_m

    def calculate_resistance2(self,
                             mean_radius: float,
                             num_coils: int,
                             turns_per_coil: int) -> float:
        """Calculate total resistance of all coils"""
        # Average length of one turn
        turn_length = 2 * math.pi * mean_radius * 1e-3  # Convert to meters

        # Total wire length
        total_length = turn_length * turns_per_coil * num_coils

        return self.wire_spec.resistance_per_m * total_length

    def calculate_resistance(self, mean_radius: float, num_coils: int, turns_per_coil: int) -> float:
        """Calculate total resistance of all coils"""
        # Calculate average length of one turn including end turns
        # Add 20% for end turns
        turn_length = 2.2 * math.pi * mean_radius * 1e-3  # Convert to meters

        # Total wire length
        total_length = turn_length * turns_per_coil * num_coils

        return self.wire_spec.resistance_per_m * total_length

    def calculate_efficiency2(self, current: float, resistance: float) -> float:
        """Calculate motor efficiency"""
        # Power losses in copper
        copper_loss = current * current * resistance

        # Input power
        input_power = self.voltage * current

        return (input_power - copper_loss) / input_power * 100

    def calculate_efficiency(self, current: float, resistance: float) -> float:
        """Calculate motor efficiency"""
        # Power losses in copper
        copper_loss = current * current * resistance

        # Mechanical power output (assuming 80% of electrical power converts to mechanical)
        mech_power = self.voltage * current * 0.8

        # Total input power
        input_power = self.voltage * current

        # Efficiency calculation
        if input_power > 0:
            return (mech_power - copper_loss) / input_power * 100
        return 0.0

    def find_viable_configurations(self) -> List[MotorConfiguration]:
        """Search for viable motor configurations meeting the specifications"""
        viable_configs = []

        # Calculate available space for magnets and coils
        max_radius = self.target_diameter / 2 - self.min_wall_thickness
        min_radius = max_radius * 0.3

        # Parameter ranges to explore
        pole_counts = range(4, 32, 2)  # Common pole counts
        coil_multipliers = [1.25, 1.5, 1.75, 2.0]  # Must be > 1 to ensure coils > poles
        turns_range = range(5, 101, 5)  # Reduced maximum turns

        tolerance_range = (
            self.target_torque * (1 - self.tolerance),
            self.target_torque * (1 + self.tolerance)
        )

        print(f"Searching for configurations with torque between {tolerance_range[0]:.2f} "
              f"and {tolerance_range[1]:.2f} Nm")

        # Calculate magnetic field once since it's independent of other parameters
        b_field = self.calculate_magnetic_field_with_area(self.min_air_gap)
        print(f"Magnetic field strength: {b_field:.3f} Tesla")

        mean_radius = (max_radius + min_radius) / 2
        print(f"Mean radius: {mean_radius:.1f} mm")

        for num_poles in pole_counts:
            for multiplier in coil_multipliers:
                num_coils = int(num_poles * multiplier)

                # Skip invalid combinations:
                # - Must have even number of coils
                # - Must have more coils than poles
                # - 1:1 ratio not allowed
                if (num_coils % 2 != 0 or
                        num_coils <= num_poles or
                        num_coils == num_poles):
                    continue

                print(f"\nTesting pole:coil ratio {num_poles}:{num_coils}")

                for turns in turns_range:
                    # Calculate resistance
                    resistance = self.calculate_resistance(mean_radius, num_coils, turns)

                    # Calculate current (limited by max current)
                    current = min(self.voltage / resistance, self.max_current)

                    if current < 0.5:  # Skip if current is too low
                        continue

                    # Calculate torque
                    torque = self.calculate_torque(
                        mean_radius, num_poles, num_coils, turns, current, b_field)

                    # Debug print
                    print(f"Testing - Turns: {turns}, Current: {current:.1f}A, "
                          f"Torque: {torque:.3f} Nm")

                    # Check if configuration meets torque requirements
                    if tolerance_range[0] <= torque <= tolerance_range[1]:
                        efficiency = self.calculate_efficiency(current, resistance)

                        # Skip configurations with very low efficiency
                        if efficiency < 40:
                            continue

                        # Calculate physical dimensions
                        coil_width = (num_coils * self.wire_spec.diameter *
                                      turns / (2 * math.pi))
                        stator_thickness = (self.magnet_spec.thickness +
                                            2 * self.min_air_gap)

                        config = MotorConfiguration(
                            outer_radius=max_radius,
                            inner_radius=min_radius,
                            num_poles=num_poles,
                            num_coils=num_coils,
                            turns_per_coil=turns,
                            estimated_torque=torque,
                            total_resistance=resistance,
                            efficiency=efficiency,
                            coil_width=coil_width,
                            stator_thickness=stator_thickness
                        )

                        viable_configs.append(config)
                        print(f"\nFound viable config!")
                        print(f"Poles: {num_poles}, Coils: {num_coils}, "
                              f"Turns: {turns}")
                        print(f"Current: {current:.1f}A")
                        print(f"Torque: {torque:.3f} Nm")
                        print(f"Efficiency: {efficiency:.1f}%")
                        print(f"Resistance: {resistance:.2f} Ω")

        return viable_configs


def magnet_wire_resistance(diameter_mm):
    """
    Calculate the resistance per meter (Ω/m) of a copper magnet wire based on its diameter.

    Parameters:
    diameter_mm (float): Diameter of the wire in millimeters.

    Returns:
    float: Resistance per meter in ohms (Ω/m).
    """

    # Resistivity of copper in Ω·m
    resistivity = 1.68e-8  # Ω·m

    # Convert diameter to radius in meters
    radius_m = (diameter_mm / 2) / 1000  # mm to m

    # Calculate cross-sectional area in m²
    area_m2 = math.pi * (radius_m ** 2)

    # Calculate resistance per meter
    resistance_per_meter = resistivity / area_m2

    return resistance_per_meter

    # Example usage:
    #  diameter = 0.65  # Diameter in mm
    #  resistance = magnet_wire_resistance(diameter)
    #  print(f"Resistance per meter for {diameter} mm diameter magnet wire: {resistance:.4f} Ω/m")


def write_motor_output(filename: str, calculator: AxialMotorCalculator):
    """Generate and write motor configuration results to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        # Calculate config parameters
        max_radius = calculator.target_diameter / 2 - calculator.min_wall_thickness
        min_radius = max_radius * 0.3
        mean_radius = (max_radius + min_radius) / 2
        b_field = calculator.calculate_magnetic_field_with_area(calculator.min_air_gap)
        tolerance_range = (
            calculator.target_torque * (1 - calculator.tolerance),
            calculator.target_torque * (1 + calculator.tolerance)
        )

        # Write initial parameters
        f.write(f"Searching for configurations with torque between {tolerance_range[0]:.2f} "
                f"and {tolerance_range[1]:.2f} Nm\n")
        f.write(f"Magnetic field strength: {b_field:.3f} Tesla\n")
        f.write(f"Mean radius: {mean_radius:.1f} mm\n\n")

        # Test configurations
        pole_counts = range(4, 32, 2)
        coil_multipliers = [1.25, 1.5, 1.75, 2.0]
        turns_range = range(5, 101, 5)
        viable_configs = []

        for num_poles in pole_counts:
            for multiplier in coil_multipliers:
                num_coils = int(num_poles * multiplier)

                if (num_coils % 2 != 0 or
                    num_coils <= num_poles or
                    num_coils == num_poles):
                    continue

                f.write(f"\nTesting pole:coil ratio {num_poles}:{num_coils}\n")

                for turns in turns_range:
                    resistance = calculator.calculate_resistance(mean_radius, num_coils, turns)
                    current = min(calculator.voltage / resistance, calculator.max_current)

                    if current < 0.5:
                        continue

                    torque = calculator.calculate_torque(
                        mean_radius, num_poles, num_coils, turns, current, b_field)

                    f.write(f"Testing - Turns: {turns}, Current: {current:.1f}A, "
                           f"Torque: {torque:.3f} Nm\n")

                    if tolerance_range[0] <= torque <= tolerance_range[1]:
                        efficiency = calculator.calculate_efficiency(current, resistance)

                        if efficiency < 40:
                            continue

                        coil_width = (num_coils * calculator.wire_spec.diameter *
                                    turns / (2 * math.pi))
                        stator_thickness = (calculator.magnet_spec.thickness +
                                          2 * calculator.min_air_gap)

                        config = MotorConfiguration(
                            outer_radius=max_radius,
                            inner_radius=min_radius,
                            num_poles=num_poles,
                            num_coils=num_coils,
                            turns_per_coil=turns,
                            estimated_torque=torque,
                            total_resistance=resistance,
                            efficiency=efficiency,
                            coil_width=coil_width,
                            stator_thickness=stator_thickness
                        )
                        viable_configs.append(config)

                        f.write(f"\nFound viable config!\n")
                        f.write(f"Poles: {num_poles}, Coils: {num_coils}, "
                              f"Turns: {turns}\n")
                        f.write(f"Current: {current:.1f}A\n")
                        f.write(f"Torque: {torque:.3f} Nm\n")
                        f.write(f"Efficiency: {efficiency:.1f}%\n")
                        f.write(f"Resistance: {resistance:.2f} Ω\n")

        # Write final summary
        f.write(f"\nFound {len(viable_configs)} viable configurations:\n")
        for i, config in enumerate(viable_configs, 1):
            f.write(f"\nConfiguration {i}:\n")
            f.write(f"Poles: {config.num_poles}\n")
            f.write(f"Coils: {config.num_coils}\n")
            f.write(f"Turns per coil: {config.turns_per_coil}\n")
            f.write(f"Estimated torque: {config.estimated_torque:.3f} Nm\n")
            f.write(f"Total resistance: {config.total_resistance:.2f} Ω\n")
            f.write(f"Efficiency: {config.efficiency:.1f}%\n")
            f.write(f"Dimensions: {config.outer_radius * 2:.1f}mm diameter, {config.stator_thickness:.1f}mm thick\n")


def find_motor_configurations(given_calculator: AxialMotorCalculator, given_filename: str = 'motor_output.txt'):

    motor_configurations = given_calculator.find_viable_configurations()

    print(f"Found {len(motor_configurations)} viable configurations:")
    for i, config in enumerate(motor_configurations, 1):
        print(f"\nConfiguration {i}:")
        print(f"Poles: {config.num_poles}")
        print(f"Coils: {config.num_coils}")
        print(f"Turns per coil: {config.turns_per_coil}")
        print(f"Estimated torque: {config.estimated_torque:.3f} Nm")
        print(f"Total resistance: {config.total_resistance:.2f} Ω")
        print(f"Efficiency: {config.efficiency:.1f}%")
        print(f"Dimensions: {config.outer_radius * 2:.1f}mm diameter, {config.stator_thickness:.1f}mm thick")

    write_motor_output(given_filename, given_calculator)


def generate_motor_report(given_filename: str = 'motor_output.txt'):
    with open(given_filename, 'r') as f:
        output_text = f.read()

    analyzer = MotorAnalyzer('motor_analysis.html')
    analyzer.save_report(output_text)


if __name__ == "__main__":
    # Example parameters

    # magnet = RectangleMagnetSpec(
    #     width=10.0,  # mm
    #     length=5.0,  # mm
    #     thickness=3.0,  # mm
    #     br=1.2  # Tesla (N42 NdFeB)
    # )

    magnet = CircleMagnetSpec(
        diameter=10.0,  # mm
        thickness=3.0,  # mm
        br=1.2  # Tesla (N42 NdFeB)
    )

    wire_diameter = 0.65  # mm
    wire = WireSpec(
        diameter=wire_diameter,  # mm
        resistance_per_m=magnet_wire_resistance(wire_diameter)
    )

    calculator = AxialMotorCalculator(
        target_diameter=50.0,  # mm
        target_torque=0.1,  # Nm
        magnet_spec=magnet,
        wire_spec=wire,
        tolerance=0.2,  # ±20%
        voltage=12.0,  # V
        max_current=5.0  # A
    )

    find_motor_configurations(calculator)
    generate_motor_report()
