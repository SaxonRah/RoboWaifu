import math
from typing import List

from MotorAnalyzer_Unified import MotorAnalyzer
from MotorParameters_Unified import UnifiedMotorParameters


class AxialMotorCalculator:

    def __init__(
            self,
            given_params: UnifiedMotorParameters):
        """Initialize calculator with basic required parameters"""
        self.params = given_params

        # Electrical
        self.wire_resistance = magnet_wire_resistance(self.params.wire_diameter)
        self._validate_wire_selection()

        # Physical constants
        self.mu0 = 4 * math.pi * 1e-7  # H/m (vacuum permeability)
        self.copper_resistivity = 1.68e-8  # Ω⋅m at 20°C

        # Design constraints
        self.min_air_gap = 1.0  # mm
        self.min_wall_thickness = 2.0  # mm

    def _validate_wire_selection(self):
        """Validate wire selection based on current requirements"""
        # Calculate maximum current density (A/mm²)
        wire_area = math.pi * (self.params.wire_diameter / 2) ** 2
        current_density = self.params.max_current / wire_area

        # Check against typical limits (5-10 A/mm² for air-cooled motors)
        if current_density > 10:
            print(f"Warning: Current density {current_density:.1f} A/mm² exceeds recommended maximum")
            print("Consider using larger wire diameter")

    def _generate_configs(
            self) -> list[UnifiedMotorParameters]:
        """Generate potential motor configurations based on design parameters."""
        configs = []

        # Calculate available space for magnets and coils
        max_radius = self.params.outer_radius - self.min_wall_thickness
        min_radius = max_radius * 0.3  # Common minimum radius ratio
        mean_radius = (max_radius + min_radius) / 2

        # Calculate magnetic field
        b_field = self.calculate_magnetic_field(self.min_air_gap)

        # Parameter ranges to explore
        pole_counts = range(4, 32, 2)  # Common pole counts (even numbers)
        coil_multipliers = [1.25, 1.5, 1.75, 2.0]  # Coil to pole ratios
        min_turns = 5
        max_turns = 100
        # turns_range = range(min_turns, max_turns + 1, 5)

        # Calculate wire area for current density checks
        max_current_density = 7.0  # A/mm² for continuous operation
        peak_current_density = 15.0  # A/mm² absolute maximum
        """
        For 0.65mm wire
            Continuous duty: 4-6 A/mm²
            Intermittent duty: 6-10 A/mm²
            Peak (short duration): 10-20 A/mm²
        """
        wire_area_mm2 = math.pi * (self.params.wire_diameter / 2) ** 2
        max_safe_current = peak_current_density * wire_area_mm2  # max_current_density A/mm² max for air-cooled motors

        # Calculate torque tolerance range
        min_torque = self.params.target_torque * (1 - self.params.tolerance)
        max_torque = self.params.target_torque * (1 + self.params.tolerance)

        for num_poles in pole_counts:
            for multiplier in coil_multipliers:
                num_coils = int(num_poles * multiplier)

                # Skip invalid combinations
                if num_coils % 2 != 0 or num_coils <= num_poles:
                    continue

                # Calculate minimum turns needed for target torque
                min_turns_needed = self._calculate_min_turns(
                    mean_radius, num_poles, num_coils, b_field, min_torque
                )

                if min_turns_needed > max_turns:
                    continue

                # Check configurations with different turns
                start_turns = max(min_turns, min_turns_needed)
                adjusted_turns_range = range(start_turns, max_turns + 1, 5)

                for turns in adjusted_turns_range:
                    # 1. First calculate all parameters
                    resistance = self.calculate_resistance(mean_radius, num_coils, turns)
                    current = min(self.params.voltage / resistance, self.params.max_current)
                    current_density = current / wire_area_mm2
                    torque = self.calculate_torque(
                        mean_radius, num_poles, num_coils, turns, current, b_field
                    )
                    efficiency = self.calculate_efficiency(current, resistance)

                    # 2. Log the configuration's key parameters
                    print(f"Info: Config {num_poles}P/{num_coils}C with {turns} turns:")
                    print(f"  Current Density: {current_density:.1f} A/mm²")
                    print(f"  Current: {current:.2f}A")
                    print(f"  Torque: {torque:.3f} Nm")
                    print(f"  Efficiency: {efficiency:.1f}%")

                    # 3. Check all requirements together
                    is_valid = True
                    validation_messages = []

                    # Check minimum current
                    if current < 0.5:
                        is_valid = False
                        validation_messages.append("Current too low")

                    # Check peak current density limit
                    if current > max_safe_current:
                        is_valid = False
                        validation_messages.append(f"Exceeds peak current density: {current_density:.1f} A/mm²")

                    # Check continuous current density limit
                    if current_density > max_current_density:
                        is_valid = False
                        validation_messages.append(f"Exceeds continuous current density: {current_density:.1f} A/mm²")

                    # Check torque requirements
                    # if not (min_torque <= torque <= max_torque):
                    if torque < min_torque:  # Only check minimum torque requirement
                        is_valid = False
                        # validation_messages.append(
                        #     f"Torque {torque:.3f} Nm outside range {min_torque:.3f}-{max_torque:.3f} Nm")
                        validation_messages.append(f"Torque {torque:.3f} Nm below minimum {min_torque:.3f} Nm")

                    # Check efficiency
                    # if efficiency < 40:
                    #     is_valid = False
                    #     validation_messages.append(f"Efficiency too low: {efficiency:.1f}%")

                    # If configuration passes all checks, add it
                    if is_valid:
                        config = UnifiedMotorParameters(
                            poles=num_poles,
                            coils=num_coils,
                            turns_per_coil=turns,
                            wire_diameter=self.params.wire_diameter,
                            voltage=self.params.voltage,
                            max_current=self.params.max_current,
                            magnet_type=self.params.magnet_type,
                            magnet_width=self.params.magnet_width,
                            magnet_length=self.params.magnet_length,
                            magnet_thickness=self.params.magnet_thickness,
                            magnet_br=self.params.magnet_br,
                            outer_radius=max_radius,
                            inner_radius=min_radius,
                            air_gap=self.min_air_gap,
                            stator_thickness=self.params.stator_thickness,
                            rotor_thickness=self.params.rotor_thickness,
                            target_diameter=self.params.target_diameter,
                            torque=torque,
                            tolerance=self.params.tolerance,
                            target_torque=self.params.target_torque,
                            estimated_torque=torque,
                            efficiency=efficiency,
                            resistance=resistance,
                            current=current
                        )
                        configs.append(config)
                        print("\nFound viable config!")
                        print(f"Poles: {num_poles}, Coils: {num_coils}, Turns: {turns}")
                        print(f"Current: {current:.1f}A (Current Density: {current_density:.1f} A/mm²)")
                        print(f"Torque: {torque:.3f} Nm")
                        print(f"Efficiency: {efficiency:.1f}%")
                        print(f"Resistance: {resistance:.2f} Ω")
                    else:
                        print(f"  Invalid configuration: {', '.join(validation_messages)}")
        if not configs:
            print("\nWarning: No valid configurations found.")
            print(f"Consider one of the following adjustments:")
            print("1. Increase wire diameter (currently {:.2f}mm)".format(self.params.wire_diameter))
            print("2. Increase voltage (currently {:.1f}V)".format(self.params.voltage))
            print("3. Decrease target torque (currently {:.3f}Nm)".format(self.params.target_torque))
            print("4. Adjust tolerance range (currently ±{:.0f}%)".format(self.params.tolerance * 100))

        return configs

    def calculate_magnetic_field(
            self,
            air_gap: float) -> float:
        """Calculate magnetic field using area and flux."""
        # Convert air_gap to meters
        air_gap_m = air_gap * 1e-3

        if self.params.magnet_type == "circle":
            area = math.pi * (self.params.magnet_width / 2000) ** 2  # mm² to m²
        else:  # square or rectangular
            area = (self.params.magnet_width * self.params.magnet_length) * 1e-6  # mm² to m²

        # Convert magnet thickness to meters
        thickness_m = self.params.magnet_thickness * 1e-3

        # Calculate reluctance of air gap
        reluctance_gap = air_gap_m / (self.mu0 * area)

        # Calculate reluctance of magnet
        reluctance_magnet = thickness_m / (self.mu0 * self.params.magnet_br * area)

        # Calculate effective magnetic field
        return self.params.magnet_br / (1 + reluctance_gap / reluctance_magnet)

    @staticmethod
    def calculate_torque(
            mean_radius: float,
            num_poles: int,
            num_coils: int,
            turns_per_coil: int,
            current: float,
            b_field: float) -> float:
        """Calculate theoretical torque for given configuration"""
        # Convert mean_radius to meters
        mean_radius_m = mean_radius * 1e-3

        # Calculate active length for a single coil side
        active_length = mean_radius_m * math.pi / num_poles

        # Each coil has two active sides
        total_force = 2 * b_field * current * active_length * turns_per_coil * num_coils

        # Torque = Force * radius
        return total_force * mean_radius_m

    def calculate_resistance(
            self,
            mean_radius: float,
            num_coils: int,
            turns_per_coil: int) -> float:
        """Calculate total resistance of all coils"""
        # Calculate average length of one turn including end turns
        # Add 20% for end turns
        turn_length = 2.2 * math.pi * mean_radius * 1e-3  # Convert to meters

        # Total wire length
        total_length = turn_length * turns_per_coil * num_coils

        # Calculate resistance using magnet wire resistance per meter
        resistance_per_meter = magnet_wire_resistance(self.params.wire_diameter)

        return resistance_per_meter * total_length
        #  return (self.copper_resistivity * total_length) / (math.pi * (self.params.wire_diameter * 1e-3 / 2) ** 2)

    def calculate_efficiency_doublecountinglosses(
            self,
            current: float,
            resistance: float) -> float:
        """Calculate motor efficiency"""
        # Power losses in copper
        copper_loss = current * current * resistance

        # Mechanical power output (assuming 80% of electrical power converts to mechanical)
        mech_power = self.params.voltage * current * 0.8

        # Total input power
        input_power = self.params.voltage * current

        # Efficiency calculation
        if input_power > 0:
            return (mech_power - copper_loss) / input_power * 100
        return 0.0

    def calculate_efficiency_ignoresotherlosses(
        # def calculate_efficiency(
            self,
            current: float,
            resistance: float) -> float:
        """Calculate motor efficiency"""
        # Power losses in copper
        copper_loss = current * current * resistance

        # Total input power
        input_power = self.params.voltage * current

        # Mechanical power is what remains after losses
        mech_power = input_power - copper_loss

        # Efficiency calculation
        if input_power > 0:
            return (mech_power / input_power) * 100
        return 0.0

    # def calculate_efficiency_new(
    def calculate_efficiency(
            self,
            current: float,
            resistance: float) -> float:
        """Calculate motor efficiency"""
        # Input electrical power
        input_power = self.params.voltage * current

        # Copper losses (I²R)
        copper_loss = current * current * resistance

        # Core/iron losses (simplified approximation)
        # Usually 2-3% of input power for small motors
        core_loss = input_power * 0.03

        # Mechanical losses (simplified approximation)
        # Usually 1-2% of input power for small motors
        mech_loss = input_power * 0.02

        # Total losses
        total_losses = copper_loss + core_loss + mech_loss

        # Output mechanical power
        mech_power = input_power - total_losses

        # Efficiency calculation
        if input_power > 0:
            return (mech_power / input_power) * 100
        return 0.0

    def _calculate_min_turns(
            self,
            mean_radius: float,
            num_poles: int,
            num_coils: int,
            b_field: float,
            min_torque: float) -> int:
        """Calculate minimum turns needed to achieve minimum torque."""
        # Calculate torque per amp-turn
        torque_per_amp_turn = (self.calculate_torque(
            mean_radius, num_poles, num_coils, 1, 1, b_field
        ))

        # Calculate required amp-turns for minimum torque
        required_amp_turns = min_torque / torque_per_amp_turn

        # Estimate minimum turns assuming max current
        min_turns = math.ceil(required_amp_turns / self.params.max_current)

        return min_turns

    def find_viable_configurations(
            self) -> List[UnifiedMotorParameters]:
        """Search for viable motor configurations meeting the specifications."""
        return self._generate_configs()

    def calculate_magnetic_field_with_thickness(
            self,
            air_gap: float) -> float:
        """Calculate magnetic field strength in the air gap."""
        temp_x = self.params.magnet_br * self.params.magnet_thickness
        temp_y = self.mu0 * air_gap / self.params.magnet_br
        temp_z = self.params.magnet_thickness + temp_y
        return temp_x / temp_z

    def calculate_magnetic_field_with_area(
            self,
            air_gap: float) -> float:
        """Calculate magnetic field using area and flux."""
        # Convert air_gap to meters
        air_gap_m = air_gap * 1e-3

        if self.params.magnet_type == "circle":
            area = math.pi * (self.params.magnet_width / 2000) ** 2  # mm² to m²
        elif self.params.magnet_type == "square":
            area = (self.params.magnet_width * self.params.magnet_length) * 1e-6  # mm² to m²
        else:
            raise TypeError("Unsupported magnet specification")

        # Convert magnet thickness to meters
        thickness_m = self.params.magnet_thickness * 1e-3

        # Calculate reluctance of air gap
        reluctance_gap = air_gap_m / (self.mu0 * area)

        # Calculate reluctance of magnet
        reluctance_magnet = thickness_m / (self.mu0 * self.params.magnet_br * area)

        # Calculate effective magnetic field
        return self.params.magnet_br / (1 + reluctance_gap / reluctance_magnet)


def magnet_wire_resistance(
        diameter_mm) -> float:
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


def write_motor_output(
        given_filename: str,
        given_calculator: AxialMotorCalculator) -> None:
    """Generate and write motor configuration results to a file."""
    with open(given_filename, 'w', encoding='utf-8') as f:
        # Calculate magnetic field
        b_field = given_calculator.calculate_magnetic_field(given_calculator.min_air_gap)

        # Calculate radii
        max_radius = given_calculator.params.outer_radius - given_calculator.min_wall_thickness
        min_radius = max_radius * 0.3
        mean_radius = (max_radius + min_radius) / 2

        # Calculate torque range
        tolerance_range = (
            given_calculator.params.target_torque * (1 - given_calculator.params.tolerance),
            given_calculator.params.target_torque * (1 + given_calculator.params.tolerance)
        )

        # Write initial parameters
        f.write(f"Motor Design Analysis Results\n")
        f.write(f"============================\n\n")
        f.write(f"Target Parameters:\n")
        f.write(f"- Torque range: {tolerance_range[0]:.2f} to {tolerance_range[1]:.2f} Nm\n")
        f.write(f"- Magnetic field: {b_field:.3f} Tesla\n")
        f.write(f"- Mean radius: {mean_radius:.1f} mm\n\n")
        f.write(f"Fixed Parameters:\n")
        f.write(f"- Wire diameter: {given_calculator.params.wire_diameter:.2f} mm\n")
        f.write(f"- Voltage: {given_calculator.params.voltage:.1f} V\n")
        f.write(f"- Max current: {given_calculator.params.max_current:.1f} A\n\n")

        # Find viable configurations
        found_viable_configs = given_calculator.find_viable_configurations()

        # Write configuration details
        f.write(f"\nViable Configurations ({len(found_viable_configs)} found):\n")
        f.write(f"=======================================\n")

        for i, config in enumerate(found_viable_configs, 1):
            f.write(f"\nConfiguration {i}:\n")
            f.write(f"- Poles: {config.poles}\n")
            f.write(f"- Coils: {config.coils}\n")
            f.write(f"- Turns per coil: {config.turns_per_coil}\n")
            f.write(f"- Estimated torque: {config.estimated_torque:.3f} Nm\n")
            f.write(f"- Total resistance: {config.resistance:.2f} Ω\n")
            f.write(f"- Operating current: {config.current:.2f} A\n")
            f.write(f"- Efficiency: {config.efficiency:.1f}%\n")
            f.write(f"- Dimensions: {config.outer_radius * 2:.1f}mm diameter, "
                    f"{config.stator_thickness:.1f}mm thick\n")


def find_motor_configurations(
        given_calculator: AxialMotorCalculator,
        given_filename: str = 'motor_output.txt') -> None:
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


def generate_motor_report(
        given_filename: str = 'motor_output.txt') -> None:
    with open(given_filename, 'r') as f:
        output_text = f.read()

    analyzer = MotorAnalyzer('motor_analysis.html')
    analyzer.save_report(output_text)


def create_default_parameters(
) -> UnifiedMotorParameters:
    """Create default parameters for testing"""
    return UnifiedMotorParameters(
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
        target_torque=0.1,  # Nm
        estimated_torque=0.0,
        tolerance=0.2,  # ±20%
        efficiency=0.0,
        resistance=0.0,
        current=0.0,
        coil_width=None,
        coil_height=None,
        total_height=None
    )


if __name__ == "__main__":
    # Create default parameters
    params = create_default_parameters()

    # Create calculator
    calculator = AxialMotorCalculator(params)

    # Generate output file
    filename = "motor_output_unified.txt"
    write_motor_output(filename, calculator)

    print(f"Motor analysis completed and saved to {filename}")
