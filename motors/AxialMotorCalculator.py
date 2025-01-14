import math
from typing import List, Dict, Any

from MotorAnalyzer import MotorAnalyzer
from MotorParameters import MotorParameters, create_default_parameters


class AxialMotorCalculator:

    def __init__(
            self,
            given_params: MotorParameters):
        """Initialize calculator with basic required parameters"""
        self.params = given_params
        self._validate_parameters()

        # Electrical
        self.wire_resistance = self.magnet_wire_resistance(self.params.wire_diameter)
        self._validate_wire_selection()

        # Physical constants
        self.mu0 = 4 * math.pi * 1e-7  # H/m (vacuum permeability)
        self.copper_resistivity = 1.68e-8  # Ω⋅m at 20°C

        # Design constraints
        self.min_air_gap = 1.0  # mm
        self.min_wall_thickness = 2.0  # mm
        self.wire_packing_factor = 0.75  # Typical for hand-wound coils

    def _validate_parameters(self):
        """Validate input parameters and raise appropriate exceptions."""
        is_valid, errors = self.params.validate()
        if not is_valid:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

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
            self) -> list[MotorParameters]:
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
                        config = MotorParameters(
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

    def calculate_coil_geometry(self, turns: int, wire_diameter: float,
                                mean_radius: float, num_coils: int) -> tuple[float, float]:
        """
        Calculate coil width and height based on motor geometry constraints.

        Args:
            turns: Number of turns per coil
            wire_diameter: Wire diameter in mm
            mean_radius: Mean radius of coil placement in mm
            num_coils: Number of coils

        Returns:
            Tuple of (coil_width, coil_height) in mm
        """
        # Calculate available arc length for each coil
        circumference = 2 * math.pi * mean_radius
        available_arc = (circumference / num_coils) * 0.8  # 80% of available space

        # Calculate maximum width based on available arc
        max_width = available_arc * 0.9  # Leave some spacing between coils

        # Calculate number of turns that can fit in width
        turns_per_layer = math.floor(max_width / (wire_diameter / self.wire_packing_factor))

        # Calculate required layers
        num_layers = math.ceil(turns / turns_per_layer)

        # Calculate actual dimensions
        coil_width = turns_per_layer * wire_diameter / self.wire_packing_factor
        coil_height = num_layers * wire_diameter / self.wire_packing_factor

        return coil_width, coil_height

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

    def calculate_magnetic_circuit(self) -> Dict[str, float]:
        """Calculate magnetic circuit parameters."""
        # Convert dimensions to meters
        air_gap_m = self.params.air_gap * 1e-3
        magnet_thickness_m = self.params.magnet_thickness * 1e-3

        # Calculate magnet area
        if self.params.magnet_type == "circle":
            magnet_area = math.pi * (self.params.magnet_width / 2000) ** 2  # mm² to m²
        else:  # square
            magnet_area = (self.params.magnet_width * self.params.magnet_length) * 1e-6

        # Calculate reluctance
        reluctance_air = air_gap_m / (self.mu0 * magnet_area)
        reluctance_magnet = magnet_thickness_m / (self.mu0 * self.params.magnet_br * magnet_area)

        # Calculate flux and flux density
        mmf = self.params.magnet_br * magnet_thickness_m / self.mu0
        total_reluctance = reluctance_air + reluctance_magnet
        flux = mmf / total_reluctance
        flux_density = flux / magnet_area

        return {
            "flux": flux,
            "flux_density": flux_density,
            "mmf": mmf,
            "reluctance_total": total_reluctance,
            "magnet_area": magnet_area
        }

    def calculate_performance(self) -> Dict[str, float]:
        """Calculate motor performance parameters."""
        # Get magnetic circuit parameters
        mag_params = self.calculate_magnetic_circuit()
        coil_params = self.calculate_coil_parameters()

        # Calculate mean radius for torque calculation
        mean_radius = (self.params.outer_radius + self.params.inner_radius) / 2 * 1e-3  # Convert to meters

        # Calculate active length for torque production
        active_length = mean_radius * math.pi / self.params.poles

        # Calculate force per coil
        force_per_coil = \
            2 * mag_params["flux_density"] * self.params.current * active_length * self.params.turns_per_coil

        # Calculate total torque
        torque = force_per_coil * mean_radius * self.params.coils

        # Calculate power and losses
        input_power = self.params.voltage * self.params.current
        copper_loss = self.params.current ** 2 * coil_params["resistance"]

        # Estimate iron losses (simplified)
        iron_loss_coefficient = 2.5  # W/kg
        stator_volume = (math.pi * (
                (self.params.outer_radius * 1e-3) ** 2 - (self.params.inner_radius * 1e-3) ** 2)
                         * (self.params.stator_thickness * 1e-3)
                         )
        iron_density = 7650  # kg/m³
        iron_mass = stator_volume * iron_density
        iron_losses = iron_loss_coefficient * iron_mass

        # Calculate efficiency
        total_losses = copper_loss + iron_losses
        output_power = input_power - total_losses
        efficiency = (output_power / input_power) * 100 if input_power > 0 else 0

        return {
            "torque": torque,
            "input_power": input_power,
            "output_power": output_power,
            "copper_loss": copper_loss,
            "iron_loss": iron_losses,
            "total_loss": total_losses,
            "efficiency": efficiency
        }

    def calculate_coil_parameters(self) -> Dict[str, float]:
        """Calculate coil geometric and electrical parameters."""
        # Calculate mean radius for coil placement
        mean_radius = (self.params.outer_radius + self.params.inner_radius) / 2

        # Calculate available arc length per coil
        circumference = 2 * math.pi * mean_radius
        arc_length = (circumference / self.params.coils) * 0.8  # 80% coverage

        # Calculate coil dimensions
        turns_per_layer = math.floor(arc_length / (self.params.wire_diameter / self.wire_packing_factor))
        num_layers = math.ceil(self.params.turns_per_coil / turns_per_layer)

        # Calculate actual coil dimensions
        coil_width = arc_length
        coil_height = num_layers * self.params.wire_diameter / self.wire_packing_factor

        # Calculate wire length and resistance
        mean_turn_length = 2 * (coil_width + coil_height)  # Simplified rectangle
        total_wire_length = mean_turn_length * self.params.turns_per_coil

        # Calculate resistance
        wire_area = math.pi * (self.params.wire_diameter * 1e-3 / 2) ** 2
        resistance = (self.copper_resistivity * total_wire_length * 1e-3) / wire_area

        return {
            "coil_width": coil_width,
            "coil_height": coil_height,
            "wire_length": total_wire_length,
            "resistance": resistance,
            "turns_per_layer": turns_per_layer,
            "num_layers": num_layers
        }

    def optimize_geometry(self) -> MotorParameters:
        """Optimize motor geometry for given performance targets."""
        # Start with current parameters
        best_params = self.params
        best_performance = self.calculate_performance()

        # Define optimization ranges
        radius_ratios = [0.3, 0.35, 0.4, 0.45, 0.5]  # inner/outer radius ratios
        thickness_ratios = [0.2, 0.25, 0.3, 0.35, 0.4]  # thickness/diameter ratios

        for r_ratio in radius_ratios:
            for t_ratio in thickness_ratios:
                # Create test parameters
                test_params = MotorParameters(
                    **{k: v for k, v in vars(self.params).items()}
                )

                # Modify geometric parameters
                test_params.inner_radius = self.params.outer_radius * r_ratio
                test_params.stator_thickness = self.params.outer_radius * 2 * t_ratio

                # Calculate performance
                temp_calculator = AxialMotorCalculator(test_params)
                performance = temp_calculator.calculate_performance()

                # Compare with best performance
                if (performance["efficiency"] > best_performance["efficiency"] and
                        abs(performance["torque"] - self.params.target_torque) <=
                        self.params.target_torque * self.params.tolerance):
                    best_params = test_params
                    best_performance = performance

        return best_params

    def generate_design_report(self) -> Dict[str, Any]:
        """Generate comprehensive design report."""
        magnetic_params = self.calculate_magnetic_circuit()
        coil_params = self.calculate_coil_parameters()
        performance = self.calculate_performance()

        return {
            "magnetic_circuit": magnetic_params,
            "coil_parameters": coil_params,
            "performance": performance,
            "geometry": {
                "outer_diameter": self.params.outer_radius * 2,
                "inner_diameter": self.params.inner_radius * 2,
                "total_height": (self.params.stator_thickness +
                                 2 * self.params.rotor_thickness +
                                 2 * self.params.air_gap),
                "air_gap": self.params.air_gap
            },
            "configuration": {
                "poles": self.params.poles,
                "coils": self.params.coils,
                "turns_per_coil": self.params.turns_per_coil,
                "magnet_type": self.params.magnet_type
            }
        }

    def generate_printable_parameters(self) -> Dict[str, Any]:
        """
        Generate parameters specifically formatted for the PrintableMotorPartsGenerator.

        Returns:
            Dictionary of parameters for PrintableMotorPartsGenerator
        """
        temp_params = self.generate_parameters()

        return {
            "Wire_Diameter": temp_params.wire_diameter,
            "p": temp_params.poles,
            "c": temp_params.coils,
            "ro": temp_params.outer_radius,
            "ri": temp_params.inner_radius,
            "coil_width": temp_params.coil_width,
            "coil_length": temp_params.coil_height,
            "magnet_width": temp_params.magnet_width,
            "magnet_length": temp_params.magnet_length,
            "magnet_thickness": temp_params.magnet_thickness,
            "magnet_type": temp_params.magnet_type,
            "stator_thickness": temp_params.stator_thickness,
            "rotor_thickness": temp_params.rotor_thickness,
            "air_gap": temp_params.air_gap,
            "shaft_radius": temp_params.inner_radius * 0.8,  # Reasonable default
            "coil_orientation": "axial",  # Default to axial orientation
            "cutaway": False  # Default to full model
        }

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
        resistance_per_meter = self.magnet_wire_resistance(self.params.wire_diameter)

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

    @staticmethod
    def calculate_magnet_spacing(num_poles: int, mean_radius: float) -> float:
        """
        Calculate optimal magnet spacing based on pole count and radius.

        Args:
            num_poles: Number of magnetic poles
            mean_radius: Mean radius of magnet placement in mm

        Returns:
            Optimal spacing between magnets in mm
        """
        circumference = 2 * math.pi * mean_radius
        arc_length = circumference / num_poles
        return arc_length * 0.1  # 10% gap between magnets

    def calculate_magnet_dimensions(self, num_poles: int, mean_radius: float,
                                    magnet_br: float, target_flux: float) -> tuple[float, float, float]:
        """
        Calculate optimal magnet dimensions based on electromagnetic requirements.

        Args:
            num_poles: Number of magnetic poles
            mean_radius: Mean radius of magnet placement in mm
            magnet_br: Residual flux density of magnet material in Tesla
            target_flux: Target air gap flux density in Tesla

        Returns:
            Tuple of (width, length, thickness) in mm
        """
        # Calculate available arc length per pole
        circumference = 2 * math.pi * mean_radius
        arc_length = (circumference / num_poles) * 0.8  # 80% coverage

        # Calculate radial length based on mean radius
        radial_length = (self.params.outer_radius - self.params.inner_radius) * 0.4

        # Calculate required thickness based on magnetic circuit
        # Using simplified magnetic circuit model
        required_thickness = (target_flux * self.params.air_gap) / magnet_br

        # Ensure minimum thickness
        magnet_thickness = max(required_thickness, 2.0)  # Minimum 2mm thickness

        return arc_length, radial_length, magnet_thickness

    def optimize_magnet_layout(self) -> Dict[str, float]:
        """New method to optimize magnet layout"""
        mean_radius = (self.params.outer_radius + self.params.inner_radius) / 2

        # Calculate optimal spacing
        spacing = self.calculate_magnet_spacing(self.params.poles, mean_radius)

        # Calculate optimal dimensions based on target flux
        target_flux = 0.8  # Tesla, typical for axial flux motors
        width, length, thickness = self.calculate_magnet_dimensions(
            self.params.poles,
            mean_radius,
            self.params.magnet_br,
            target_flux
        )

        return {
            "spacing": spacing,
            "width": width,
            "length": length,
            "thickness": thickness
        }

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

    def calculate_rotor_thickness(self, magnet_thickness: float) -> float:
        """
        Calculate required rotor thickness based on magnet thickness and mechanical requirements.

        Args:
            magnet_thickness: Thickness of magnets in mm

        Returns:
            Required rotor thickness in mm
        """
        # Minimum thickness is 2mm plus half magnet thickness for structural integrity
        min_thickness = 2.0 + magnet_thickness * 0.5

        # For larger motors, scale with radius
        radius_based = self.params.outer_radius * 0.05  # 5% of outer radius

        return max(min_thickness, radius_based)

    def calculate_total_height(self, stator_thickness: float, rotor_thickness: float,
                               magnet_thickness: float, air_gap: float) -> float:
        """
        Calculate total motor height including all components.

        Returns:
            Total height in mm
        """
        return (stator_thickness +
                2 * rotor_thickness +
                2 * magnet_thickness +
                2 * air_gap +
                2 * self.min_wall_thickness)

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

    def find_viable_configurations(
            self) -> List[MotorParameters]:
        """Search for viable motor configurations meeting the specifications."""
        return self._generate_configs()

    def generate_parameters(self) -> MotorParameters:
        """
        Generate complete set of parameters for motor construction.

        Returns:
            MotorParameters object with all required parameters
        """
        # Extract base parameters
        poles = self.params.poles
        coils = self.params.coils
        turns = self.params.turns_per_coil
        wire_diameter = self.params.wire_diameter

        # Calculate mean radius from outer and inner radius
        mean_radius = (self.params.outer_radius + self.params.inner_radius) / 2

        # Calculate coil geometry
        coil_width, coil_height = self.calculate_coil_geometry(turns, wire_diameter, mean_radius, self.params.coils)

        # Calculate motor height
        total_height = self.calculate_total_height(
            self.params.stator_thickness,
            self.params.rotor_thickness,
            self.params.magnet_thickness,
            self.params.air_gap
        )

        # Optimize magnet layout
        magnet_layout = self.optimize_magnet_layout()

        # Create parameters
        temp_params = MotorParameters(
            poles=poles,
            coils=coils,
            turns_per_coil=turns,
            wire_diameter=wire_diameter,
            magnet_type=self.params.magnet_type,
            magnet_width=self.params.magnet_width,
            magnet_length=self.params.magnet_length,
            magnet_thickness=self.params.magnet_thickness,
            magnet_br=getattr(self.params, 'magnet_br', 1.2),
            outer_radius=self.params.outer_radius,
            inner_radius=self.params.inner_radius,
            air_gap=self.params.air_gap,
            stator_thickness=self.params.stator_thickness,
            rotor_thickness=self.params.rotor_thickness,
            torque=self.params.torque,
            efficiency=self.params.efficiency,
            resistance=self.params.resistance,
            current=self.params.current,
            coil_width=coil_width,
            coil_height=coil_height,
            total_height=total_height,
            max_current=self.params.max_current,
            target_diameter=self.params.target_diameter,
            target_torque=self.params.target_torque,
            tolerance=self.params.tolerance,
            voltage=self.params.voltage,
        )

        # Update parameters with optimized magnet dimensions
        temp_params.magnet_width = magnet_layout["width"]
        temp_params.magnet_length = magnet_layout["length"]
        temp_params.magnet_thickness = magnet_layout["thickness"]

        # Validate geometry before returning
        self.validate_geometry(temp_params)

        return temp_params

    @staticmethod
    def validate_geometry(given_params: MotorParameters) -> None:
        """
        Validate that all geometric parameters are physically feasible.

        Args:
            given_params: MotorParameters to validate

        Returns:
            True if geometry is valid, False otherwise
        """
        # Check radial space
        if given_params.inner_radius >= given_params.outer_radius:
            raise ValueError("inner_radius is greater than or equal to outer_radius")

        # Check axial space
        if given_params.total_height <= 0:
            raise ValueError("total_height is less than Zero")

        # Check coil fit
        circumference = 2 * math.pi * ((given_params.outer_radius + given_params.inner_radius) / 2)
        if given_params.coil_width * given_params.coils > circumference:
            raise ValueError("Coils will not fit within circumference")

        # Check magnet fit
        if given_params.magnet_width * given_params.poles > circumference:
            raise ValueError("Magnets will not fit within circumference")

    @staticmethod
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
            self,
            given_filename: str) -> None:
        """Generate and write motor configuration results to a file."""
        with open(given_filename, 'w', encoding='utf-8') as f:
            # Calculate magnetic field
            b_field = self.calculate_magnetic_field(self.min_air_gap)

            # Calculate radii
            max_radius = self.params.outer_radius - self.min_wall_thickness
            min_radius = max_radius * 0.3
            mean_radius = (max_radius + min_radius) / 2

            # Calculate torque range
            tolerance_range = (
                self.params.target_torque * (1 - self.params.tolerance),
                self.params.target_torque * (1 + self.params.tolerance)
            )

            # Write initial parameters
            f.write(f"Motor Design Analysis Results\n")
            f.write(f"============================\n\n")
            f.write(f"Target Parameters:\n")
            f.write(f"- Torque range: {tolerance_range[0]:.2f} to {tolerance_range[1]:.2f} Nm\n")
            f.write(f"- Magnetic field: {b_field:.3f} Tesla\n")
            f.write(f"- Mean radius: {mean_radius:.1f} mm\n\n")
            f.write(f"Fixed Parameters:\n")
            f.write(f"- Wire diameter: {self.params.wire_diameter:.2f} mm\n")
            f.write(f"- Voltage: {self.params.voltage:.1f} V\n")
            f.write(f"- Max current: {self.params.max_current:.1f} A\n\n")

            # Find viable configurations
            found_viable_configs = self.find_viable_configurations()

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
            self,
            given_filename: str = 'motor_output.txt') -> None:
        motor_configurations = self.find_viable_configurations()

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

        self.write_motor_output(given_filename)

    @staticmethod
    def generate_motor_report(
            given_filename: str = 'motor_output.txt') -> None:
        with open(given_filename, 'r') as f:
            output_text = f.read()

        analyzer = MotorAnalyzer('motor_analysis.html')
        analyzer.save_report(output_text)


if __name__ == "__main__":
    # Create default parameters
    params = create_default_parameters()
    # Name the output file
    filename = "motor_output.txt"
    # Create calculator
    calculator = AxialMotorCalculator(params)
    # Write the motor outputs.
    calculator.write_motor_output(filename)
    print(f"Motor analysis completed and saved to {filename}")
