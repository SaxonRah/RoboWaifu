import math
from typing import List, Dict, Any, Tuple
from MotorParameters import MotorParameters, create_default_parameters
from AxialMotorCalculator import AxialMotorCalculator


class EnhancedMotorCalculator:
    def __init__(self, size_constraints: Dict[str, float], debug: bool = True):
        self.size_constraints = size_constraints
        self.debug = debug
        self.base_params = self._create_base_parameters()

    def _create_base_parameters(self) -> MotorParameters:
        """Create base parameters with debug info."""
        params = create_default_parameters()

        # Stronger magnets
        params.magnet_br = 1.4  # Tesla (N52 NdFeB)
        params.magnet_thickness = 5.0  # mm
        params.magnet_width = 15.0  # mm
        params.magnet_length = 15.0  # mm

        # Higher voltage, lower current approach
        params.voltage = 24.0  # V
        params.wire_diameter = 1.0  # mm
        wire_area = math.pi * (params.wire_diameter / 2) ** 2
        params.max_current = 5.0 * wire_area  # 5 A/mm²

        # More reasonable target torque
        params.target_torque = 0.05  # Reduce target torque to 0.05 Nm
        params.tolerance = 0.3  # 30% tolerance

        if self.debug:
            print("\nBase parameters:")
            print(f"Voltage: {params.voltage}V")
            print(f"Max current: {params.max_current:.2f}A")
            print(f"Wire diameter: {params.wire_diameter}mm")
            print(f"Target torque: {params.target_torque}Nm")
            print(f"Magnet strength: {params.magnet_br}T")

        return params

    def _calculate_magnetic_field(self, params: MotorParameters) -> float:
        """Calculate magnetic field strength with debug info."""
        air_gap = params.air_gap * 1e-3  # mm to meters
        area = math.pi * (params.magnet_width / 2000) ** 2  # mm² to m²
        thickness = params.magnet_thickness * 1e-3  # mm to meters

        # Calculate reluctances
        mu0 = 4 * math.pi * 1e-7  # H/m
        reluctance_gap = air_gap / (mu0 * area)
        reluctance_magnet = thickness / (mu0 * params.magnet_br * area)

        # Calculate field
        b_field = params.magnet_br / (1 + reluctance_gap / reluctance_magnet)

        if self.debug:
            print(f"\nMagnetic field calculation:")
            print(f"Air gap: {air_gap * 1000:.2f}mm")
            print(f"Magnet area: {area * 1e6:.2f}mm²")
            print(f"Magnet thickness: {thickness * 1000:.2f}mm")
            print(f"B field: {b_field:.3f}T")

        return b_field

    def _validate_geometry(self, params: MotorParameters) -> Tuple[bool, str]:
        """Validate geometric constraints with detailed feedback."""
        if params.inner_radius >= params.outer_radius:
            return False, "Inner radius must be less than outer radius"

        if params.air_gap <= 0:
            return False, "Air gap must be positive"

        circumference = 2 * math.pi * ((params.outer_radius + params.inner_radius) / 2)
        coil_width = params.magnet_width * 1.2  # Approximate coil width
        total_coil_width = coil_width * params.coils

        if total_coil_width > circumference:
            return False, f"Coils won't fit (need {total_coil_width:.1f}mm, have {circumference:.1f}mm)"

        magnet_width = params.magnet_width * params.poles
        if magnet_width > circumference:
            return False, f"Magnets won't fit (need {magnet_width:.1f}mm, have {circumference:.1f}mm)"

        return True, "Geometry valid"

    def _estimate_performance(self, params: MotorParameters) -> Dict[str, float]:
        """Calculate motor performance with fixed force calculations."""
        try:
            if self.debug:
                print("\nCalculating performance:")

            # 1. Magnetic Field
            b_field = self._calculate_magnetic_field(params)
            if self.debug:
                print(f"\nMagnetic field calculation:")
                print(f"Air gap: {params.air_gap:.2f}mm")
                print(f"Magnet area: {math.pi * (params.magnet_width / 2) ** 2:.1f}mm²")
                print(f"B field: {b_field:.3f}T")

            # 2. Coil Dimensions and Resistance
            mean_radius = (params.outer_radius + params.inner_radius) / 2000  # mm to m
            wire_area = math.pi * (params.wire_diameter / 2000) ** 2  # mm² to m²

            # Calculate average turn length including end turns
            turn_length = 2 * math.pi * mean_radius * 1.2  # 20% extra for end turns
            total_wire_length = turn_length * params.turns_per_coil * params.coils

            # Calculate resistance (with temperature coefficient)
            resistivity = 1.68e-8 * (1 + 0.004 * (50))  # Copper at 50°C
            resistance = (resistivity * total_wire_length) / wire_area

            if self.debug:
                print(f"\nWire calculations:")
                print(f"Mean radius: {mean_radius * 1000:.1f}mm")
                print(f"Turn length: {turn_length * 1000:.1f}mm")
                print(f"Total wire length: {total_wire_length:.2f}m")
                print(f"Resistance: {resistance:.3f}Ω")

            # 3. Current Calculation
            voltage = params.voltage * 0.95  # Account for voltage drop
            current = min(voltage / resistance, params.max_current)
            current_density = current / wire_area

            if self.debug:
                print(f"\nElectrical calculations:")
                print(f"Current: {current:.2f}A")
                print(f"Current density: {current_density / 1e6:.1f}A/mm²")

            # 4. Force and Torque Calculation
            # Calculate arc length under one pole
            pole_arc = (2 * math.pi * mean_radius) / params.poles

            # Force per turn (one active length)
            # force_per_side = b_field * current * pole_arc
            force_per_side = b_field * current * (pole_arc / 2)  # Ensure half-pole width is used

            # Each turn has two active sides
            force_per_turn = force_per_side * 2

            # Multiply by turns per coil
            force_per_coil = force_per_turn * params.turns_per_coil

            # Total force from all coils
            # num_active_coils = params.coils / 3  # Assume 3-phase, 1/3 coils active
            # num_active_coils = params.coils * 2 / 3  # Use 2/3 active coils
            num_active_coils = max(2, int(params.coils * 2 / 3))  # Ensure at least 2 active coils

            total_force = force_per_coil * num_active_coils

            # Torque = force * radius
            # torque = total_force * mean_radius
            effective_radius = (params.outer_radius + params.inner_radius) / 2 * 1e-3  # Convert to meters
            torque = total_force * effective_radius

            if self.debug:
                print(f"\nForce calculations:")
                print(f"Pole arc length: {pole_arc * 1000:.1f}mm")
                print(f"Force per side: {force_per_side:.3f}N")
                print(f"Force per turn: {force_per_turn:.3f}N")
                print(f"Force per coil: {force_per_coil:.3f}N")
                print(f"Active coils: {num_active_coils:.1f}")
                print(f"Total force: {total_force:.3f}N")
                print(f"Torque: {torque:.3f}Nm")

            # 5. Power and Efficiency
            input_power = voltage * current

            # Losses
            copper_loss = current * current * resistance
            iron_loss = input_power * 0.03  # 3% iron losses
            mech_loss = max(0.5, input_power * 0.02)  # At least 0.5W mechanical losses

            # Ensure losses don't exceed input power
            total_losses = min(input_power * 0.95, copper_loss + iron_loss + mech_loss)
            output_power = input_power - total_losses

            # Calculate efficiency
            efficiency = (output_power / input_power * 100) if input_power > 0 else 0

            if self.debug:
                print(f"\nPower calculations:")
                print(f"Input voltage: {voltage:.1f}V")
                print(f"Input power: {input_power:.1f}W")
                print(f"Copper loss: {copper_loss:.1f}W")
                print(f"Iron loss: {iron_loss:.1f}W")
                print(f"Mechanical loss: {mech_loss:.1f}W")
                print(f"Output power: {output_power:.1f}W")
                print(f"Efficiency: {efficiency:.1f}%")

            return {
                'torque': torque,
                'efficiency': efficiency,
                'current': current,
                'power': output_power,
                'current_density': current_density / 1e6  # A/mm²
            }

        except Exception as e:
            if self.debug:
                print(f"Performance calculation error: {str(e)}")
                import traceback
                traceback.print_exc()
            return {
                'torque': 0,
                'efficiency': 0,
                'current': 0,
                'power': 0,
                'current_density': 0
            }

    def search_configurations(self) -> List[Dict[str, Any]]:
        """Search for valid motor configurations with detailed logging."""
        valid_configs = []
        total_attempts = 0

        # Parameter ranges (reduced for testing)
        pole_counts = range(4, 32, 4)  # Fewer pole counts for initial testing
        coil_multipliers = [1.5, 2.0]  # Reduced multipliers
        wire_diameters = [1.0, 1.5, 2.0]  # mm
        turns_range = range(50, 201, 50)  # Fewer turn variations

        print(f"\nStarting configuration search with:")
        print(f"Pole counts: {list(pole_counts)}")
        print(f"Wire diameters: {wire_diameters}mm")
        print(f"Turns range: {list(turns_range)}")

        for wire_diameter in wire_diameters:
            wire_area = math.pi * (wire_diameter / 2) ** 2
            max_safe_current = 5.0 * wire_area

            outer_radius_range = range(
                int(self.size_constraints.get('min_diameter', 40) / 2),
                int(min(self.size_constraints['max_diameter'] / 2, 100)),
                5
            )

            for outer_radius in outer_radius_range:
                inner_radius = outer_radius * 0.3

                for num_poles in pole_counts:
                    for multiplier in coil_multipliers:
                        num_coils = int(num_poles * multiplier)
                        if num_coils % 2 != 0:
                            continue

                        for turns in turns_range:
                            total_attempts += 1

                            params = MotorParameters(
                                poles=num_poles,
                                coils=num_coils,
                                turns_per_coil=turns,
                                wire_diameter=wire_diameter,
                                voltage=24.0,
                                max_current=max_safe_current,
                                magnet_type="circle",
                                magnet_width=15.0,
                                magnet_length=15.0,
                                magnet_thickness=5.0,
                                magnet_br=1.4,
                                outer_radius=outer_radius,
                                inner_radius=inner_radius,
                                air_gap=1.0,
                                stator_thickness=15.0,
                                rotor_thickness=5.0,
                                target_diameter=outer_radius * 2,
                                torque=0,
                                target_torque=0.05,  # Reduced target
                                tolerance=0.3,
                                efficiency=0.0,
                                resistance=0.0,
                                current=0.0
                            )

                            if self.debug:
                                print(f"\nTesting {num_poles}P/{num_coils}C, "
                                      f"{turns} turns, {wire_diameter}mm wire")

                            performance = self._estimate_performance(params)

                            # Check if configuration meets requirements
                            min_torque = params.target_torque * (1 - params.tolerance)
                            if (performance['torque'] >= min_torque and
                                    performance['efficiency'] > 60):
                                config = {
                                    'parameters': params,
                                    'performance': performance
                                }
                                valid_configs.append(config)
                                print(f"\nFound valid configuration!")
                                print(f"Torque: {performance['torque']:.3f}Nm")
                                print(f"Efficiency: {performance['efficiency']:.1f}%")

        print(f"\nSearch completed:")
        print(f"Configurations attempted: {total_attempts}")
        print(f"Valid configurations found: {len(valid_configs)}")

        return valid_configs


def main():
    constraints = {
        'max_diameter': 100,
        'max_thickness': 50,
        'min_diameter': 40,
        'min_thickness': 10
    }

    calculator = EnhancedMotorCalculator(constraints, debug=True)
    configs = calculator.search_configurations()

    if not configs:
        print("\nNo valid configurations found.")
        print("Try adjusting:")
        print("1. Target torque (currently 0.05 Nm)")
        print("2. Size constraints")
        print("3. Wire diameter range")
        return

    # Process valid configurations...
    print("\nValid configurations found!")


if __name__ == "__main__":
    main()
