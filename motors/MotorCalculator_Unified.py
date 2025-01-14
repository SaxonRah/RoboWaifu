from typing import Dict, Any
import math

from MotorParameters_Unified import UnifiedMotorParameters


class UnifiedMotorCalculator:
    """Enhanced calculator for unified motor parameters with advanced calculations."""

    def __init__(self, params: UnifiedMotorParameters):
        """Initialize calculator with UnifiedMotorParameters."""
        self.params = params
        self._validate_parameters()

        # Physical constants
        self.mu0 = 4 * math.pi * 1e-7  # H/m (vacuum permeability)
        self.copper_resistivity = 1.68e-8  # Ω⋅m at 20°C

        # Design constraints
        self.min_wall_thickness = 2.0  # mm
        self.min_air_gap = 0.5  # mm
        self.wire_packing_factor = 0.75  # Typical for hand-wound coils

    def _validate_parameters(self):
        """Validate input parameters and raise appropriate exceptions."""
        is_valid, errors = self.params.validate()
        if not is_valid:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

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

    def generate_unified_parameters(self) -> UnifiedMotorParameters:
        """
        Generate complete set of unified parameters for motor construction.

        Returns:
            UnifiedMotorParameters object with all required parameters
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

        # Create unified parameters
        params = UnifiedMotorParameters(
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
        params.magnet_width = magnet_layout["width"]
        params.magnet_length = magnet_layout["length"]
        params.magnet_thickness = magnet_layout["thickness"]

        # Validate geometry before returning
        self.validate_geometry(params)

        return params

    def generate_printable_parameters(self) -> Dict[str, Any]:
        """
        Generate parameters specifically formatted for the PrintableMotorPartsGenerator.

        Returns:
            Dictionary of parameters for PrintableMotorPartsGenerator
        """
        unified_params = self.generate_unified_parameters()

        return {
            "Wire_Diameter": unified_params.wire_diameter,
            "p": unified_params.poles,
            "c": unified_params.coils,
            "ro": unified_params.outer_radius,
            "ri": unified_params.inner_radius,
            "coil_width": unified_params.coil_width,
            "coil_length": unified_params.coil_height,
            "magnet_width": unified_params.magnet_width,
            "magnet_length": unified_params.magnet_length,
            "magnet_thickness": unified_params.magnet_thickness,
            "magnet_type": unified_params.magnet_type,
            "stator_thickness": unified_params.stator_thickness,
            "rotor_thickness": unified_params.rotor_thickness,
            "air_gap": unified_params.air_gap,
            "shaft_radius": unified_params.inner_radius * 0.8,  # Reasonable default
            "coil_orientation": "axial",  # Default to axial orientation
            "cutaway": False  # Default to full model
        }

    @staticmethod
    def validate_geometry(params: UnifiedMotorParameters) -> None:
        """
        Validate that all geometric parameters are physically feasible.

        Args:
            params: UnifiedMotorParameters to validate

        Returns:
            True if geometry is valid, False otherwise
        """
        # Check radial space
        if params.inner_radius >= params.outer_radius:
            raise ValueError("inner_radius is greater than or equal to outer_radius")

        # Check axial space
        if params.total_height <= 0:
            raise ValueError("total_height is less than Zero")

        # Check coil fit
        circumference = 2 * math.pi * ((params.outer_radius + params.inner_radius) / 2)
        if params.coil_width * params.coils > circumference:
            raise ValueError("Coils will not fit within circumference")

        # Check magnet fit
        if params.magnet_width * params.poles > circumference:
            raise ValueError("Magnets will not fit within circumference")

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

    def optimize_geometry(self) -> UnifiedMotorParameters:
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
                test_params = UnifiedMotorParameters(
                    **{k: v for k, v in vars(self.params).items()}
                )

                # Modify geometric parameters
                test_params.inner_radius = self.params.outer_radius * r_ratio
                test_params.stator_thickness = self.params.outer_radius * 2 * t_ratio

                # Calculate performance
                calculator = UnifiedMotorCalculator(test_params)
                performance = calculator.calculate_performance()

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


def create_default_parameters() -> UnifiedMotorParameters:
    """Create default parameters for testing."""
    return UnifiedMotorParameters(
        poles=12,
        coils=18,
        turns_per_coil=100,
        wire_diameter=0.65,
        voltage=12.0,
        max_current=10.0,
        magnet_type="circle",
        magnet_width=10.0,
        magnet_length=10.0,
        magnet_thickness=3.0,
        magnet_br=1.2,
        outer_radius=35.0,
        inner_radius=15.0,
        air_gap=1.0,
        stator_thickness=10.0,
        rotor_thickness=5.0,
        target_diameter=70.0,
        torque=0.1,
        efficiency=80.0,
        resistance=1.0,
        current=5.0,
        target_torque=0.1,
        tolerance=0.2
    )
