import math
from dataclasses import dataclass
from typing import List, Tuple
import os
import itertools

# Constants
# COIL_RADIUS_RATIO = 0.4
COIL_RADIUS_RATIO = 0.6  # Increased from 0.4 to allow more space

# MAGNET_RADIUS_RATIO = 0.7
MAGNET_RADIUS_RATIO = 0.8  # Increased from 0.7

MAGNET_WIDTH_RATIO = 0.8


@dataclass
class WireSpecs:
    gauge: int
    diameter: float  # mm
    current_capacity: float  # A
    resistance_per_meter: float  # Ω/m
    parallel_strands: int = 1  # Number of parallel strands


@dataclass
class CoilSpecs:
    turns: int
    wire: WireSpecs
    inner_diameter: float  # mm
    outer_diameter: float  # mm
    height: float  # mm
    resistance: float  # Ω
    parallel_strands: int  # Number of parallel strands


@dataclass
class MotorConfiguration:
    num_magnets: int
    num_coils: int
    magnet_thickness: float
    coil_thickness: float
    air_gap: float
    stator_thickness: float
    rotor_thickness: float
    coil_specs: CoilSpecs = None


@dataclass
class MotorSpecs:
    diameter: float  # mm
    torque: float  # Nm
    voltage: float = 48  # V
    speed: float = 1000  # RPM
    shaft_diameter: float = 20  # mm
    target_thickness: float = 50  # mm overall thickness


class WireCalculator:
    # Common wire gauges with diameter (mm) and current capacity (A)
    WIRE_GAUGES = {
        18: WireSpecs(18, 1.024, 16, 0.0206),
        20: WireSpecs(20, 0.812, 11, 0.0327),
        22: WireSpecs(22, 0.644, 7, 0.0521),
        24: WireSpecs(24, 0.511, 3.5, 0.0827),
        26: WireSpecs(26, 0.405, 2.2, 0.1317)
    }

    @staticmethod
    def select_wire(current: float, max_strands: int = 4) -> WireSpecs:
        """Select appropriate wire gauge and number of parallel strands based on current requirements."""
        for wire in sorted(WireCalculator.WIRE_GAUGES.values(), key=lambda x: x.gauge):

            # Calculate required parallel strands (with 20% safety margin)
            required_strands = math.ceil((current * 1.2) / wire.current_capacity)

            # Limit to reasonable number of parallel strands
            if required_strands <= max_strands:  # Maximum 4 parallel strands
                wire_copy = WireSpecs(
                    wire.gauge,
                    wire.diameter,
                    wire.current_capacity * required_strands,
                    wire.resistance_per_meter / required_strands,  # Parallel resistance
                    required_strands
                )
                return wire_copy

        raise ValueError(
            f"Current requirement of {current}A too high even with parallel windings."
            f"Consider different motor configuration.")

    @staticmethod
    def calculate_coil_specs(
            voltage: float,
            current: float,
            inner_d: float,
            outer_d: float,
            target_height: float,  # Add target height parameter
            rpm_per_volt: int = 20) -> CoilSpecs:
        """Calculate coil specifications based on voltage and current requirements."""
        wire = WireCalculator.select_wire(current)

        # Calculate required number of turns based on voltage
        required_turns = max(int(voltage * rpm_per_volt / 1000), 10)  # Minimum 10 turns

        # Adjust wire space requirements for parallel strands
        effective_wire_diameter = wire.diameter * math.sqrt(wire.parallel_strands)

        # Calculate turns based on target height
        turns_per_layer = math.floor((outer_d - inner_d) / (effective_wire_diameter * 2))
        target_layers = math.floor(target_height / (effective_wire_diameter * 2))

        # Adjust number of turns to meet both voltage and height requirements
        required_turns = max(required_turns, turns_per_layer * target_layers)
        actual_layers = math.ceil(required_turns / turns_per_layer)

        # Calculate actual height based on layers
        height = actual_layers * effective_wire_diameter * 2

        # Calculate total wire length and resistance
        avg_turn_length = math.pi * ((outer_d + inner_d) / 2)
        total_wire_length = avg_turn_length * required_turns / 1000  # in meters
        resistance = wire.resistance_per_meter * total_wire_length

        return CoilSpecs(
            turns=required_turns,
            wire=wire,
            inner_diameter=inner_d,
            outer_diameter=outer_d,
            height=height,
            resistance=resistance,
            parallel_strands=wire.parallel_strands
        )


class CoilLayoutCalculator:
    @staticmethod
    def calculate_coil_positions(num_coils: int, motor_diameter: float, coil_outer_d: float) -> List[
        Tuple[float, float]]:
        """Calculate non-overlapping coil positions."""
        positions = []
        motor_radius = motor_diameter / 2

        # Calculate placement radius - start from the inside and work outward
        base_radius = motor_radius * COIL_RADIUS_RATIO

        # Calculate minimum spacing between coil centers
        min_spacing = coil_outer_d * 1.1  # Add 10% margin

        # Calculate minimum radius needed for coils
        min_radius = (min_spacing * num_coils) / (2 * math.pi)

        # Use the larger of the calculated values
        placement_radius = max(base_radius, min_radius)

        # Verify the coils will fit within the motor diameter
        if placement_radius + coil_outer_d / 2 > motor_radius:
            # Try to reduce the coil spacing if possible
            if min_radius + coil_outer_d / 2 > motor_radius:
                raise ValueError(
                    f"Coils too large to fit {num_coils} positions. "
                    f"Max coil diameter should be {(motor_radius * 2 / num_coils) * 0.9:.1f}mm"
                )
            placement_radius = motor_radius - coil_outer_d / 2

        # Calculate positions
        angle_step = 360 / num_coils
        for i in range(num_coils):
            angle = math.radians(i * angle_step)
            x = placement_radius * math.cos(angle)
            y = placement_radius * math.sin(angle)
            positions.append((x, y))

        # Verify no overlaps
        for (x1, y1), (x2, y2) in itertools.combinations(positions, 2):
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < coil_outer_d:
                raise ValueError(
                    f"Coil overlap detected. Distance between centers: {distance:.1f}mm, "
                    f"Coil diameter: {coil_outer_d:.1f}mm"
                )

        return positions


class AxialFluxCalculator:
    def __init__(self, specs: MotorSpecs):
        self.specs = specs

    def calculate_configurations(
            self,
            stator_backing_mm: float = 5,
            rotor_backing_mm: float = 5) -> List[MotorConfiguration]:
        """Calculate various valid motor configurations based on specs."""
        configurations = []

        # Calculate basic parameters
        speed_rad_s = self.specs.speed * 2 * math.pi / 60
        power = self.specs.torque * speed_rad_s
        current = power / self.specs.voltage

        # Scale backing thickness with motor size and target thickness
        thickness_scale = self.specs.target_thickness / 50  # Reference thickness of 50mm
        diameter_scale = self.specs.diameter / 300  # Reference diameter of 300mm
        overall_scale = (thickness_scale + diameter_scale) / 2

        scaled_rotor_backing = min(rotor_backing_mm * overall_scale, 40)  # Increased cap to 40mm
        scaled_stator_backing = min(stator_backing_mm * overall_scale, 30)  # Increased cap to 30mm

        target_stator_thickness = (
                self.specs.target_thickness -
                (2 * scaled_rotor_backing) -
                4  # 2mm air gap on each side for larger motor
        )

        # Try different numbers of pole pairs
        for pole_pairs in range(4, 17, 2):  # Increased range for larger motor
            num_magnets = pole_pairs * 2
            num_coils = num_magnets * 3 // 4

            # Adjusted scaling factors for larger motors
            motor_scale_factor = (self.specs.diameter / 300) * (self.specs.target_thickness / 50)
            size_factor = (1 - (num_coils / 32)) * motor_scale_factor  # Adjusted coil count scaling

            # Scale dimensions with motor size
            inner_diameter = self.specs.diameter * 0.15 * size_factor
            outer_diameter = self.specs.diameter * 0.25 * size_factor

            try:
                # Calculate magnet thickness based on power requirements and scale with size
                base_magnet_thickness = self.calculate_magnet_thickness(power, num_magnets)
                scaled_magnet_thickness = base_magnet_thickness * overall_scale

                # Verify coil positions
                CoilLayoutCalculator.calculate_coil_positions(
                    num_coils,
                    self.specs.diameter,
                    outer_diameter
                )

                # Calculate coil specs with scaled target height
                target_coil_height = min(
                    target_stator_thickness - scaled_stator_backing,
                    100 * overall_scale  # Increased cap for larger motors
                )

                coil_specs = WireCalculator.calculate_coil_specs(
                    self.specs.voltage / num_coils,
                    current,
                    inner_diameter,
                    outer_diameter,
                    target_coil_height
                )

                config = MotorConfiguration(
                    num_magnets=num_magnets,
                    num_coils=num_coils,
                    magnet_thickness=scaled_magnet_thickness,
                    coil_thickness=coil_specs.height,
                    air_gap=2.0,  # Increased air gap for larger motor
                    stator_thickness=coil_specs.height + scaled_stator_backing,
                    rotor_thickness=scaled_magnet_thickness + scaled_rotor_backing,
                    coil_specs=coil_specs
                )

                total_thickness = (
                        config.stator_thickness +
                        2 * config.rotor_thickness +
                        2 * config.air_gap
                )

                # Relaxed thickness tolerance for larger motors
                thickness_tolerance = max(self.specs.target_thickness * 0.25, 50)  # 25% or 50mm, whichever is larger
                if abs(total_thickness - self.specs.target_thickness) <= thickness_tolerance:
                    configurations.append(config)

            except ValueError as e:
                print(f"Skipping configuration with {num_coils} coils: {str(e)}")
                continue

        if not configurations:
            raise ValueError(
                f"No valid configurations found for {self.specs.target_thickness}mm thickness. "
                "Consider reducing target thickness or increasing motor diameter."
            )

        # Sort by closest to target thickness
        configurations.sort(key=lambda x: abs(
            (x.rotor_thickness * 2 + x.stator_thickness + x.air_gap * 2) -
            self.specs.target_thickness
        ))

        return configurations

    def calculate_magnet_thickness(self, power: float, num_magnets: int, base_thickness_mm: float = 3) -> float:
        """Calculate required magnet thickness based on power and number of magnets."""
        # Scale thickness with square root of power and motor size
        power_factor = math.sqrt(power / 1000)
        size_factor = math.sqrt(self.specs.diameter / 300)
        thickness = base_thickness_mm * power_factor * size_factor

        # Adjusted bounds for larger motors
        min_thickness = 2 * size_factor
        max_thickness = 30 * size_factor  # Increased maximum thickness

        return min(max(thickness, min_thickness), max_thickness)


class MotorDesigner:
    def __init__(self, specs: MotorSpecs):
        self.specs = specs
        self.calculator = AxialFluxCalculator(specs)

    def generate_design(self, output_dir: str = "motor_design"):
        """Generate motor design based on specifications"""
        configurations = self.calculator.calculate_configurations()

        # Filter configurations based on target thickness
        valid_configs = []
        for config in configurations:
            total_thickness = (
                    config.rotor_thickness +  # Rotor plate
                    config.stator_thickness +  # Stator plate
                    config.air_gap  # Air gap
            )

            if abs(total_thickness - self.specs.target_thickness) <= 5:  # 5mm tolerance
                valid_configs.append(config)

        if not valid_configs and len(configurations) > 0:
            print("Warning: No configurations meet the target thickness. Using closest match.")
            # Find closest match
            configurations.sort(key=lambda x: abs(
                (x.rotor_thickness + x.stator_thickness + x.air_gap) -
                self.specs.target_thickness
            ))
            valid_configs = [configurations[0]]

        # Use the best configuration
        best_config = valid_configs[0]

        # Print design details
        self._print_design_details(best_config)

        # Generate OpenSCAD files
        generator = OpenSCADGenerator(self.specs, best_config)
        generator.generate_files(output_dir)

        return best_config

    def _print_design_details(self, config: MotorConfiguration):
        """Print detailed specifications of the motor design"""
        total_thickness = config.rotor_thickness + config.stator_thickness + config.air_gap

        print("\nMotor Design Specifications:")
        print(f"Overall Dimensions:")
        print(f"- Diameter: {self.specs.diameter}mm")
        print(f"- Total Thickness: {total_thickness:.1f}mm")
        print(f"- Shaft Diameter: {self.specs.shaft_diameter}mm")

        print(f"\nPerformance:")
        print(f"- Torque: {self.specs.torque}Nm")
        print(f"- Speed: {self.specs.speed}RPM")
        print(f"- Voltage: {self.specs.voltage}V")

        print(f"\nConfiguration:")
        print(f"- Number of Magnets: {config.num_magnets}")
        print(f"- Number of Coils: {config.num_coils}")
        print(f"- Magnet Thickness: {config.magnet_thickness:.1f}mm")
        print(f"- Air Gap: {config.air_gap}mm")

        print(f"\nCoil Specifications:")
        print(f"- Wire Gauge: AWG {config.coil_specs.wire.gauge}")
        print(f"- Number of Turns: {config.coil_specs.turns}")
        print(f"- Parallel Strands: {config.coil_specs.parallel_strands}")
        print(f"- Coil Height: {config.coil_specs.height:.1f}mm")
        print(f"- Resistance per Coil: {config.coil_specs.resistance:.2f}Ω")


class OpenSCADGenerator:
    def __init__(self, specs: MotorSpecs, config: MotorConfiguration):
        self.specs = specs
        self.config = config

    def generate_files(self, output_dir: str):
        """Generate OpenSCAD files for all motor components."""
        os.makedirs(output_dir, exist_ok=True)

        self.generate_rotor(f"{output_dir}/rotor.scad")
        self.generate_stator(f"{output_dir}/stator.scad")
        self.generate_housing(f"{output_dir}/housing.scad")
        self.generate_assembly(f"{output_dir}/assembly.scad")

    def generate_rotor(self, filename: str):
        """Generate OpenSCAD file for rotor with magnets."""
        scad_code = f"""
// Rotor plate with magnets
$fn = 100;

module rotor() {{
    difference() {{
        // Main disk
        cylinder(h={self.config.rotor_thickness}, d={self.specs.diameter});

        // Center hole
        translate([0, 0, -1])
            cylinder(h={self.config.rotor_thickness + 2}, d={self.specs.shaft_diameter});
    }}

    // Magnets
    {self._generate_magnets()}
}}

rotor();
"""
        with open(filename, 'w') as f:
            f.write(scad_code)

    def generate_stator(self, filename: str):
        """Generate OpenSCAD file for stator with coils."""
        scad_code = f"""
// Stator with coils
$fn = 100;

module coil() {{
    // Coil assembly
    color("Silver") // Coil former color
    difference() {{
        cylinder(h={self.config.coil_specs.height}, 
                d={self.config.coil_specs.outer_diameter});

        translate([0, 0, -1])
            cylinder(h={self.config.coil_specs.height + 2}, 
                    d={self.config.coil_specs.inner_diameter});
    }}

    // Wire winding visualization
    color("PeachPuff")
    translate([0, 0, 0.5])  // Slight offset for visibility
    difference() {{
        cylinder(h={self.config.coil_specs.height - 1}, 
                d={self.config.coil_specs.outer_diameter - 1});

        translate([0, 0, -1])
            cylinder(h={self.config.coil_specs.height + 2}, 
                    d={self.config.coil_specs.inner_diameter + 1});
    }}
}}

module stator() {{
    // Base stator disk
    difference() {{
        cylinder(h={self.config.stator_thickness}, d={self.specs.diameter});

        // Center hole
        translate([0, 0, -1])
            cylinder(h={self.config.stator_thickness + 2}, d={self.specs.shaft_diameter});
    }}

    // Coils
    {self._generate_coils()}
}}

stator();
"""
        with open(filename, 'w') as f:
            f.write(scad_code)

    def _generate_magnets(self) -> str:
        """Generate OpenSCAD code for magnet placement."""
        magnet_angle = 360 / self.config.num_magnets

        radius = (
                self.specs.diameter / 2 * MAGNET_RADIUS_RATIO
        )  # Place at 70% of radius

        magnet_width = (
                math.pi * (radius * 2) / self.config.num_magnets * MAGNET_WIDTH_RATIO
        )  # 80% of available arc length

        magnets = []
        for i in range(self.config.num_magnets):
            angle = i * magnet_angle
            magnets.append(f"""
    // Magnet {i + 1}
    translate([{radius} * cos({angle}), {radius} * sin({angle}), 0])
        rotate([0, 0, {angle}])
        cylinder(h={self.config.magnet_thickness}, 
                d={magnet_width},
                $fn=30);""")

        return "\n".join(magnets)

    def _generate_coils(self) -> str:
        """Generate OpenSCAD code for coil placement."""
        positions = CoilLayoutCalculator.calculate_coil_positions(
            self.config.num_coils,
            self.specs.diameter,
            self.config.coil_specs.outer_diameter
        )

        coils = []
        for i, (x, y) in enumerate(positions):
            coils.append(f"""
    // Coil {i + 1}
    translate([{x}, {y}, {self.config.stator_thickness - self.config.coil_specs.height}])
        coil();""")

        return "\n".join(coils)

    def generate_housing(self, filename: str):
        """Generate OpenSCAD file for motor housing."""
        # Housing implementation here
        scad_code = f"""
        // Housing (placeholder)
        $fn = 100;
        module housing() {{
            cylinder(h={self.specs.target_thickness}, d={self.specs.diameter + 10});
        }}
        housing();
        """

        with open(filename, 'w') as f:
            f.write(scad_code)

    def generate_assembly(self, filename: str):
        """Generate OpenSCAD file showing complete assembly."""
        scad_code = f"""
// Complete motor assembly
$fn = 100;

// Import individual components
use <rotor.scad>
use <stator.scad>
use <housing.scad>

// Assembly
module assembly() {{
    // Housing
    housing();
    
    // Bottom rotor
    rotor();

    // Stator (with air gap)
    translate([0, 0, {self.config.rotor_thickness + self.config.air_gap}])
        stator();

    // Top rotor
    translate([0, 0, {self.config.rotor_thickness + self.config.air_gap + self.config.stator_thickness + self.config.air_gap}])
        rotor();
}}

assembly();
"""
        with open(filename, 'w') as f:
            f.write(scad_code)


def main():
    # Example usage with all parameters
    # specs = MotorSpecs(
    #     diameter=100,  # 100mm diameter
    #     torque=5,  # 5Nm
    #     voltage=12,  # 12V
    #     speed=1000,  # 1000 RPM
    #     shaft_diameter=5,  # 5mm shaft
    #     target_thickness=30  # 30mm total thickness
    # )

    specs = MotorSpecs(
        diameter=200,  # 200mm diameter
        torque=10,  # 10Nm
        voltage=48,  # 48V
        speed=1000,  # 1000 RPM
        shaft_diameter=10,  # 10mm shaft
        target_thickness=125  # 125mm total thickness
    )

    # Create designer and generate design
    designer = MotorDesigner(specs)
    designer.generate_design("motor_output")


if __name__ == "__main__":
    main()
