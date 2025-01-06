import math
from dataclasses import dataclass
from typing import List, Tuple
import os
import itertools

# Constants
COIL_RADIUS_RATIO = 0.4
MAGNET_RADIUS_RATIO = 0.7
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
            rpm_per_volt: int = 20) -> CoilSpecs:
        """Calculate coil specifications based on voltage and current requirements."""
        wire = WireCalculator.select_wire(current)

        # Calculate required number of turns based on voltage
        # Adjusted voltage constant for more realistic turn count
        required_turns = max(int(voltage * rpm_per_volt / 1000), 10)  # Minimum 10 turns

        # Adjust wire space requirements for parallel strands
        effective_wire_diameter = wire.diameter * math.sqrt(wire.parallel_strands)

        # Calculate coil dimensions
        turns_per_layer = math.floor((outer_d - inner_d) / (effective_wire_diameter * 2))
        num_layers = math.ceil(required_turns / turns_per_layer)

        if num_layers * effective_wire_diameter * 2 < 5:
            # Adjust number of turns to ensure minimum height
            required_turns = math.ceil(5 / (effective_wire_diameter * 2)) * turns_per_layer
            num_layers = math.ceil(required_turns / turns_per_layer)
            print(f"Adjusted turns to meet minimum height: {required_turns}")

        # TODO: Find out why height is always selecting 5mm via max()
        # height = max(num_layers * effective_wire_diameter * 2, 5)  # Minimum 5mm height
        height = num_layers * effective_wire_diameter * 2

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

        # Calculate placement radius to ensure coils fit
        min_radius = (coil_outer_d * num_coils) / (2 * math.pi)
        # Place at either calculated minimum radius or 40% of motor radius, whichever is larger
        mean_radius = max(min_radius * 1.1, motor_diameter * COIL_RADIUS_RATIO)

        if mean_radius + coil_outer_d / 2 > motor_diameter / 2:
            raise ValueError("Coil positions exceed motor diameter. Adjust coil size or number of coils.")

        angle_step = 360 / num_coils

        for i in range(num_coils):
            angle = i * angle_step
            x = mean_radius * math.cos(math.radians(angle))
            y = mean_radius * math.sin(math.radians(angle))
            positions.append((x, y))

        if any(
                math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < coil_outer_d
                for (x1, y1), (x2, y2) in itertools.combinations(positions, 2)
        ):
            raise ValueError("Calculated coil positions overlap. Adjust coil size or number of coils.")

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
        circumference = math.pi * self.specs.diameter
        speed_rad_s = self.specs.speed * 2 * math.pi / 60
        power = self.specs.torque * speed_rad_s
        current = power / self.specs.voltage

        # Try different numbers of pole pairs
        for pole_pairs in range(4, 13, 2):  # 8 to 24 poles
            num_magnets = pole_pairs * 2
            num_coils = num_magnets * 3 // 4  # Typical ratio for axial flux
            # NOTE: num_coils = num_magnets * 3 // 4 are based on typical ratios but may not always apply.
            # Allow these ratios to be configurable or validate them against real-world constraints.

            # Calculate magnet thickness based on power requirements
            magnet_thickness = self.calculate_magnet_thickness(power, num_magnets)

            # Calculate coil dimensions
            inner_diameter = self.specs.diameter * 0.3  # 30% of motor diameter
            outer_diameter = self.specs.diameter * 0.45  # 45% of motor diameter (reduced from 70%)

            try:
                coil_specs = WireCalculator.calculate_coil_specs(
                    self.specs.voltage / num_coils,  # Voltage per coil
                    current,
                    inner_diameter,
                    outer_diameter
                )

                config = MotorConfiguration(
                    num_magnets=num_magnets,
                    num_coils=num_coils,
                    magnet_thickness=magnet_thickness,
                    coil_thickness=coil_specs.height,
                    air_gap=1.0,  # mm, typical minimum air gap
                    stator_thickness=coil_specs.height + stator_backing_mm,
                    rotor_thickness=magnet_thickness + rotor_backing_mm,
                    coil_specs=coil_specs
                )

                configurations.append(config)
            except ValueError as e:
                continue  # Skip invalid configurations

        return configurations

    def calculate_magnet_thickness(self, power: float, num_magnets: int, base_thickness_mm: float = 3) -> float:
        """Calculate required magnet thickness based on power and number of magnets."""
        power_factor = math.sqrt(power / 1000)  # Scale with square root of power
        return base_thickness_mm * power_factor


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
    color("Copper")
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
        diameter=300,  # 300mm diameter
        torque=10,  # 10Nm
        voltage=48,  # 48V
        speed=1000,  # 1000 RPM
        shaft_diameter=20,  # 20mm shaft
        target_thickness=500  # 100mm total thickness
    )

    # Create designer and generate design
    designer = MotorDesigner(specs)
    designer.generate_design("motor_output")


if __name__ == "__main__":
    main()
