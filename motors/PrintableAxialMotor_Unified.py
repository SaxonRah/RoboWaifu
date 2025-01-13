import math
from pathlib import Path
from typing import Dict, Any
from UnifiedMotorParameters import UnifiedMotorParameters


class PrintableMotorPartsGenerator:
    def __init__(self, params: UnifiedMotorParameters, print_settings: Dict[str, Any] = None):
        """Initialize the generator with unified parameters and print settings."""
        self.params = params

        # Default 3D printing parameters
        self.print_settings = {
            'wall_thickness': 2.0,  # Wall thickness for structural integrity (mm)
            'tolerance': 0.2,  # Printing tolerance for fits (mm)
            'layer_height': 0.2,  # Standard 3D printing layer height (mm)
            'segments': 100,  # Segments for smooth circles
            'coil_orientation': 'axial',  # 'axial' or 'radial'
            'cutaway': False  # Generate cutaway view
        }

        # Update with user-provided settings if any
        if print_settings:
            self.print_settings.update(print_settings)

    def _to_printable_params(self) -> Dict[str, Any]:
        """Convert UnifiedMotorParameters to printable parameters."""
        return {
            'Wire_Diameter': self.params.wire_diameter,
            'p': self.params.poles,
            'c': self.params.coils,
            'ro': self.params.outer_radius,
            'ri': self.params.inner_radius,
            'coil_width': self.params.coil_width,
            'coil_height': self.params.coil_height or 15.0,  # Default if not set
            'magnet_width': self.params.magnet_width,
            'magnet_length': self.params.magnet_length,
            'magnet_thickness': self.params.magnet_thickness,
            'magnet_type': self.params.magnet_type,
            'stator_thickness': self.params.stator_thickness,
            'rotor_thickness': self.params.rotor_thickness,
            'air_gap': self.params.air_gap,
            'shaft_radius': self.params.inner_radius * 0.8  # Common ratio for shaft
        }

    def generate_coil_spool(self) -> str:
        """Generate OpenSCAD code for a printable coil spool with simple removable cap."""
        p = self._to_printable_params()

        spool_height = p['coil_width'] + 2 * self.print_settings['wall_thickness']
        inner_radius = p['Wire_Diameter'] * 5  # Core radius for winding
        flange_radius = inner_radius * 2  # Outer flange to hold wire
        cap_height = self.print_settings['wall_thickness']

        return f"""
// Main Coil Spool Body
module spool_body() {{
    difference() {{
        union() {{
            // Center core
            cylinder(h={spool_height}, r={inner_radius}, 
                    center=false, $fn={self.print_settings['segments']});

            // Bottom flange only
            cylinder(h={self.print_settings['wall_thickness']}, r={flange_radius}, 
                    center=false, $fn={self.print_settings['segments']});
        }}

        // Center hole for mounting
        cylinder(h={spool_height + 1}, r={inner_radius / 2}, 
                center=false, $fn={self.print_settings['segments']});
    }}
}}

// Simple Cap (matching core diameter)
module spool_cap() {{
    difference() {{
        cylinder(h={cap_height}, r={flange_radius}, 
                center=false, $fn={self.print_settings['segments']});

        // Center hole matching core
        cylinder(h={cap_height + 1}, r={inner_radius}, 
                center=false, $fn={self.print_settings['segments']});
    }}
}}

// Define copper coil appearance for assembly
module copper_coil() {{
    color("Orange", 0.8) {{
        spool_body();
        translate([0, 0, {spool_height}])
        spool_cap();
    }}
}}

// For individual part export
if ($preview) {{
    // Preview assembled
    copper_coil();
}} else {{
    // Export separate parts
    spool_body();
    translate([{flange_radius * 2.5}, 0, 0])
    spool_cap();
}}"""

    def generate_stator_plate(self) -> str:
        """Generate OpenSCAD code for the stator plate with coil slots."""
        p = self._to_printable_params()

        # Calculate angular spacing
        angle_per_coil = 360 / p['c']

        # Calculate slot dimensions
        slot_width = p['coil_width'] + self.print_settings['tolerance']
        slot_depth = p['coil_height'] + self.print_settings['tolerance']
        plate_thickness = p['stator_thickness']

        # Generate coil slot pattern
        coil_slots = self._generate_coil_slots(p['c'], angle_per_coil, slot_width, slot_depth, plate_thickness)

        return f"""
// Printable Stator Plate
module stator() {{
    difference() {{
        // Main plate
        cylinder(h={plate_thickness}, r={p['ro']}, 
                center=false, $fn={self.print_settings['segments']});

        // Center hole
        translate([0, 0, -1])
        cylinder(h={plate_thickness + 2}, r={p['ri']}, 
                center=false, $fn={self.print_settings['segments']});

        // Coil slots
        union() {{
            {coil_slots}
        }}
    }}
}}

stator();"""

    def generate_rotor_plate(self) -> str:
        """Generate OpenSCAD code for the rotor plate with magnet recesses."""
        p = self._to_printable_params()

        # Calculate angular spacing and dimensions
        angle_per_pole = 360 / p['p']
        magnet_width = p['magnet_width'] + self.print_settings['tolerance']
        magnet_depth = p['magnet_thickness'] + self.print_settings['tolerance']
        rotor_thickness = p['rotor_thickness'] + magnet_depth

        # Generate magnet recess pattern
        magnet_recesses = self._generate_magnet_recesses(
            rotor_thickness, p['p'], angle_per_pole, p['magnet_type'],
            magnet_width, p['magnet_length'], magnet_depth)

        return f"""
// Printable Rotor Plate
module rotor() {{
    difference() {{
        // Main plate
        cylinder(h={rotor_thickness}, r={p['ro']}, 
                center=false, $fn={self.print_settings['segments']});

        // Center hole
        translate([0, 0, -1])
        cylinder(h={rotor_thickness + 2}, r={p['ri']}, 
                center=false, $fn={self.print_settings['segments']});

        // Magnet recesses
        union() {{
            {magnet_recesses}
        }}
    }}
}}

rotor();"""

    def generate_housing(self) -> str:
        """Generate OpenSCAD code for the motor housing."""
        p = self._to_printable_params()

        outer_radius = p['ro'] + self.print_settings['wall_thickness'] * 2
        total_height = (p['stator_thickness'] +
                        p['rotor_thickness'] * 2 +
                        p['air_gap'] * 4 +
                        self.print_settings['wall_thickness'] * 2)

        # Generate mounting points
        mounting_points = self._generate_mounting_points(outer_radius,
                                                         self.print_settings['wall_thickness'])

        return f"""
// Motor Housing
module housing() {{
    difference() {{
        union() {{
            // Main cylinder
            difference() {{
                cylinder(h={total_height}, r={outer_radius}, 
                        center=false, $fn={self.print_settings['segments']});
                translate([0, 0, -{self.print_settings['wall_thickness']}])
                cylinder(h={total_height - self.print_settings['wall_thickness']}, 
                        r={outer_radius - self.print_settings['wall_thickness']}, 
                        center=false, $fn={self.print_settings['segments']});
            }}

            // Bottom mounting points
            {mounting_points}
        }}

        // Shaft hole
        cylinder(h={total_height + 1}, r={p['shaft_radius'] + self.print_settings['tolerance']},
                center=false, $fn={self.print_settings['segments']});
    }}
}}

housing();"""

    def generate_assembly(self) -> str:
        """Generate OpenSCAD code for the complete motor assembly."""
        p = self._to_printable_params()

        # Calculate assembly dimensions
        mean_radius = (p['ro'] + p['ri']) / 2
        base_height = self.print_settings['wall_thickness'] + p['air_gap']

        # Generate coil placement based on orientation
        if self.print_settings['coil_orientation'] == 'axial':
            coil_placement = self._generate_axial_coil_placement(
                p['c'], mean_radius, base_height)
        # TODO: Radial Coil Placement.
        # else:
        #     coil_placement = self._generate_radial_coil_placement(
        #         p['c'], mean_radius, base_height)

        assembly = f"""
// Import individual components
use <motor_part_coil_spool.scad>
use <motor_part_housing.scad>
use <motor_part_rotor.scad>
use <motor_part_stator.scad>

// Full Motor Assembly
module assembly() {{
    // Housing at the base
    translate([0, 0, -({self.print_settings['wall_thickness']}+{p['air_gap']})])
    color("lightgray", 0.5)
    housing();

    // Bottom rotor
    color("red", 0.5)
    rotor();

    // Stator in the middle
    color("blue", 0.5)
    translate([0, 0, {p['rotor_thickness']}+{p['air_gap']}])
    stator();

    // Top rotor
    color("red", 0.5)
    translate([0, 0, {p['rotor_thickness']}+{p['stator_thickness']}-{p['air_gap']}])
    rotor();

    // Place coils around stator
    {coil_placement}
}}

// Create the assembly
{(f'difference() {{'
  f'    assembly();'
  f'    translate([0, -500, -50])'
  f'    cube([1000, 1000, 1000]);'
  f'}}') if self.print_settings['cutaway'] else 'assembly();'}

// Add viewing parameters
$fn = {self.print_settings['segments']};
$vpt = [0, 0, 20];
$vpr = [60, 0, 45];
$vpd = 400;"""

        return assembly

    def _generate_coil_slots(self, num_coils: int, angle_per_coil: float,
                             slot_width: float, slot_depth: float, plate_thickness: float) -> str:
        """Generate OpenSCAD code for coil slots pattern."""
        mean_radius = (self.params.outer_radius + self.params.inner_radius) / 2
        slots = []

        for i in range(num_coils):
            angle = i * angle_per_coil
            if self.print_settings['coil_orientation'] == 'axial':
                slots.append(f"""
                rotate([0, 0, {angle}])
                translate([{mean_radius}, 0, {plate_thickness}-({slot_depth}-1)])
                cylinder(h={slot_depth}, r={slot_width / 2}, 
                        center=false, $fn={self.print_settings['segments']});""")
            else:
                slots.append(f"""
                rotate([0, 0, {angle}])
                translate([{mean_radius}, 0, {plate_thickness}-({slot_depth}-1)])
                rotate([90, 0, 0])
                cylinder(h={slot_depth}, r={slot_width / 2}, 
                        center=true, $fn={self.print_settings['segments']});""")

        return "\n".join(slots)

    def _generate_magnet_recesses(self, rotor_thickness: float, num_poles: int,
                                  angle_per_pole: float, magnet_type: str,
                                  magnet_width: float, magnet_length: float,
                                  magnet_depth: float) -> str:
        """Generate OpenSCAD code for magnet recess pattern."""
        mean_radius = (self.params.outer_radius + self.params.inner_radius) / 2
        recesses = []

        for i in range(num_poles):
            angle = i * angle_per_pole
            if magnet_type == "circle":
                recesses.append(f"""
                rotate([0, 0, {angle}])
                translate([{mean_radius}, 0, {rotor_thickness - magnet_depth}])
                cylinder(h=({magnet_depth}+1), d={magnet_width}, 
                        center=false, $fn={self.print_settings['segments']});""")
            else:  # square or rectangular
                recesses.append(f"""
                rotate([0, 0, {angle}])
                translate([{mean_radius}, -{magnet_width / 2}, {rotor_thickness - magnet_depth}])
                cube([{magnet_length}, {magnet_width}, ({magnet_depth}+1)]);""")

        return "\n".join(recesses)

    def _generate_mounting_points(self, radius: float, thickness: float) -> str:
        """Generate OpenSCAD code for mounting points."""
        num_points = 12
        mount_radius = radius * 0.15  # 15% of outer radius
        points = []

        for i in range(num_points):
            angle = i * (360 / num_points)
            points.append(f"""
            rotate([0, 0, {angle}])
            translate([{radius - mount_radius}, 0, 0])
            cylinder(h={thickness}, r={mount_radius}, 
                    center=false, $fn={self.print_settings['segments']});""")

        return "\n".join(points)

    def _generate_axial_coil_placement(self, num_coils: int, mean_radius: float,
                                       base_height: float) -> str:
        """Generate OpenSCAD code for axial coil placement."""
        placements = []
        angle_per_coil = 360 / num_coils

        for i in range(num_coils):
            angle = i * angle_per_coil
            placements.append(f"""
            rotate([0, 0, {angle}])
            translate([{mean_radius}, 0, {base_height}])
            rotate([0, 0, 90])
            copper_coil();""")

        return "\n".join(placements)


def generate_printable_parts(params: UnifiedMotorParameters,
                             output_prefix: str = "motor_part",
                             print_settings: Dict[str, Any] = None) -> Dict[str, str]:
    """Generate all printable parts and save to separate files.

    Args:
        params: UnifiedMotorParameters containing motor specifications
        output_prefix: Prefix for output files
        print_settings: Dictionary of 3D printing settings

    Returns:
        Dictionary containing OpenSCAD code for each part
    """
    # Create generator instance
    generator = PrintableMotorPartsGenerator(params, print_settings)

    # Generate each part
    parts = {
        "coil_spool": generator.generate_coil_spool(),
        "stator": generator.generate_stator_plate(),
        "rotor": generator.generate_rotor_plate(),
        "housing": generator.generate_housing(),
        "assembly": generator.generate_assembly()
    }

    # Create output directory
    Path("./printable").mkdir(parents=True, exist_ok=True)

    # Save each part to a separate file
    for name, scad_code in parts.items():
        filename = f"./printable/{output_prefix}_{name}_unified.scad"
        with open(filename, 'w') as f:
            f.write(scad_code)

        print(f"Generated {filename}")

    return parts


def create_sample_motor() -> None:
    """Create example motor parts using default parameters."""
    from UnifiedMotorParameters import UnifiedMotorParameters

    # Create sample parameters
    params = UnifiedMotorParameters(
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
        target_diameter=50,
        torque=0.1,
        tolerance=0.2,
        target_torque=0.1,
        estimated_torque=None,
        efficiency=0.0,
        resistance=0.0,
        current=0.0,
        coil_width=8.0,
        coil_height=15.0,
        total_height=None
    )

    # Print settings
    print_settings = {
        'wall_thickness': 2.0,
        'tolerance': 0.2,
        'layer_height': 0.2,
        'segments': 100,
        'coil_orientation': 'axial',
        'cutaway': True
    }

    # Generate parts
    generate_printable_parts(params, print_settings=print_settings)
    print("Sample motor parts generated successfully!")


if __name__ == "__main__":
    create_sample_motor()
