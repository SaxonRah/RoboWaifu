import math
from pathlib import Path


class PrintableMotorPartsGenerator:
    def __init__(self, params):
        self.params = params
        self.segments = 100  # For smooth circles

        # Add additional parameters for 3D printing
        self.wall_thickness = 2.0  # Wall thickness for structural integrity
        self.tolerance = 0.2  # Printing tolerance for fits
        self.layer_height = 0.2  # Standard 3D printing layer height
        self.coil_orientation = params.get('coil_orientation', 'axial')  # 'axial' or 'radial'

    def generate_coil_spool(self):
        """Generate a printable spool for winding coils"""
        coil_width = self.params.get('coil_width', 8)
        wire_diameter = self.params.get('Wire_Diameter', 0.65)

        # Calculate spool dimensions
        spool_height = coil_width + 2 * self.wall_thickness
        inner_radius = wire_diameter * 5  # Core radius for winding
        flange_radius = inner_radius * 2  # Outer flange to hold wire

        return f"""
// Coil Winding Spool
module coil_spool() {{
    difference() {{
        union() {{
            // Center core
            cylinder(h={spool_height}, r={inner_radius}, center=true, $fn={self.segments});

            // Bottom flange
            translate([0, 0, -{spool_height / 2}])
            cylinder(h={self.wall_thickness}, r={flange_radius}, center=false, $fn={self.segments});

            // Top flange
            translate([0, 0, {spool_height / 2 - self.wall_thickness}])
            cylinder(h={self.wall_thickness}, r={flange_radius}, center=false, $fn={self.segments});
        }}

        // Center hole for mounting
        cylinder(h={spool_height + 1}, r={inner_radius / 2}, center=true, $fn={self.segments});
    }}
}}

// Define copper coil appearance
module copper_coil() {{
    color("Orange", 0.8)
    coil_spool();
}}

copper_coil();
"""

    def generate_stator_plate(self):
        """Generate a printable stator plate with coil slots"""
        num_coils = self.params['c']
        outer_radius = self.params['ro']
        inner_radius = self.params['ri']
        angle_per_coil = 360 / num_coils

        # Calculate slot dimensions
        slot_width = self.params.get('coil_width', 8) + self.tolerance
        slot_depth = self.params.get('coil_length', 15) + self.tolerance

        stator_plate_thickness = self.params.get('stator_thickness', 15) + self.tolerance
        magnet_depth = self.params.get('magnet_thickness', 3) + self.tolerance
        #  plate_thickness = stator_plate_thickness + magnet_depth + 2  # Add material below magnets
        plate_thickness = stator_plate_thickness - magnet_depth

        return f"""
// Printable Stator Plate
module stator() {{

    //translate([0, 0, {plate_thickness}])
    difference() {{
        // Main plate
        cylinder(h={plate_thickness}, r={outer_radius}, center=false, $fn={self.segments});

        // Center hole
        translate([0, 0, -1])
        cylinder(h={plate_thickness + 2}, r={inner_radius}, center=false, $fn={self.segments});

        // Coil slots
        union() {{
            {self._generate_coil_slots(num_coils, angle_per_coil, slot_width, slot_depth, plate_thickness)}
        }}
    }}
}}

stator();
"""

    def generate_rotor_plate(self):
        """Generate a printable rotor plate with magnet recesses"""
        num_poles = self.params['p']
        outer_radius = self.params['ro']
        inner_radius = self.params['ri']
        angle_per_pole = 360 / num_poles

        # Calculate magnet recess dimensions
        magnet_width = self.params.get('magnet_width', 10) + self.tolerance
        magnet_length = (outer_radius - inner_radius) * 0.4  # 40% of radial space
        #  magnet_length = self.params.get('magnet_length', 10) + self.tolerance
        magnet_depth = self.params.get('magnet_thickness', 3) + self.tolerance
        #  "magnet_type": "circle",  # circle, square, rectangle
        magnet_type = self.params['magnet_type']
        plate_thickness = magnet_depth + 2  # Add material below magnets
        rotor_thickness = self.params['rotor_thickness'] + magnet_depth

        built_magnets = (self._generate_magnet_recesses(
                    rotor_thickness,
                    num_poles,
                    angle_per_pole,
                    magnet_type,
                    magnet_width,
                    magnet_length,
                    magnet_depth)
        )

        return f"""
// Printable Rotor Plate
module rotor() {{
    difference() {{
        // Main plate
        cylinder(h={rotor_thickness}, r={outer_radius}, center=false, $fn={self.segments});

        // Center hole
        translate([0, 0, -1])
        cylinder(h={rotor_thickness + 2}, r={inner_radius}, center=false, $fn={self.segments});

        // Magnet recesses
        union() {{
            {built_magnets}
        }}
    }}
}}

rotor();
"""

    def generate_housing(self):
        """Generate a printable housing for the motor assembly"""
        outer_radius = self.params['ro'] + self.wall_thickness * 2
        stator_plate_thickness = self.params.get('stator_thickness', 15) + self.tolerance
        magnet_depth = self.params.get('magnet_thickness', 3) + self.tolerance
        #  plate_thickness = stator_plate_thickness + magnet_depth + 2  # Add material below magnets
        plate_thickness = stator_plate_thickness + magnet_depth
        air_gap = self.params.get('air_gap', 1)
        height = (plate_thickness +
                  self.params.get('rotor_thickness', 5) * 2 +
                  air_gap * 4 +
                  self.wall_thickness * 2 +
                  air_gap * 2)

        return f"""
// Motor Housing
module housing() {{
    difference() {{
        union() {{
            // Main cylinder
            difference() {{
                cylinder(h={height}, r={outer_radius}, center=false, $fn={self.segments});
                translate([0, 0, -{self.wall_thickness}])
                cylinder(h={height - self.wall_thickness}, r={outer_radius - self.wall_thickness}, 
                        center=false, $fn={self.segments});
            }}

            // Bottom mounting points
            // TODO: Add screw holes
            {self._generate_mounting_points(outer_radius, self.wall_thickness)}
        }}

        // Shaft hole
        cylinder(h={height + 1}, r={self.params.get('shaft_radius', 10) + self.tolerance},
                center=false, $fn={self.segments});
    }}
}}

housing();
"""

    def _generate_axial_coil_placement(self, num_coils, mean_radius, base_height):
        """Generate coil placement code for axial orientation"""
        coil_placements = []
        angle_per_coil = 360 / num_coils
        # stator_thickness = self.params.get('stator_thickness', 15)
        air_gap = self.params.get('air_gap', 1)
        magnet_depth = self.params.get('magnet_thickness', 3) + self.tolerance
        rotor_thickness = self.params['rotor_thickness'] + magnet_depth

        for i in range(num_coils):
            angle = i * angle_per_coil
            coil_placements.append(f"""
            rotate([0, 0, {angle}])
            translate([{mean_radius}, 0, {rotor_thickness} + {air_gap}])
            rotate([0, 0, 90])
            copper_coil();""")

        return "\n".join(coil_placements)

    def _generate_radial_coil_placement(self, num_coils, mean_radius, base_height):
        """Generate coil placement code for radial orientation"""
        coil_placements = []
        angle_per_coil = 360 / num_coils

        # stator_thickness = self.params.get('stator_thickness', 15)
        air_gap = self.params.get('air_gap', 1)
        magnet_depth = self.params.get('magnet_thickness', 3) + self.tolerance
        rotor_thickness = self.params['rotor_thickness'] + magnet_depth

        for i in range(num_coils):
            angle = i * angle_per_coil
            coil_placements.append(f"""
            rotate([0, 0, {angle}])
            translate([{mean_radius}, 0, {rotor_thickness} + {air_gap}])
            rotate([90, 0, 0])
            copper_coil();""")

        return "\n".join(coil_placements)

    def generate_assembly(self):
        """Generate an assembly of all motor parts"""
        air_gap = self.params.get('air_gap', 1)
        stator_thickness = self.params.get('stator_thickness', 15)
        num_coils = self.params['c']
        mean_radius = (self.params['ro'] + self.params['ri']) / 2
        base_height = self.wall_thickness + air_gap

        magnet_depth = self.params.get('magnet_thickness', 3) + self.tolerance
        plate_thickness = stator_thickness + magnet_depth
        housing_height = (
                plate_thickness +
                self.params.get('rotor_thickness', 5) * 2 +
                self.params.get('air_gap', 1) * 4 +
                self.wall_thickness * 2
        )

        rotor_thickness = self.params['rotor_thickness'] + magnet_depth

        # Generate coil placement based on orientation
        if self.coil_orientation == 'axial':
            coil_placement = self._generate_axial_coil_placement(num_coils, mean_radius, base_height)
        else:  # radial orientation
            coil_placement = self._generate_radial_coil_placement(num_coils, mean_radius, base_height)

        will_cutaway = self.params['cutaway']

        temp_assembly = f"""
// Import individual components
use <motor_part_coil_spool.scad>
use <motor_part_housing.scad>
use <motor_part_rotor.scad>
use <motor_part_stator.scad>

// Full Motor Assembly

// Assembly module
module assembly() {{
    // Housing at the base
    
    translate([0, 0, -({self.wall_thickness}+{air_gap})])
    color("lightgray", 0.5)
    housing();

    // Bottom rotor
    color("red", 0.5)
    translate([0, 0, 0])
    rotor();
    
    // Stator in the middle
    color("blue", 0.5)
    translate([0, 0, {rotor_thickness}+{air_gap}])
    stator();

    // Top rotor
    color("red", 0.5)
    translate([0, 0, {rotor_thickness}+{stator_thickness}-{air_gap}])
    rotor();

    // Place coils around stator
    {coil_placement}
}}
"""
        cutaway_assembly = f"""

// Create the assembly
assembly();

// Add viewing parameters for better visualization
$fn = {self.segments};  // Smooth circles
$vpt = [0, 0, 20];  // View point
$vpr = [60, 0, 45];  // View rotation
$vpd = 400;  // View distance
"""
        base_assembly = f"""
// Create the assembly
difference() {{
        assembly();

        // Cutaway
        translate([0, -500, -50])
        cube([1000, 1000, 1000]);
}}
// Add viewing parameters for better visualization
$fn = {self.segments};  // Smooth circles
$vpt = [0, 0, 20];  // View point
$vpr = [60, 0, 45];  // View rotation
$vpd = 400;  // View distance
"""
        return (temp_assembly + base_assembly) if will_cutaway else (temp_assembly + cutaway_assembly)

    def _generate_coil_slots(self, num_coils, angle_per_coil, slot_width, slot_depth, plate_thickness):
        """Helper method to generate coil slots"""
        slots = []
        mean_radius = (self.params['ro'] + self.params['ri']) / 2

        coil_width = self.params.get('coil_width', 8)
        wire_diameter = self.params.get('Wire_Diameter', 0.65)

        # Calculate spool dimensions
        spool_height = coil_width + 2 * self.wall_thickness
        inner_radius = wire_diameter * 5  # Core radius for winding
        flange_radius = inner_radius * 2  # Outer flange to hold wire

        air_gap = self.params.get('air_gap', 1)

        # Adjust slot orientation based on coil orientation
        if self.coil_orientation == 'axial':
            for i in range(num_coils):
                angle = i * angle_per_coil
                slots.append(f"""
                rotate([0, 0, {angle}])
                translate([{mean_radius}, 0, {plate_thickness}-{self.wall_thickness + air_gap}])
                cylinder(h={spool_height}, r={flange_radius}, center=false, $fn={self.segments});""")

        else:  # radial orientation
            for i in range(num_coils):
                angle = i * angle_per_coil
                slots.append(f"""
                rotate([0, 0, {angle}])
                translate([{mean_radius}, 0, {plate_thickness}-{self.wall_thickness + air_gap}])
                cylinder(h={spool_height}, r={flange_radius}, center=false, $fn={self.segments});""")

        return "\n".join(slots)

    def _generate_magnet_recesses(self, rotor_thickness, num_poles, angle_per_pole, magnet_type, magnet_width, magnet_length, magnet_depth):
        """Helper method to generate magnet recesses"""
        recesses = []
        mean_radius = (self.params['ro'] + self.params['ri']) / 2
        air_gap = self.params.get('air_gap', 1)

        if magnet_type == "square":
            #  //translate([{mean_radius}, 0, {self.wall_thickness}])
            for i in range(num_poles):
                angle = i * angle_per_pole
                recesses.append(f"""
                rotate([0, 0, {angle}])
                translate([{mean_radius}, 0, {rotor_thickness}-{self.wall_thickness + air_gap}])
                cube([{magnet_length}, {magnet_width}, {rotor_thickness}-{magnet_depth}], center=true);""")
        else:
            #  //translate([{mean_radius}, 0, {self.wall_thickness}])
            for i in range(num_poles):
                angle = i * angle_per_pole
                recesses.append(f"""
                rotate([0, 0, {angle}])
                translate([{mean_radius}, 0, 0])
                cylinder(h={rotor_thickness}-{magnet_depth}, d={magnet_width}, center=true, $fn={self.segments});""")

        return "\n".join(recesses)

    def _generate_mounting_points(self, radius, thickness):
        """Helper method to generate mounting points"""
        mount_points = []
        num_points = 12
        mount_radius = 10

        for i in range(num_points):
            angle = i * (360 / num_points)
            mount_points.append(f"""
            rotate([0, 0, {angle}])
            translate([{radius - mount_radius}, 0, 0])
            cylinder(h={thickness}, r={mount_radius}, center=false, $fn={self.segments});""")

        return "\n".join(mount_points)


def generate_printable_parts(params, output_prefix="motor_part"):
    """Generate all printable parts and save to separate files"""
    generator = PrintableMotorPartsGenerator(params)

    # Generate each part
    parts = {
        "coil_spool": generator.generate_coil_spool(),
        "stator": generator.generate_stator_plate(),
        "rotor": generator.generate_rotor_plate(),
        "housing": generator.generate_housing(),
        "assembly": generator.generate_assembly()
    }

    Path("./printable").mkdir(parents=True, exist_ok=True)
    # Save each part to a separate file
    for name, scad_code in parts.items():
        filename = f"./printable/{output_prefix}_{name}.scad"
        with open(filename, 'w') as f:
            f.write(scad_code)

    return parts


# Example usage
if __name__ == "__main__":

    small_motor_params = {
        "Wire_Diameter": 0.65,
        "p": 12,  # poles
        "c": 18,  # coils
        "ro": 35,  # outer radius in mm
        "ri": 15,  # inner radius in mm
        "coil_width": 8,
        "coil_length": 15,
        "magnet_width": 10,  # 10mmX3mm
        #  3mmX2mmX88, 8mmX2mmX70, 10mmX2mmX50, 12mmX2mmX25, 15mmX2mmX20, 32mmX2mmX2
        "magnet_length": 10,
        "magnet_thickness": 3,
        "magnet_type": "circle",  # circle or square
        # if circle then the magnet_width is used for the circle's diameter.
        "stator_thickness": 10,
        "rotor_thickness": 5,
        "air_gap": 1,
        "shaft_radius": 10,
        "coil_orientation": "axial",  # axial or radial
        "cutaway": True  # True or False
    }

    big_motor_params = {
        "Wire_Diameter": 0.65,
        "p": 12,  # poles
        "c": 18,  # coils
        "ro": 65,  # outer radius in mm
        "ri": 35,  # inner radius in mm
        "coil_width": 8,
        "coil_length": 15,
        "magnet_width": 20,
        "magnet_length": 10,
        "magnet_thickness": 3,
        "magnet_type": "circle",  # circle or square
        # if circle then the magnet_width is used for the circle's diameter.
        "stator_thickness": 15,
        "rotor_thickness": 5,
        "air_gap": 1,
        "shaft_radius": 10,
        "coil_orientation": "axial",  # axial or radial
        "cutaway": False  # True or False (Cuts model in half for cutaway view)
    }

    parts = generate_printable_parts(big_motor_params)
    print("3D printable motor parts generated successfully!")
