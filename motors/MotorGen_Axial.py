import math


class AxialFluxMotorGenerator:
    def __init__(self, params):
        self.params = params
        self.segments = 72  # For smooth circles

    def generate_header(self):
        """Generate OpenSCAD file header with parameters as variables"""
        header = "// Axial Flux Motor Parameters\n"
        for key, value in self.params.items():
            # Convert scientific notation to decimal
            if isinstance(value, float) and 'e' in str(value).lower():
                value = f"{value:.10f}"
            header += f"{key} = {value};\n"
        return header

    def generate_rotor_disc(self):
        """Generate rotor disc with magnet positions"""
        magnets = []
        num_poles = self.params['p']
        angle_per_pole = 360 / num_poles
        magnet_arc = angle_per_pole * 0.8  # 80% of pole pitch

        rotor = f"""
// Rotor Disc
color("silver")
difference() {{
    cylinder(h=rotor_thickness, r=ro, center=true, $fn={self.segments});
    cylinder(h=rotor_thickness + 1, r=shaft_radius, center=true, $fn={self.segments});
}}

// Permanent Magnets
color("blue")
"""

        for i in range(num_poles):
            angle = i * angle_per_pole
            polarity = 1 if i % 2 == 0 else -1
            magnets.append(f"""
    rotate([0, 0, {angle}])
    translate([0, 0, {polarity * 0.5}])
    rotate_extrude(angle={magnet_arc}, $fn={self.segments})
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);""")

        return rotor + "\n".join(magnets)

    def generate_stator_disc(self):
        """Generate stator disc with coil slots"""
        num_coils = self.params['c']
        angle_per_coil = 360 / num_coils

        stator = f"""
// Stator Disc
color("gray")
difference() {{
    cylinder(h=stator_thickness, r=ro, center=true, $fn={self.segments});
    cylinder(h=stator_thickness + 1, r=ri, center=true, $fn={self.segments});
"""

        # Generate coil slots
        for i in range(num_coils):
            angle = i * angle_per_coil
            phase = i % 3  # Three-phase winding
            stator += f"""
    // Coil slot {i + 1}
    rotate([0, 0, {angle}])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);"""

        stator += "\n}"
        return stator

    def generate_windings(self):
        """Generate simplified coil visualization"""
        num_coils = self.params['c']
        angle_per_coil = 360 / num_coils
        windings = []
        colors = ["red", "green", "blue"]  # Phase colors

        for i in range(num_coils):
            angle = i * angle_per_coil
            phase = i % 3
            windings.append(f"""
    // Coil {i + 1}
    color("{colors[phase]}")
    rotate([0, 0, {angle}])
    translate([(ri + ro)/2, 0, 0])
    cylinder(h=coil_height, r=wire_diameter/2, center=true, $fn=16);""")

        return "\n".join(windings)

    def generate_full_model(self):
        """Generate complete axial flux motor model"""
        # Additional parameters for visualization
        additional_params = """
// Additional parameters
rotor_thickness = 5;    // Rotor disc thickness
stator_thickness = 15;  // Stator disc thickness
air_gap = 1;           // Air gap between rotor and stator
shaft_radius = 10;     // Shaft hole radius
magnet_thickness = 3;  // Permanent magnet thickness
magnet_width = 10;     // Radial width of magnets
coil_width = 8;        // Width of coil slots
coil_length = 15;      // Length of coil slots
coil_height = 12;      // Height of coil windings
wire_diameter = Wire_Diameter;  // From input parameters
"""

        assembly = f"""
// Main Assembly
union() {{
    // Bottom rotor
    translate([0, 0, -(stator_thickness/2 + air_gap + rotor_thickness/2)])
    {self.generate_rotor_disc()}

    // Stator
    {self.generate_stator_disc()}

    // Windings
    union() {{
        {self.generate_windings()}
    }}

    // Top rotor (mirrored magnets)
    translate([0, 0, (stator_thickness/2 + air_gap + rotor_thickness/2)])
    mirror([0, 0, 1])
    {self.generate_rotor_disc()}
}}
"""

        return f"""
// Generated Axial Flux Motor Model
{self.generate_header()}
{additional_params}
{assembly}
"""


def create_axial_motor_scad(params, output_file="axial_motor_model.scad"):
    """Create OpenSCAD file for axial flux motor with given parameters"""
    generator = AxialFluxMotorGenerator(params)
    scad_code = generator.generate_full_model()

    with open(output_file, 'w') as f:
        f.write(scad_code)
    return scad_code


# Example usage
if __name__ == "__main__":
    params = {
        "DC_Bus_Voltage": 24,
        "Base_Speed": 700,
        "Rated_Current": 7,
        "Wire_Diameter": 0.65,
        "p": 12,  # poles
        "c": 18,  # coils
        "ro": 65,  # outer radius in mm
        "ri": 35,  # inner radius in mm
        "B": 0.5,  # magnetic flux density
        "Acoil": 4.92e-4,
        "Arotor": 9.42e-3
    }

    scad_code = create_axial_motor_scad(params)
    print("Axial flux motor OpenSCAD file generated successfully!")
