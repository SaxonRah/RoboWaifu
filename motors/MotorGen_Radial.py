# import math
# Basic Radial Flux Generated Motor Model


class MotorScadGenerator:
    def __init__(self, params):
        self.params = params
        self.stator_segments = 72  # For smooth circles

    def generate_header(self):
        """Generate OpenSCAD file header with parameters as variables"""
        header = "// Motor Parameters\n"
        for key, value in self.params.items():
            # Convert scientific notation to decimal
            if isinstance(value, float) and 'e' in str(value).lower():
                value = f"{value:.10f}"
            header += f"{key} = {value};\n"
        return header

    def generate_rotor(self):
        """Generate rotor cylinder"""
        return f"""
// Rotor
color("silver")
difference() {{
    cylinder(h=motor_length, r=ri, center=true, $fn={self.stator_segments});
    // Shaft hole
    cylinder(h=motor_length + 1, r=shaft_radius, center=true, $fn={self.stator_segments});
}}
"""

    def generate_stator(self):
        """Generate stator with slots"""
        angle_per_slot = 360 / self.params['c']
        return f"""
// Stator
color("gray")
difference() {{
    cylinder(h=motor_length, r=ro, center=true, $fn={self.stator_segments});
    cylinder(h=motor_length + 1, r=ri + gap, center=true, $fn={self.stator_segments});
    {self.generate_slots(angle_per_slot)}
}}
"""

    def generate_slots(self, angle_per_slot):
        """Generate slot cutouts in stator"""
        slots = []
        for i in range(self.params['c']):
            angle = i * angle_per_slot
            slots.append(f"""
    rotate([0, 0, {angle}])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);""")
        return "\n".join(slots)

    def generate_windings(self):
        """Generate simplified winding visualization"""
        angle_per_slot = 360 / self.params['c']
        windings = []
        for i in range(self.params['c']):
            angle = i * angle_per_slot
            phase = i % 3  # For three-phase coloring
            colors = ["red", "green", "blue"]
            windings.append(f"""
    // Winding {i + 1}
    color("{colors[phase]}")
    rotate([0, 0, {angle}])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);""")
        return "\n".join(windings)

    def generate_full_model(self):
        """Generate complete OpenSCAD model"""
        motor_length = 50  # Default motor length
        shaft_radius = 10  # Default shaft radius
        gap = 0.5  # Air gap
        slot_depth = 15  # Slot depth
        slot_width = 5  # Slot width

        additional_params = f"""
// Additional parameters
motor_length = {motor_length};
shaft_radius = {shaft_radius};
gap = {gap};
slot_depth = {slot_depth};
slot_width = {slot_width};
"""

        scad_code = f"""
// Generated Electric Motor Model
{self.generate_header()}
{additional_params}

// Main assembly
union() {{
    {self.generate_rotor()}
    {self.generate_stator()}
    // Windings
    union() {{
        {self.generate_windings()}
    }}
}}
"""
        return scad_code


def create_motor_scad(params, output_file="radial_motor_model.scad"):
    """Create OpenSCAD file with given parameters"""
    generator = MotorScadGenerator(params)
    scad_code = generator.generate_full_model()

    with open(output_file, 'w') as f:
        f.write(scad_code)
    return scad_code


# Example usage
if __name__ == "__main__":
    # Convert scientific notation to regular float
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
        "Acoil": 4.92e-4,  # converted from scientific notation
        "Arotor": 9.42e-3  # converted from scientific notation
    }

    scad_code = create_motor_scad(params)
    print("OpenSCAD file generated successfully!")
