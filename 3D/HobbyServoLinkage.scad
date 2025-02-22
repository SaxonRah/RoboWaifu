// Servo Gear Assembly with Torque Amplification and Integrated Bearing
include <gears.scad>

/* Parameters */
// Servo dimensions (typical 20kg servo)
servo_width = 40.5;
servo_length = 20.5;
servo_tabs_length = 54.5;
servo_height = 42;
servo_shaft_diameter = 5.9;
servo_shaft_height = 4;

// Bearing dimensions
bearing_outer_diameter = 22;
bearing_inner_diameter = 8;
bearing_thickness = 7;

// Gear parameters
modul = 2;  // Affects gear tooth size
driver_teeth = 12;  // Servo gear
driven_teeth = 30; // Output gear - 2.5:1 ratio
pressure_angle = 20; // Standard pressure angle
helix_angle = 15;  // Add slight helix for smoother engagement
gear_width = 8;   // Width of gears

// Mounting plate dimensions
plate_thickness = 8;  // Increased thickness for bearing integration
plate_margin = 10;
plate_width = servo_width + (2 * plate_margin) + bearing_outer_diameter;
plate_length = servo_length + (2 * plate_margin);

// Limb attachment parameters
limb_length = 100;
limb_width = 20;
limb_thickness = 8;

/* Main Components */
module servo() {
    // Servo body
    color("blue")
    cube([servo_width, servo_length, servo_height]);

    color("blue")
    translate([-servo_tabs_length/8, 0, 28])
    cube([servo_tabs_length, servo_length, 5]);

    // Servo shaft
    color("silver")
    translate([servo_width/2, servo_length/2, servo_height])
    cylinder(h=servo_shaft_height, d=servo_shaft_diameter, $fn=20);
}

module bearing() {
    difference() {
        color("silver")
        cylinder(h=bearing_thickness, d=bearing_outer_diameter, $fn=50);
        translate([0, 0, -1])
        cylinder(h=bearing_thickness + 2, d=bearing_inner_diameter, $fn=50);
    }
}

module mounting_plate() {
    driven_gear_x = plate_width/2 + modul*(driver_teeth + driven_teeth)/2;

    difference() {
        // Base plate
        cube([plate_width*2, plate_length, plate_thickness]);

        // Servo cutout
        translate([plate_margin, plate_margin, -1])
        cube([servo_width, servo_length, plate_thickness + 2]);

        // Bearing seat cutout
        translate([driven_gear_x, plate_length/2, plate_thickness - bearing_thickness])
        cylinder(h=bearing_thickness + 1, d=bearing_outer_diameter, $fn=50);

        // Through hole for bearing shaft
        translate([driven_gear_x, plate_length/2, -1])
        cylinder(h=plate_thickness + 2, d=bearing_inner_diameter, $fn=50);

        // Mounting holes for plate
        for(x = [5, plate_width-5])
            for(y = [5, plate_length-5])
                translate([x, y, -1])
                cylinder(h=plate_thickness + 2, d=3, $fn=20);
    }
}

module output_shaft() {
    // Main shaft
    color("purple")
    cylinder(h=servo_height + gear_width + 10, d=bearing_inner_diameter - 0.2, $fn=30);

    // Shoulder for gear mounting
    translate([0, 0, servo_height + 5])
    cylinder(h=2, d=bearing_inner_diameter + 2, $fn=30);
}

module output_limb() {
    difference() {
        cube([limb_length, limb_width, limb_thickness]);
        // Mounting hole for driven gear shaft
        translate([limb_width/2, limb_width/2, -1])
        cylinder(h=limb_thickness + 2, d=bearing_inner_diameter, $fn=30);
    }
}

/* Assembly */
module assembly() {
    // Calculate positions
    driven_gear_x = plate_width/2 + modul*(driver_teeth + driven_teeth)/2;

    // Mounting plate
    color("gray")
    mounting_plate();

    // Servo
    translate([plate_margin, plate_margin, 0])
    servo();

    // Lower bearing (integrated in mounting plate)
    translate([driven_gear_x, plate_length/2, plate_thickness - bearing_thickness])
    bearing();

    // Output shaft
    translate([driven_gear_x, plate_length/2, 0])
    output_shaft();

    // Driver gear (on servo) - using herringbone for better engagement
    color("red")
    translate([plate_width/2, plate_length/2, plate_thickness + servo_height + servo_shaft_height])
    herringbone_gear(
        modul=modul,
        tooth_number=driver_teeth,
        width=gear_width,
        bore=servo_shaft_diameter,
        pressure_angle=pressure_angle,
        helix_angle=helix_angle,
        optimized=false);

    // Driven gear
    color("orange")
    translate([driven_gear_x, plate_length/2, plate_thickness + servo_height + servo_shaft_height])
    herringbone_gear(
        modul=modul,
        tooth_number=driven_teeth,
        width=gear_width,
        bore=bearing_inner_diameter,
        pressure_angle=pressure_angle,
        helix_angle=helix_angle,
        optimized=false);

    // Output limb
    translate([driven_gear_x - limb_width/2,
              plate_length/2 - limb_width/2,
              plate_thickness + servo_height + servo_shaft_height + gear_width])
    output_limb();
}

// Render the assembly
assembly();

/*
Design Notes:
1. Integrated bearing support directly into mounting plate:
   - Bearing seat machined into plate
   - Increased plate thickness to accommodate bearing
   - Through-hole for shaft alignment
2. Added output shaft with:
   - Tight tolerance fit for bearing
   - Shoulder for gear mounting
   - Extended length for limb attachment
3. Maintained 2.5:1 gear ratio with herringbone pattern
4. Simplified design by removing separate bearing support
5. All dimensions remain parametric for customization
*/