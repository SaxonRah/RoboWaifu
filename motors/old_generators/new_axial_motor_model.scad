
// Generated Axial Flux Motor Model
// Axial Flux Motor Parameters
DC_Bus_Voltage = 24;
Base_Speed = 700;
Rated_Current = 7;
Wire_Diameter = 0.65;
p = 12;
c = 18;
ro = 65;
ri = 35;
B = 0.5;
Acoil = 0.000492;
Arotor = 0.00942;
configuration = single_rotor_single_stator;



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


// Main Assembly
union() {

    // Single Rotor
    translate([0, 0, -air_gap - rotor_thickness / 2])
    
// Rotor Disc
color("SlateGray")
difference() {
    cylinder(h=rotor_thickness, r=ro, center=true, $fn=100);
    cylinder(h=rotor_thickness + 1, r=shaft_radius, center=true, $fn=100);
}

// Permanent Magnets
union() {

    color("RoyalBlue")
    rotate([0, 0, 0.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("IndianRed")
    rotate([0, 0, 30.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("RoyalBlue")
    rotate([0, 0, 60.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("IndianRed")
    rotate([0, 0, 90.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("RoyalBlue")
    rotate([0, 0, 120.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("IndianRed")
    rotate([0, 0, 150.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("RoyalBlue")
    rotate([0, 0, 180.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("IndianRed")
    rotate([0, 0, 210.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("RoyalBlue")
    rotate([0, 0, 240.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("IndianRed")
    rotate([0, 0, 270.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("RoyalBlue")
    rotate([0, 0, 300.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);

    color("IndianRed")
    rotate([0, 0, 330.0])
    translate([0, 0, 0])
    rotate_extrude(angle=24.0, $fn=100)
    translate([(ri + ro)/2, 0])
    square([magnet_width, magnet_thickness], center=true);
}

    // Single Stator
    
// Stator Disc
color("OliveDrab")
difference() {
    cylinder(h=stator_thickness, r=ro, center=true, $fn=100);
    cylinder(h=stator_thickness + 1, r=ri, center=true, $fn=100);

    // Coil slot 1
    rotate([0, 0, 0.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 2
    rotate([0, 0, 20.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 3
    rotate([0, 0, 40.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 4
    rotate([0, 0, 60.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 5
    rotate([0, 0, 80.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 6
    rotate([0, 0, 100.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 7
    rotate([0, 0, 120.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 8
    rotate([0, 0, 140.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 9
    rotate([0, 0, 160.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 10
    rotate([0, 0, 180.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 11
    rotate([0, 0, 200.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 12
    rotate([0, 0, 220.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 13
    rotate([0, 0, 240.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 14
    rotate([0, 0, 260.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 15
    rotate([0, 0, 280.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 16
    rotate([0, 0, 300.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 17
    rotate([0, 0, 320.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
    // Coil slot 18
    rotate([0, 0, 340.0])
    translate([(ri + ro)/2, 0, 0])
    cube([coil_width, coil_length, stator_thickness + 1], center=true);
}

}
