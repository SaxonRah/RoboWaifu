
// Import individual components
use <motor_part_coil_spool_unified.scad>
use <motor_part_housing_unified.scad>
use <motor_part_rotor_unified.scad>
use <motor_part_stator_unified.scad>

// Full Motor Assembly
module assembly() {
    // Housing at the base
    translate([0, 0, -(2.0+1.0)])
    color("lightgray", 0.5)
    housing();

    // Bottom rotor
    color("red", 0.5)
    rotor();

    // Stator in the middle
    color("blue", 0.5)
    translate([0, 0, 5.0+1.0])
    stator();

    // Top rotor
    color("red", 0.5)
    translate([0, 0, 5.0+10.0-1.0])
    rotor();

    // Place coils around stator
    
            rotate([0, 0, 0.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 20.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 40.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 60.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 80.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 100.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 120.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 140.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 160.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 180.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 200.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 220.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 240.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 260.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 280.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 300.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 320.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();

            rotate([0, 0, 340.0])
            translate([25.0, 0, 3.0])
            rotate([0, 0, 90])
            copper_coil();
}

// Create the assembly
difference() {    assembly();    translate([0, -500, -50])    cube([1000, 1000, 1000]);}

// Add viewing parameters
$fn = 100;
$vpt = [0, 0, 20];
$vpr = [60, 0, 45];
$vpd = 400;