
// Complete motor assembly
$fn = 100;

// Import individual components
use <rotor.scad>
use <stator.scad>
use <housing.scad>

// Assembly
module assembly() {
    // Housing
    housing();
    
    // Bottom rotor
    rotor();

    // Stator (with air gap)
    translate([0, 0, 9.069980123839466])
        stator();

    // Top rotor
    translate([0, 0, 20.069980123839464])
        rotor();
}

assembly();
