
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
    translate([0, 0, 13.88549476816575])
        stator();

    // Top rotor
    translate([0, 0, 119.3803708342557])
        rotor();
}

assembly();
