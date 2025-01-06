
// Stator with coils
$fn = 100;

module coil() {
    // Coil assembly
    color("Silver") // Coil former color
    difference() {
        cylinder(h=5, 
                d=90.0);

        translate([0, 0, -1])
            cylinder(h=7, 
                    d=60.0);
    }

    // Wire winding visualization
    color("Copper")
    translate([0, 0, 0.5])  // Slight offset for visibility
    difference() {
        cylinder(h=4, 
                d=89.0);

        translate([0, 0, -1])
            cylinder(h=7, 
                    d=61.0);
    }
}

module stator() {
    // Base stator disk
    difference() {
        cylinder(h=10, d=200);

        // Center hole
        translate([0, 0, -1])
            cylinder(h=12, d=20);
    }

    // Coils
    
    // Coil 1
    translate([94.53803619658584, 0.0, 5])
        coil();

    // Coil 2
    translate([47.269018098292925, 81.87234097013612, 5])
        coil();

    // Coil 3
    translate([-47.2690180982929, 81.87234097013614, 5])
        coil();

    // Coil 4
    translate([-94.53803619658584, 1.1577570342582546e-14, 5])
        coil();

    // Coil 5
    translate([-47.26901809829296, -81.87234097013611, 5])
        coil();

    // Coil 6
    translate([47.269018098292925, -81.87234097013612, 5])
        coil();
}

stator();
