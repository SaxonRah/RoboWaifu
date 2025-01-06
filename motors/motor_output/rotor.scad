
// Rotor plate with magnets
$fn = 100;

module rotor() {
    difference() {
        // Main disk
        cylinder(h=8.069980123839466, d=300);

        // Center hole
        translate([0, 0, -1])
            cylinder(h=10.069980123839466, d=20);
    }

    // Magnets
    
    // Magnet 1
    translate([105.0 * cos(0.0), 105.0 * sin(0.0), 0])
        rotate([0, 0, 0.0])
        cylinder(h=3.0699801238394655, 
                d=65.97344572538566,
                $fn=30);

    // Magnet 2
    translate([105.0 * cos(45.0), 105.0 * sin(45.0), 0])
        rotate([0, 0, 45.0])
        cylinder(h=3.0699801238394655, 
                d=65.97344572538566,
                $fn=30);

    // Magnet 3
    translate([105.0 * cos(90.0), 105.0 * sin(90.0), 0])
        rotate([0, 0, 90.0])
        cylinder(h=3.0699801238394655, 
                d=65.97344572538566,
                $fn=30);

    // Magnet 4
    translate([105.0 * cos(135.0), 105.0 * sin(135.0), 0])
        rotate([0, 0, 135.0])
        cylinder(h=3.0699801238394655, 
                d=65.97344572538566,
                $fn=30);

    // Magnet 5
    translate([105.0 * cos(180.0), 105.0 * sin(180.0), 0])
        rotate([0, 0, 180.0])
        cylinder(h=3.0699801238394655, 
                d=65.97344572538566,
                $fn=30);

    // Magnet 6
    translate([105.0 * cos(225.0), 105.0 * sin(225.0), 0])
        rotate([0, 0, 225.0])
        cylinder(h=3.0699801238394655, 
                d=65.97344572538566,
                $fn=30);

    // Magnet 7
    translate([105.0 * cos(270.0), 105.0 * sin(270.0), 0])
        rotate([0, 0, 270.0])
        cylinder(h=3.0699801238394655, 
                d=65.97344572538566,
                $fn=30);

    // Magnet 8
    translate([105.0 * cos(315.0), 105.0 * sin(315.0), 0])
        rotate([0, 0, 315.0])
        cylinder(h=3.0699801238394655, 
                d=65.97344572538566,
                $fn=30);
}

rotor();
