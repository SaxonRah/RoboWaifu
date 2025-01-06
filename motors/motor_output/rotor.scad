
// Rotor plate with magnets
$fn = 100;

module rotor() {
    difference() {
        // Main disk
        cylinder(h=11.88549476816575, d=200);

        // Center hole
        translate([0, 0, -1])
            cylinder(h=13.88549476816575, d=10);
    }

    // Magnets
    
    // Magnet 1
    translate([80.0 * cos(0.0), 80.0 * sin(0.0), 0])
        rotate([0, 0, 0.0])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 2
    translate([80.0 * cos(11.25), 80.0 * sin(11.25), 0])
        rotate([0, 0, 11.25])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 3
    translate([80.0 * cos(22.5), 80.0 * sin(22.5), 0])
        rotate([0, 0, 22.5])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 4
    translate([80.0 * cos(33.75), 80.0 * sin(33.75), 0])
        rotate([0, 0, 33.75])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 5
    translate([80.0 * cos(45.0), 80.0 * sin(45.0), 0])
        rotate([0, 0, 45.0])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 6
    translate([80.0 * cos(56.25), 80.0 * sin(56.25), 0])
        rotate([0, 0, 56.25])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 7
    translate([80.0 * cos(67.5), 80.0 * sin(67.5), 0])
        rotate([0, 0, 67.5])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 8
    translate([80.0 * cos(78.75), 80.0 * sin(78.75), 0])
        rotate([0, 0, 78.75])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 9
    translate([80.0 * cos(90.0), 80.0 * sin(90.0), 0])
        rotate([0, 0, 90.0])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 10
    translate([80.0 * cos(101.25), 80.0 * sin(101.25), 0])
        rotate([0, 0, 101.25])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 11
    translate([80.0 * cos(112.5), 80.0 * sin(112.5), 0])
        rotate([0, 0, 112.5])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 12
    translate([80.0 * cos(123.75), 80.0 * sin(123.75), 0])
        rotate([0, 0, 123.75])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 13
    translate([80.0 * cos(135.0), 80.0 * sin(135.0), 0])
        rotate([0, 0, 135.0])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 14
    translate([80.0 * cos(146.25), 80.0 * sin(146.25), 0])
        rotate([0, 0, 146.25])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 15
    translate([80.0 * cos(157.5), 80.0 * sin(157.5), 0])
        rotate([0, 0, 157.5])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 16
    translate([80.0 * cos(168.75), 80.0 * sin(168.75), 0])
        rotate([0, 0, 168.75])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 17
    translate([80.0 * cos(180.0), 80.0 * sin(180.0), 0])
        rotate([0, 0, 180.0])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 18
    translate([80.0 * cos(191.25), 80.0 * sin(191.25), 0])
        rotate([0, 0, 191.25])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 19
    translate([80.0 * cos(202.5), 80.0 * sin(202.5), 0])
        rotate([0, 0, 202.5])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 20
    translate([80.0 * cos(213.75), 80.0 * sin(213.75), 0])
        rotate([0, 0, 213.75])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 21
    translate([80.0 * cos(225.0), 80.0 * sin(225.0), 0])
        rotate([0, 0, 225.0])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 22
    translate([80.0 * cos(236.25), 80.0 * sin(236.25), 0])
        rotate([0, 0, 236.25])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 23
    translate([80.0 * cos(247.5), 80.0 * sin(247.5), 0])
        rotate([0, 0, 247.5])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 24
    translate([80.0 * cos(258.75), 80.0 * sin(258.75), 0])
        rotate([0, 0, 258.75])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 25
    translate([80.0 * cos(270.0), 80.0 * sin(270.0), 0])
        rotate([0, 0, 270.0])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 26
    translate([80.0 * cos(281.25), 80.0 * sin(281.25), 0])
        rotate([0, 0, 281.25])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 27
    translate([80.0 * cos(292.5), 80.0 * sin(292.5), 0])
        rotate([0, 0, 292.5])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 28
    translate([80.0 * cos(303.75), 80.0 * sin(303.75), 0])
        rotate([0, 0, 303.75])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 29
    translate([80.0 * cos(315.0), 80.0 * sin(315.0), 0])
        rotate([0, 0, 315.0])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 30
    translate([80.0 * cos(326.25), 80.0 * sin(326.25), 0])
        rotate([0, 0, 326.25])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 31
    translate([80.0 * cos(337.5), 80.0 * sin(337.5), 0])
        rotate([0, 0, 337.5])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);

    // Magnet 32
    translate([80.0 * cos(348.75), 80.0 * sin(348.75), 0])
        rotate([0, 0, 348.75])
        cylinder(h=3.968828101499084, 
                d=12.566370614359172,
                $fn=30);
}

rotor();
