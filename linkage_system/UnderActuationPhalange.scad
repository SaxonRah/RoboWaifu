// Configuration parameters
proximal_phalange_length = 30;
middle_phalange_length = proximal_phalange_length * 0.75;
distal_phalange_length = proximal_phalange_length * 0.5;

proximal_phalange_width = 15;
middle_phalange_width = proximal_phalange_width * 0.9;
distal_phalange_width = proximal_phalange_width * 0.8;

phalange_height = 10;

joint_radius = 4;
bellcrank_size = 8;
bellcrank_thickness = 3;
pin_radius = 1;
pin_height = 12;
linkage_width = 3;

module phalange_arm() {
    difference() {
        cube([bellcrank_thickness*2, bellcrank_thickness, bellcrank_size]);
        // Pin hole for bell crank connection
        translate([bellcrank_thickness, bellcrank_thickness+1, bellcrank_size/2])
        rotate([90, 0, 0])
        cylinder(r=pin_radius*1.2, h=bellcrank_thickness*2, $fn=32);
    }
}

// Helper module for creating a tapered phalange with bell crank arm
module phalange(length, width, height, taper=0.8) {
    union() {
        hull() {
            cube([2, width, height]);
            translate([length-2, width*taper/2, 0])
            cube([2, width*(1-taper/2), height]);
        }

        // Bell crank attachment arm - moved to top and properly aligned
        translate([0, width+8, 0]) {
        rotate([90, 0, 0]){
        phalange_arm();
            }
        }
    }
}

// Enhanced bell crank with configurable angles
module bellcrank(size, thickness, angle1=0, angle2=90) {
    difference() {
        union() {
            // Center hub
            cylinder(r=size/2.5, h=thickness, $fn=32);

            // First arm with end cap
            rotate([0, 0, angle1]) {
                translate([-size/3, 0, 0])
                cube([size/1.5, size, thickness]);

                // End cap
                translate([0, size, 0])
                cylinder(r=size/3, h=thickness, $fn=32);
            }

            // Second arm with end cap
            rotate([0, 0, angle2]) {
                translate([-size/3, 0, 0])
                cube([size/1.5, size, thickness]);

                // End cap
                translate([0, size, 0])
                cylinder(r=size/3, h=thickness, $fn=32);
            }
        }

        // Pin hole at center
        translate([0, 0, -1])
        cylinder(r=pin_radius*1.2, h=thickness+2, $fn=32);

        // Connection holes at ends
        rotate([0, 0, angle1])
        translate([0, size, -1])
        cylinder(r=pin_radius*1.2, h=thickness+2, $fn=32);

        rotate([0, 0, angle2])
        translate([0, size, -1])
        cylinder(r=pin_radius*1.2, h=thickness+2, $fn=32);
    }
}

// Enhanced linkage bar with offset option
module linkage_bar(length, offset=0) {
    difference() {
        union() {
            translate([0, -linkage_width/2, 0])
            cube([length, linkage_width, 2]);

            cylinder(r=linkage_width/2, h=2, $fn=32);
            translate([length, 0, 0])
            cylinder(r=linkage_width/2, h=2, $fn=32);
        }

        // Pin holes at ends
        translate([0, 0, -1])
        cylinder(r=pin_radius*1.2, h=4, $fn=32);
        translate([length, 0, -1])
        cylinder(r=pin_radius*1.2, h=4, $fn=32);
    }
}

// Proximal assembly with dual bell cranks
module proximal_assembly() {
    color("LightBlue")
    phalange(proximal_phalange_length, proximal_phalange_width, phalange_height, 0.9);

    // Primary bell crank (for middle phalange)
    translate([proximal_phalange_length/2,
              proximal_phalange_width/2,
              phalange_height + bellcrank_size])
    rotate([0, 180, 0])
    color("Red")
    bellcrank(bellcrank_size, bellcrank_thickness, -45, 45);

    // Secondary bell crank (for distal phalange)
    translate([proximal_phalange_length/2 + bellcrank_size*1.5,
              proximal_phalange_width/2,
              phalange_height + bellcrank_size])
    rotate([0, 180, 30])
    color("Orange")
    bellcrank(bellcrank_size, bellcrank_thickness, -45, 45);

    // Linkage bars
    // Primary linkage for middle phalange control
    translate([proximal_phalange_length/2,
              proximal_phalange_width/2 + bellcrank_size,
              phalange_height + bellcrank_size])
    rotate([0, 0, 90])
    color("Gray")
    linkage_bar(proximal_phalange_length * 0.4);

    // Secondary linkage for distal phalange control
    translate([proximal_phalange_length/2 + bellcrank_size*1.5,
              proximal_phalange_width/2 + bellcrank_size,
              phalange_height + bellcrank_size])
    rotate([0, 0, 120])
    color("DarkGray")
    linkage_bar(proximal_phalange_length * 0.5);
}

// Middle assembly with single bell crank
module middle_assembly() {
    color("LightGreen")
    phalange(middle_phalange_length, middle_phalange_width, phalange_height, 0.85);

    // Middle joint bell crank
    translate([middle_phalange_length/2,
              middle_phalange_width/2,
              phalange_height + bellcrank_size])
    rotate([0, 180, 0])
    color("Yellow")
    bellcrank(bellcrank_size, bellcrank_thickness, -45, 45);

    // Linkage for distal control
    translate([middle_phalange_length/2,
              middle_phalange_width/2 + bellcrank_size,
              phalange_height + bellcrank_size])
    rotate([0, 0, 90])
    color("Gray")
    linkage_bar(middle_phalange_length * 0.4);
}

// Complete assembly
module finger_assembly() {
    proximal_assembly();

    // Position middle phalange
    translate([proximal_phalange_length, 0, 0])
    middle_assembly();

    // Position distal phalange
    translate([proximal_phalange_length + middle_phalange_length, 0, 0])
    color("Pink")
    phalange(distal_phalange_length, distal_phalange_width, phalange_height, 0.7);
}

// Render the complete finger
finger_assembly();
