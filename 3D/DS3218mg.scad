// DS3218MG Servo Motor OpenSCAD Model
// All dimensions in mm

// Main body dimensions
body_length = 40.5;
body_width = 20.5;
body_height = 41;
body_bump_height = 42;

// Mounting ears dimensions
mount_ears_length = 55;
mount_ears_width = 10;
mount_ears_height = 3.5;
mount_holes_distance = 49.5;
mount_ears_offset_height = 28;
mount_ears_offset_width = (mount_ears_length - body_length) / 2;

// Output shaft housing dimensions
shaft_dia = 6;
shaft_height = 4;
shaft_housing_height = 1;
shaft_housing_dia = 13;
shaft_offset = 10;

// Wire
wire_length = 4;
wire_width = 7;
wire_height = 5;

module servo_body() {
    difference() {
        union() {
            // Main body
            color("pink")
            cube([body_length, body_width, body_height]);
            
            // Mounting ears
            color("blue")
            translate([-mount_ears_offset_width, 0, mount_ears_offset_height])
            cube([mount_ears_length, body_width, mount_ears_height]);
            
            // Output shaft housing
            color("purple")
            translate([10, body_width/2, body_height])
            cylinder(h=shaft_housing_height, d=shaft_housing_dia, $fn=32);
            
            // Output shaft
            color("green")
            translate([10, body_width/2, body_height+shaft_housing_height])
            cylinder(h=shaft_height, d=shaft_dia, $fn=32);
        }
        
        // Mounting holes
        // Left hole
        translate([-4.5, 5, mount_ears_offset_height-1])
        cylinder(h=mount_ears_height+2, d=3, $fn=20);
        translate([-4.5, 5+mount_ears_width, mount_ears_offset_height-1])
        cylinder(h=mount_ears_height+2, d=3, $fn=20);
        
        // Right hole
        translate([44, 5, mount_ears_offset_height-1])
        cylinder(h=mount_ears_height+2, d=3, $fn=20);
        translate([44, 5+mount_ears_width, mount_ears_offset_height-1])
        cylinder(h=mount_ears_height+2, d=3, $fn=20);
    }
    
    // Wire exit
    color("grey")
    translate([0, body_width/2-wire_width/2, 3.75])
    rotate([0, -90, 0])
    cube([wire_length, wire_width, wire_height]);
    translate([-wire_length, body_width/2-wire_width/2+0.5, 3.75+wire_height/5])
    rotate([0, -90, 0])
    cube([wire_length-2, wire_width-1, wire_height]);
}

// Render the servo
servo_body();