// Constants for the design
$fn = 50;  // Smoothness of circles
thickness = 5;  // Thickness of all parts
joint_radius = 3;  // Radius of the joint holes
bar_width = 15;  // Width of the bars

// Colors
black = [0, 0, 0];
blue = [0, 0, 1];
cyan = [0, 1, 1];
green = [0, 1, 0];
red = [1, 0, 0];

// Module for creating a bar with holes at the ends
module bar(length, width=bar_width) {
    difference() {
        // Main bar
        hull() {
            translate([0, 0, 0])
                cylinder(r=width/2, h=thickness);
            translate([length, 0, 0])
                cylinder(r=width/2, h=thickness);
        }
        // Holes for joints
        translate([0, 0, -1])
            cylinder(r=joint_radius, h=thickness+2);
        translate([length, 0, -1])
            cylinder(r=joint_radius, h=thickness+2);
    }
}

// Module for creating a triangular link with three holes
module triangle_link(side_length) {
    difference() {
        // Main triangular body
        hull() {
            translate([0, 0, 0])
                cylinder(r=bar_width/2, h=thickness);
            translate([side_length, 0, 0])
                cylinder(r=bar_width/2, h=thickness);
            translate([side_length/2, side_length*0.866, 0])
                cylinder(r=bar_width/2, h=thickness);
        }
        // Three holes for joints
        translate([0, 0, -1])
            cylinder(r=joint_radius, h=thickness+2);
        translate([side_length, 0, -1])
            cylinder(r=joint_radius, h=thickness+2);
        translate([side_length/2, side_length*0.866, -1])
            cylinder(r=joint_radius, h=thickness+2);
    }
}

// Module for creating a joint pin
module joint() {
    color(red)
        cylinder(r=joint_radius-0.5, h=thickness);
}

// Fixed mount 
module fixed_mount() {
    color(black)
    translate([-15, -5, 0])
        cube([10, 10, thickness]);
}

// Main assembly
translate([0, 0, 0]) {
    // Black square
    fixed_mount();
    
    // Blue link
    color(blue)
    translate([0, 0, 0])
    rotate([0, 0, 0])
        bar(80);
    
    // Cyan link
    color(cyan)
    translate([0, 0, 0])
    rotate([0, 0, 45])
        bar(40);
    
    // Green triangular link
    color(green)
    translate([60, -20, 0])
    rotate([0, 0, -30])
        triangle_link(30);
}

// Add joint pins at connection points
joint_positions = [
    [-10, 0, 0],    // Mount point
    [0, 0, 0],      // First joint
    [40, 0, 0],     // Middle joint
    [80, 0, 0],     // End of blue bar
    [60, -20, 0],   // Triangle base
    [75, -30, 0],   // Triangle right
    [65, -40, 0]    // Triangle left
];

for(pos = joint_positions) {
    translate(pos)
        joint();
}