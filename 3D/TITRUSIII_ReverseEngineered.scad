
include <hobbyservo_horn.scad>;

$fn = 100; // Smoothness of curves

width = 10; // Width of the linkage
thickness = 2; // width of the linkage
joint_diameter = 3; // Diameter of rotational joint

servo_arm_1_angle = 45; // Angle of the bend
servo_arm_1_length1 = 15; // Length before angle
servo_arm_1_length2 = 12; // Length after angle

servo_arm_2_angle = 45; // Angle of the bend
servo_arm_2_length1 = 52; // Length before angle
servo_arm_2_length2 = 12; // Length after angle

passive_linkage_angle = 30;
passive_linkage_length1 = 40;
passive_linkage_length2 = 15;

long_bar_length = 70;
extra_hole_driven_distance = 30;
extra_hole_passive_distance = 20;

short_bar_length = 20;

triangle_link_length_1 = short_bar_length;
triangle_link_length_2 = short_bar_length;
triangle_link_length_3 = extra_hole_passive_distance;

foot_bar_angle = 38;
foot_bar_length1 = 70;
foot_bar_length2 = 20;


// Colors
black = [0, 0, 0];
blue = [0, 0, 1];
lightblue = [0.65, 0.85, 1];
cyan = [0, 1, 1];
green = [0, 1, 0];
red = [1, 0, 0];
purple = [0.5, 0.0, 0.5];

// Module for creating a bar with customizable holes
module bar(length, width=width, hole1=true, hole2=true, thickness=thickness, extra_hole=false, extra_hole_distance=0) {
    difference() {
        // Main bar
        hull() {
            translate([0, 0, 0])
                cylinder(r=width/2, h=thickness);
            translate([length, 0, 0])
                cylinder(r=width/2, h=thickness);
        }
        // Holes for joints (conditional)
        if (hole1) {
            translate([0, 0, -1])
                cylinder(d=joint_diameter, h=thickness+2);
        }
        if (hole2) {
            translate([length, 0, -1])
                cylinder(d=joint_diameter, h=thickness+2);
        }
        if (extra_hole) {
            translate([extra_hole_distance, 0, -1])
                cylinder(d=joint_diameter, h=thickness+2);
        }
    }
}

// Module for creating a triangular link with three holes
module triangle_link_foot_old(side_length,
                            front_hole=true,
                            top_hole=true) {
    difference() {
        // Main triangular body
        hull() {
            // Top Circle
            translate([side_length/2, side_length, 0])
                cylinder(d=width, h=thickness);
           // Rear Cube
            translate([0, 0, 0])
                cube([width, width, thickness]);
            // Front Cube
            translate([side_length, 0, 0])
                cube([width*1.35, width, thickness]);
        }
        // Holes for joints
        if (front_hole) {
            translate([side_length+7, side_length*0.25+2, -1])
                cylinder(d=joint_diameter, h=thickness+2);
        }
        if (top_hole) {
            translate([side_length/2, side_length*0.866, -1])
                cylinder(d=joint_diameter, h=thickness+2);
        }
    }
}

module triangle_link_foot(side_length, front_hole=true, top_hole=true) {
    difference() {
        // Main triangular body
        hull() {
            // Top Circle
            translate([side_length / 2, side_length, 0])
                cylinder(d=width, h=thickness);
            // Rear Cube
            translate([0, 0, 0])
                cube([width, width, thickness]);
            // Front Cube
            translate([side_length, 0, 0])
                cube([width * 1.35, width, thickness]);
        }
        // Holes for joints (adjusted based on short_bar_length)
        if (front_hole) {
            translate([short_bar_length, short_bar_length * 0.25 + 2, -1])
                cylinder(d=joint_diameter, h=thickness + 2);
        }
        if (top_hole) {
            translate([short_bar_length / 2, short_bar_length * 0.866, -1])
                cylinder(d=joint_diameter, h=thickness + 2);
        }
    }
}

// Module for creating a triangular link with three holes
module triangle_link_old(side_a, side_b, side_c, rear_hole=true, front_hole=true, top_hole=true) {
    
    // Compute triangle vertex positions using the law of cosines
    function get_vertex(side1, side2, side3) = 
        let (cos_angle = (side1*side1 + side2*side2 - side3*side3) / (2 * side1 * side2))
        let (angle = acos(cos_angle))
        [side1 * cos(angle), side1 * sin(angle), 0];

    // Calculate the position of the third vertex
    vertex = get_vertex(side_c, side_a, side_b);
    
    difference() {
        // Main triangular body using hull()
        hull() {
            translate([0, 0, 0]) cylinder(d=width, h=thickness);  // Rear Circle
            translate([side_c, 0, 0]) cylinder(d=width, h=thickness);  // Front Circle
            translate(vertex) cylinder(d=width, h=thickness);  // Top Circle
        }

        // Holes for joints
        if (rear_hole) {
            translate([0, 0, -1])
                cylinder(d=joint_diameter, h=thickness+2);
        }
        if (front_hole) {
            translate([side_c, 0, -1])
                cylinder(d=joint_diameter, h=thickness+2);
        }
        if (top_hole) {
            translate(vertex - [0, 0, 1])  // Adjust for Z position
                cylinder(d=joint_diameter, h=thickness+2);
        }
    }
}

// Module for creating a triangular link with three holes
module triangle_link(side_a, side_b, side_c, rear_hole=true, front_hole=true, top_hole=true, right_triangle=false) {
    
    // Compute triangle vertex positions
    function get_vertex(side1, side2, side3, right_angle) = 
        right_angle 
        ? [side1, side2, 0] // Right triangle: vertex is simply (side_a, side_b)
        : let (cos_angle = (side1*side1 + side2*side2 - side3*side3) / (2 * side1 * side2))
          let (angle = acos(cos_angle))
          [side1 * cos(angle), side1 * sin(angle), 0];

    // Calculate the position of the third vertex
    vertex = get_vertex(side_c, side_a, side_b, right_triangle);
    
    difference() {
        // Main triangular body using hull()
        hull() {
            translate([0, 0, 0]) cylinder(d=width, h=thickness);  // Rear Circle
            translate([side_c, 0, 0]) cylinder(d=width, h=thickness);  // Front Circle
            translate(vertex) cylinder(d=width, h=thickness);  // Top Circle
        }

        // Holes for joints
        if (rear_hole) {
            translate([0, 0, -1])
                cylinder(d=joint_diameter, h=thickness+2);
        }
        if (front_hole) {
            translate([side_c, 0, -1])
                cylinder(d=joint_diameter, h=thickness+2);
        }
        if (top_hole) {
            translate(vertex - [0, 0, 1])  // Adjust for Z position
                cylinder(d=joint_diameter, h=thickness+2);
        }
    }
}

module foot_bar() {
    foot_thickness = thickness*5;
    
    bar(foot_bar_length1, hole1=false, hole2=false, thickness=foot_thickness);
    
    translate([foot_bar_length1/4, width/2, 0])
        triangle_link_foot(foot_bar_length2,
                            front_hole=true,
                            top_hole=true);
    
    translate([foot_bar_length1/4, width/2, foot_thickness-thickness])
        triangle_link_foot(foot_bar_length2,
                            front_hole=true,
                            top_hole=true);
}

module half_foot_bar_old(flip=false) {
    foot_thickness = thickness * 2;
    triangle_offset_y = width / 2;
    
    if (flip) {
        triangle_offset_x = foot_bar_length1/4 + foot_bar_length1/2;
        translate([triangle_offset_x, triangle_offset_y, thickness])
        //translate([foot_bar_length1/4, width/2, thickness])
            rotate([0, 180, 0])
            triangle_link_foot(foot_bar_length2,
                                front_hole=true,
                                top_hole=true);
    } else {
        
        triangle_offset_x = foot_bar_length1/4;
        translate([triangle_offset_x, triangle_offset_y, -thickness+2])
            triangle_link_foot(foot_bar_length2,
                                front_hole=true,
                                top_hole=true);
    }

    // Main bar
    bar(foot_bar_length1, hole1=true, hole2=true, thickness=foot_thickness);
}


module fill_triangle() {
    hull() {
        // Attach bottom left to foot_bar
        translate([-width/2, -width/2, 0])
            #cylinder(d=width, h=thickness);

        // Attach bottom right of foot_bar
        translate([foot_bar_length1-width*3-8, 0, 0])
            #cylinder(d=width/8, h=thickness);

        // Attach near top of short_bar (avoiding holes)
        translate([short_bar_length - 17.25, short_bar_length+9, 0])
            #cylinder(d=width/8, h=thickness);
    }
}

module half_foot_bar2(flip=false) {
    foot_thickness = thickness * 2;
    
    // Offset for short_bar position
    short_bar_offset_y = width / 2;
    
    // Flip if necessary
    if (flip) {
        short_bar_offset_x = foot_bar_length1/4 + foot_bar_length1/2;
        translate([short_bar_offset_x, short_bar_offset_y, thickness])
            rotate([0, 180, 0])
            rotate([0, 0, 360-45])
            translate([-15, 25, 0])
            short_bar();
    } else {
        short_bar_offset_x = foot_bar_length1/4;
        translate([short_bar_offset_x, short_bar_offset_y, -thickness + 2])
            rotate([0, 0, 360-45])
            translate([-15, 25, 0])
            short_bar();
    }

    // Main foot bar
    bar(foot_bar_length1, hole1=true, hole2=true, thickness=foot_thickness);
}

module half_foot_bar(flip=false) {
    foot_thickness = thickness * 2;
    short_bar_offset_y = width / 2;
    
    if (flip) {
        short_bar_offset_x = foot_bar_length1 / 4 + foot_bar_length1 / 2;
        translate([short_bar_offset_x, short_bar_offset_y, thickness])
            rotate([0, 180, 0])
            rotate([0, 0, 360 - 45])
            translate([-15, 25, 0])
            //short_bar();
            bar(short_bar_length*2+(width/2),
                hole1=true, hole2=false,
                extra_hole=true, extra_hole_distance=short_bar_length);
        
        // Triangle filler for flipped side
        translate([short_bar_offset_x, short_bar_offset_y, thickness])
            rotate([0, 180, 0])
            fill_triangle();
        
    } else {
        short_bar_offset_x = foot_bar_length1 / 4;
        translate([short_bar_offset_x, short_bar_offset_y, -thickness + 2])
            rotate([0, 0, 360 - 45])
            translate([-15, 25, 0])
            //short_bar();
            bar(short_bar_length*2+(width/2),
                hole1=true, hole2=false,
                extra_hole=true, extra_hole_distance=short_bar_length);
        
        // Triangle filler for normal side
        translate([short_bar_offset_x, short_bar_offset_y, -thickness + 2])
            fill_triangle();
    }

    // Main foot bar
    bar(foot_bar_length1, hole1=true, hole2=true, thickness=foot_thickness);
}


module spacer() {
    difference() {
        cylinder(r=width/2, h=thickness);
        translate([0, 0, -1])
            cylinder(d=joint_diameter, h=thickness+2);
    }
}


module servo_arm_1() {
    // First segment with hole at start
    bar(servo_arm_1_length1, hole1=true, hole2=false);

    // Second segment rotated by angle, with hole at end
    translate([servo_arm_1_length1, 0, 0])
        rotate([0, 0, servo_arm_1_angle])
            bar(servo_arm_1_length2, hole1=false, hole2=true);
}

module servo_arm_2() {
    // First segment with hole at start
    bar(servo_arm_2_length1, hole1=true, hole2=false);

    // Second segment rotated by angle, with hole at end
    translate([servo_arm_2_length1, 0, 0])
        rotate([0, 0, servo_arm_2_angle])
            bar(servo_arm_2_length2, hole1=false, hole2=true);
}

module passive_linkage() {
    // First segment with hole at start
    bar(passive_linkage_length1, hole1=true, hole2=false);

    // Second segment rotated by angle, with hole at end
    translate([passive_linkage_length1, 0, 0])
        rotate([0, 0, passive_linkage_angle])
            bar(passive_linkage_length2, hole1=false, hole2=true);
}

module long_bar(extra_hole=false, extra_hole_distance=0) {
    // First segment with hole at start
    bar(long_bar_length, hole1=true, hole2=true, extra_hole=extra_hole, extra_hole_distance=extra_hole_distance);
}

module bar4_driven() {
    long_bar(extra_hole=true, extra_hole_distance=extra_hole_driven_distance);
}

module bar4_passive() {
    long_bar(extra_hole=true, extra_hole_distance=extra_hole_passive_distance);
}

module short_bar() {
    // First segment with hole at start
    bar(short_bar_length, hole1=true, hole2=true);
}

module printable_set() {
    spacing = 24;
    
    translate([0, 0, 0]) color(blue)
        metal_horn(horn_ring=false);
    translate([0, 0, 0]) color(blue)
        servo_arm_1();
    
    translate([0, spacing, 0]) color(blue)
        metal_horn(horn_ring=false);
    translate([0, spacing, 0]) color(blue)
        servo_arm_2();
    
    translate([0, spacing*2, 0]) color(green)
        long_bar();
    
    translate([0, spacing*3-5, 0]) color(green)
        long_bar();
    
    translate([spacing*4, spacing*3-15, 0]) color(green)
    rotate([0, 0, 90])
        long_bar();
    
    translate([0, spacing*4-10, 0]) color(green)
        bar4_driven();
    
    translate([0, spacing*5-15, 0]) color(lightblue)
        bar4_passive();
    
        
    translate([0, spacing*6-20, 0]) color(lightblue)
        triangle_link(side_a=triangle_link_length_1,
                        side_b=triangle_link_length_2,
                        side_c=triangle_link_length_3,
                        rear_hole=true, front_hole=true, top_hole=true,
                        right_triangle=true);
        
    translate([0, spacing*7-5, 0]) color(purple)
        passive_linkage();
        
    translate([spacing*4-15, spacing*3, 0]) color(lightblue)
    rotate([0, 0, 90])
        short_bar();
 
    /*
    translate([spacing*7, 0, width/2]) color(blue)
        rotate([90, 0, 0])
        foot_bar();
    */
    
    translate([spacing*2, 0, 0]) color(blue)
        half_foot_bar();
    translate([spacing*2, spacing*6, 0]) color(blue)
        half_foot_bar(flip=true);
        
    translate([spacing*3, spacing*5, 0]) color(blue)
        spacer();
    translate([spacing*3-11, spacing*5, 0]) color(blue)
        spacer();
    translate([spacing*3, spacing*5+11, 0]) color(blue)
        spacer();
    translate([spacing*3-11, spacing*5+11, 0]) color(blue)
        spacer();
}

show_printable = true;

if (show_printable) {
    printable_set();
}