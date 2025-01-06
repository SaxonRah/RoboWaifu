
// Generated Electric Motor Model
// Motor Parameters
DC_Bus_Voltage = 24;
Base_Speed = 700;
Rated_Current = 7;
Wire_Diameter = 0.65;
p = 12;
c = 18;
ro = 65;
ri = 35;
B = 0.5;
Acoil = 0.000492;
Arotor = 0.00942;


// Additional parameters
motor_length = 50;
shaft_radius = 10;
gap = 0.5;
slot_depth = 15;
slot_width = 5;


// Main assembly
union() {
    
// Rotor
color("silver")
difference() {
    cylinder(h=motor_length, r=ri, center=true, $fn=72);
    // Shaft hole
    cylinder(h=motor_length + 1, r=shaft_radius, center=true, $fn=72);
}

    
// Stator
color("gray")
difference() {
    cylinder(h=motor_length, r=ro, center=true, $fn=72);
    cylinder(h=motor_length + 1, r=ri + gap, center=true, $fn=72);
    
    rotate([0, 0, 0.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 20.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 40.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 60.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 80.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 100.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 120.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 140.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 160.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 180.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 200.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 220.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 240.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 260.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 280.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 300.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 320.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);

    rotate([0, 0, 340.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cube([slot_depth, slot_width, motor_length + 1], center=true);
}

    // Windings
    union() {
        
    // Winding 1
    color("red")
    rotate([0, 0, 0.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 2
    color("green")
    rotate([0, 0, 20.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 3
    color("blue")
    rotate([0, 0, 40.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 4
    color("red")
    rotate([0, 0, 60.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 5
    color("green")
    rotate([0, 0, 80.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 6
    color("blue")
    rotate([0, 0, 100.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 7
    color("red")
    rotate([0, 0, 120.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 8
    color("green")
    rotate([0, 0, 140.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 9
    color("blue")
    rotate([0, 0, 160.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 10
    color("red")
    rotate([0, 0, 180.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 11
    color("green")
    rotate([0, 0, 200.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 12
    color("blue")
    rotate([0, 0, 220.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 13
    color("red")
    rotate([0, 0, 240.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 14
    color("green")
    rotate([0, 0, 260.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 15
    color("blue")
    rotate([0, 0, 280.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 16
    color("red")
    rotate([0, 0, 300.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 17
    color("green")
    rotate([0, 0, 320.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);

    // Winding 18
    color("blue")
    rotate([0, 0, 340.0])
    translate([ri + slot_depth/2 + gap, 0, 0])
    cylinder(h=motor_length - 2, r=slot_width/3, center=true, $fn=16);
    }
}
