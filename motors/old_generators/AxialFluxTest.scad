// Axial Flux Motor and Gear Assembly Generator
// Parameters are in millimeters unless otherwise specified

/* [Motor Parameters] */
// Motor outer diameter
motor_outer_diameter = 120; // [50:200]
// Motor inner diameter
motor_inner_diameter = 50; // [20:100]
// Motor stack length (total thickness)
motor_stack_length = 80; // [40:150]
// Number of stator poles
stator_poles = 12; // [4:24]
// Number of rotor magnets
rotor_magnets = 14; // [4:28]
// Air gap in mm
air_gap = 0.5; // [0.3:0.1:1.0]
// Stator thickness
stator_thickness = 15; // [5:30]
// Rotor disc thickness
rotor_thickness = 8; // [5:20]

/* [Gear Parameters] */
// Gear type (1=SPUR, 2=PLANETARY, 3=HARMONIC)
gear_type = 1; // [1:3]
// Number of gear stages
num_stages = 2; // [1:3]
// Stage ratios (comma-separated list)
stage_ratios = "5,4"; 
// Overall size factor
size_factor = 1.2; // [1.0:0.1:2.0]

/* [Visualization] */
// Show cross-section
show_cross_section = true;
// Show gears
show_gears = true;
// Show dimensions
show_dimensions = true;
// Show winding pattern
show_windings = true;

// Constants
$fn = 100;

module stator_disc(od, id, thickness, poles) {
    difference() {
        cylinder(h=thickness, d=od, center=true);
        cylinder(h=thickness+1, d=id, center=true);
        
        // Stator slots for windings
        for(i = [0:poles-1]) {
            rotate([0, 0, i * (360/poles)])
            translate([(od+id)/4, 0, 0])
            union() {
                // Core slot
                cube([od/6, od/20, thickness+1], center=true);
                // Winding area
                if (show_windings) {
                    color("Copper")
                    translate([0, 0, thickness/4])
                    rotate([0, 90, 0])
                    cylinder(h=od/6, d=thickness/2, center=true);
                }
            }
        }
    }
}

module rotor_disc(od, id, thickness, magnets) {
    difference() {
        union() {
            cylinder(h=thickness, d=od, center=true);
            // Back iron
            translate([0, 0, -thickness/4])
            cylinder(h=thickness/2, d=od, center=true);
        }
        cylinder(h=thickness+1, d=id, center=true);
        
        // Magnet slots
        for(i = [0:magnets-1]) {
            rotate([0, 0, i * (360/magnets)])
            translate([(od+id)/4, 0, thickness/4]) {
                // Magnet pocket
                cube([od/8, od/12, thickness/2+1], center=true);
                // Show magnets
                color("DarkGray")
                cube([od/8-1, od/12-1, thickness/2], center=true);
            }
        }
    }
}

module spur_gear(ratio, diameter) {
    height = diameter * 0.3;
    tooth_height = diameter * 0.1;
    num_teeth = round(ratio * 10);
    
    difference() {
        union() {
            cylinder(h=height, d=diameter, center=true);
            for(i = [0:num_teeth-1]) {
                rotate([0, 0, i * (360/num_teeth)])
                translate([diameter/2, 0, 0])
                cube([tooth_height, 2, height], center=true);
            }
        }
        cylinder(h=height+1, d=diameter*0.3, center=true);
    }
}

module planetary_gear_stage(ratio, diameter) {
    height = diameter * 0.25;
    
    // Ring gear
    difference() {
        cylinder(h=height, d=diameter, center=true);
        cylinder(h=height+1, d=diameter*0.8, center=true);
    }
    
    // Planet gears
    for(i = [0:2]) {
        rotate([0, 0, i * 120])
        translate([diameter*0.3, 0, 0])
        cylinder(h=height, d=diameter*0.2, center=true);
    }
    
    // Sun gear
    cylinder(h=height, d=diameter*0.15, center=true);
}

module harmonic_drive(diameter) {
    height = diameter * 0.2;
    
    // Outer ring
    difference() {
        cylinder(h=height, d=diameter, center=true);
        cylinder(h=height+1, d=diameter*0.9, center=true);
    }
    
    // Flexspline
    difference() {
        cylinder(h=height*0.8, d=diameter*0.85, center=true);
        cylinder(h=height*0.8+1, d=diameter*0.75, center=true);
    }
}

module gear_assembly() {
    stage_list = split(stage_ratios, ",");
    current_offset = motor_stack_length;
    current_diameter = motor_outer_diameter * size_factor;
    
    for(i = [0:num_stages-1]) {
        stage_ratio = float(stage_list[i]);
        translate([0, 0, current_offset + current_diameter*0.15]) {
            if(gear_type == 1) {
                spur_gear(stage_ratio, current_diameter);
            } else if(gear_type == 2) {
                planetary_gear_stage(stage_ratio, current_diameter);
            } else {
                harmonic_drive(current_diameter);
            }
        }
        current_offset = current_offset + current_diameter*0.3;
        current_diameter = current_diameter * 0.8;
    }
}

module motor_assembly() {
    total_thickness = stator_thickness + 2*rotor_thickness + 2*air_gap;
    
    // Bottom rotor
    color("Silver")
    translate([0, 0, -total_thickness/2 + rotor_thickness/2])
    rotor_disc(motor_outer_diameter, motor_inner_diameter, rotor_thickness, rotor_magnets);
    
    // Stator
    color("Gray")
    stator_disc(motor_outer_diameter, motor_inner_diameter, stator_thickness, stator_poles);
    
    // Top rotor
    color("Silver")
    translate([0, 0, total_thickness/2 - rotor_thickness/2])
    mirror([0, 0, 1])
    rotor_disc(motor_outer_diameter, motor_inner_diameter, rotor_thickness, rotor_magnets);
    
    // Housing
    color("DarkGray", 0.3) {
        difference() {
            cylinder(h=total_thickness, d=motor_outer_diameter + 10, center=true);
            cylinder(h=total_thickness + 1, d=motor_outer_diameter, center=true);
            // Ventilation holes
            for(i = [0:11]) {
                rotate([0, 0, i * 30])
                translate([motor_outer_diameter/2 + 2, 0, 0])
                cylinder(h=total_thickness + 1, d=5, center=true);
            }
        }
    }
}

// Main assembly
module main_assembly() {
    if(show_cross_section) {
        difference() {
            union() {
                motor_assembly();
                if(show_gears) gear_assembly();
            }
            translate([-500, 0, -500])
            cube([1000, 1000, 1000]);
        }
    } else {
        motor_assembly();
        if(show_gears) gear_assembly();
    }
    
    // Dimension lines
    if(show_dimensions) {
        color("Red") {
            // Motor diameter
            translate([0, motor_outer_diameter/2 + 10, 0])
            rotate([90, 0, 0])
            cylinder(h=motor_outer_diameter + 20, d=0.5, center=true);
            
            // Motor thickness
            translate([motor_outer_diameter/2 + 10, 0, 0])
            rotate([0, 90, 0])
            cylinder(h=motor_stack_length + 20, d=0.5, center=true);
        }
    }
}

main_assembly();