include <gears.scad>;

quality = 100;

horn_diameter = 5.8;
horn_height = 2;

gear_diameter = 20;
gear_modul = 1;
gear_tooth_number = 20;
gear_partial_cone_angle = 45;
gear_tooth_width = 5;
gear_bore = horn_diameter;
gear_pressure_angle = 20;
gear_helix_angle = 0;

module servo_bevel_gear_adapter()
{
    // Servo Horn Base
    difference() {
        cylinder(h=horn_height, d=gear_diameter, center=false, $fn = quality);
        cylinder(h=gear_diameter, d=horn_diameter, center=true, $fn = quality);
    }
    
    // Bevel Gear (using library)
    translate([0, 0, horn_height])
    bevel_gear(modul=gear_modul, tooth_number=gear_tooth_number,
        partial_cone_angle=gear_partial_cone_angle, tooth_width=gear_tooth_width,
        bore=horn_diameter, pressure_angle=gear_pressure_angle, helix_angle=gear_helix_angle);
}

module mating_bevel_gear()
{
    bevel_gear(modul=gear_modul, tooth_number=gear_tooth_number,
        partial_cone_angle=gear_partial_cone_angle, tooth_width=gear_tooth_width,
        bore=horn_diameter, pressure_angle=gear_pressure_angle, helix_angle=gear_helix_angle);
}

module place_mating_bevel_gear()
{
    // Mating Bevel Gear
    translate([-10.75, 0, 11.75]) // Adjust positioning for meshing
    rotate([45, 0, 0])
    rotate([0, 90, 0])
    mating_bevel_gear();
}

module metal_horn_holes() {
    
    translate([7, 0, horn_height+.38])
    cylinder(h=horn_height*4, d=2.5, center=true, $fn = quality);
    
    translate([0, 7, horn_height+.38])
    cylinder(h=horn_height*4, d=2.5, center=true, $fn = quality);
    
    translate([-7, 0, horn_height+.38])
    cylinder(h=horn_height*4, d=2.5, center=true, $fn = quality);
    
    translate([0, -7, horn_height+.38])
    cylinder(h=horn_height*4, d=2.5, center=true, $fn = quality);
    
}

module metal_horn() {
    difference() {
        cylinder(h=horn_height, d=gear_diameter, center=true, $fn = quality);
        translate([0, 0, horn_height+.38])
            cylinder(h=2.75, d=9, center=true, $fn = quality);
        
        metal_horn_holes();
    }
    
    difference() {
        translate([0, 0, horn_height+.38])
        #cylinder(h=2.75, d=9, center=true, $fn = quality);
        
        translate([0, 0, horn_height+.38])
        #cylinder(h=3, d=6, center=true, $fn = quality);
    }
    
}

module printable_set()
{
    servo_bevel_gear_adapter();
    // Mating Bevel Gear
    translate([25, 0, 0]) // Adjust positioning for meshing
    mating_bevel_gear();
}

module display_meshing() {
    servo_bevel_gear_adapter();
    place_mating_bevel_gear();
}

// display_meshing();
// printable_set();


//difference() {
    //servo_bevel_gear_adapter();
    // translate([0, 0, horn_height*5-1.5])
    // translate([0, 0, 1])
    // #cube([25, 25, 5], center=true);
//}
metal_horn();
