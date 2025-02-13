// Assembly of the leg mechanism

// Include the linkage parts
include <TITRUSIII_ReverseEngineered.scad>;

module leg_assembly() {
    
    translate([0, 0, 0]) color(blue)
        foot_bar();
    
    translate([0, 0, 0]) color(green)
        long_bar();
    
    translate([0, 0, 0]) color(green)
        long_bar();
    
    translate([0, 0, 0]) color(green)
        bar4_driven();
    
    translate([0, 0, 0]) color(lightblue)
        bar4_passive();
    
    translate([0, 0, 0]) color(lightblue)
        short_bar();
    
    translate([0, 0, 0]) color(purple)
        passive_linkage();
        
    translate([0, 0, 0]) color(blue)
        servo_arm_1();
    
    translate([0, 0, 0]) color(blue)
        servo_arm_2();
}

// Render the assembled leg
leg_assembly();