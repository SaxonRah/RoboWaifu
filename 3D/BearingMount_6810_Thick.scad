// 65mm x 50mm x 7mm Bearing Mount
$fn=150;

// Bearing
bearing_od=65;
bearing_id=50;
bearing_thickness = 7;

// General
mount_thickness = 7;

// Bearing Outer Mount
outer_mount_od=66;
outer_mount_id=58;
mounting_ear_od=6;
mounting_ear_id=2;
mounting_ear_distance = 68/2; // Half distance from center
mounting_ear_fillet_size = 32; // Size of the fillet transition

// Bearing Inner Mount
inner_mount_od=54;
inner_mount_id=8;
inner_mount_mounting_distance = 23; // Half distance from center
inner_mount_mounting_diameter = 2;
inner_mount_servo_mounting_diameter = 1.5;
inner_mount_servo_mounting_distance_1 = 9;
inner_mount_servo_mounting_distance_2 = 13;
inner_mount_extra_x_distance = 10;
inner_mount_extra_y_distance = 5;
inner_mount_extra_diameter = 3;
inner_mount_wire_cutout_round_diameter = 8;
inner_mount_wire_cutout_width = 29;
inner_mount_wire_cutout_diameter = 36;

// Standoffs
standoff_od=3;
standoff_id=2;
standoff_height = 7;


module bearing() {
    difference() {
        cylinder(h = bearing_thickness, d=bearing_od, center = true);
        cylinder(h = bearing_thickness+2, d=bearing_id, center = true);
    }
}

module bearing_outer_mount() {
    bearing_outer_mount_thickness = mount_thickness / 2;
    

    difference() {
        cylinder(h = bearing_outer_mount_thickness, d=outer_mount_id, center = true);
        cylinder(h = bearing_outer_mount_thickness+2, d=inner_mount_od, center = true);
    }
    
    // Main difference for all cuts
    difference() {
        // All solid geometry in a union
        union() {
            // Main ring
            cylinder(h = bearing_outer_mount_thickness, d=outer_mount_od, center = true);
            
            // Add ears and fillets
            for (angle = [0:90:270]) {
                translate([mounting_ear_distance * cos(angle),
                          mounting_ear_distance * sin(angle),
                          0]) {
                    // Main ear cylinder
                    cylinder(h = bearing_outer_mount_thickness, d=mounting_ear_od, center = true);
                }
                
                // Create fillet using hull()
                hull() {
                    // Point on the ring where the ear connects
                    translate([(outer_mount_od/2 - mounting_ear_fillet_size/2) * cos(angle),
                             (outer_mount_od/2 - mounting_ear_fillet_size/2) * sin(angle),
                             0])
                        cylinder(h = bearing_outer_mount_thickness, d=mounting_ear_fillet_size, center = true);
                    
                    // Point on the ear slightly inward from its edge
                    translate([(mounting_ear_distance - mounting_ear_od/4) * cos(angle),
                             (mounting_ear_distance - mounting_ear_od/4) * sin(angle),
                             0])
                        cylinder(h = bearing_outer_mount_thickness, d=mounting_ear_od, center = true);
                }
            }
        }
        
        // All cuts in one difference
        union() {
            // Center hole
            cylinder(h = bearing_outer_mount_thickness+2, d=outer_mount_id, center = true);
            
            // Ear holes
            for (angle = [0:90:270]) {
                translate([mounting_ear_distance * cos(angle),
                          mounting_ear_distance * sin(angle),
                          0])
                    cylinder(h = bearing_outer_mount_thickness+2, d=mounting_ear_id, center = true);
            }
        }
    }
}

module wire_semi_circle_cutout() {
    translate([0, -4, 0])
    difference() {
        cylinder(h = mount_thickness+1, d=inner_mount_wire_cutout_diameter, center = true);
        translate([0, 6, 0])
            cube([inner_mount_wire_cutout_diameter, 30, mount_thickness*2], center = true);
    }
}

module sub_bearing_inner_mount() {
    difference() {
        // Inner Mount
        cylinder(h = mount_thickness, d=inner_mount_od, center = true);
        cylinder(h = mount_thickness+2, d=inner_mount_id, center = true);
        
        // Inner Mount Mounting Holes
        //for (angle = [0:90:270]) {
        for (angle = [45:90:315]) {
                translate([inner_mount_mounting_distance * cos(angle),
                          inner_mount_mounting_distance * sin(angle),
                          0])
                    cylinder(h = mount_thickness+2, d=inner_mount_mounting_diameter, center = true);
        }
        
        // Extra Holes
        translate([inner_mount_extra_x_distance, inner_mount_extra_y_distance, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_extra_diameter, center = true);
        translate([-inner_mount_extra_x_distance, inner_mount_extra_y_distance, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_extra_diameter, center = true);
        translate([inner_mount_extra_x_distance, -inner_mount_extra_y_distance, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_extra_diameter, center = true);
        translate([-inner_mount_extra_x_distance, -inner_mount_extra_y_distance, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_extra_diameter, center = true);
        
        // Servo Horn Mounting Holes
        translate([inner_mount_servo_mounting_distance_1, 0, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_servo_mounting_diameter, center = true);
        translate([inner_mount_servo_mounting_distance_2, 0, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_servo_mounting_diameter, center = true);
        translate([-inner_mount_servo_mounting_distance_1, 0, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_servo_mounting_diameter, center = true);
        translate([-inner_mount_servo_mounting_distance_2, 0, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_servo_mounting_diameter, center = true);
        
        // Wire Cutout Hole
        translate([12, -11, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_wire_cutout_round_diameter, center = true);
        translate([-12, -11, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_wire_cutout_round_diameter, center = true);
        translate([0, -11, 0])
            cube([25, 8, mount_thickness*2], center = true);
        translate([0, -16, 0])
            cylinder(h = mount_thickness+2, d=inner_mount_wire_cutout_round_diameter, center = true);
    }
}


module sub_bearing_inner_mount_solid() {
    difference() {
        // Inner Mount
        cylinder(h = mount_thickness+mount_thickness, d=inner_mount_od-4.25);
        cylinder(h = mount_thickness*5, d=inner_mount_id, center = true);
        
        // Inner Mount Mounting Holes
        //for (angle = [0:90:270]) {
        for (angle = [45:90:315]) {
                translate([inner_mount_mounting_distance * cos(angle),
                          inner_mount_mounting_distance * sin(angle),
                          0])
                    cylinder(h = mount_thickness*5, d=inner_mount_mounting_diameter, center = true);
        }
        
        // Servo Horn Mounting Holes
        translate([inner_mount_servo_mounting_distance_1, 0, 0])
            cylinder(h = mount_thickness*5, d=inner_mount_servo_mounting_diameter, center = true);
        translate([inner_mount_servo_mounting_distance_2, 0, 0])
            cylinder(h = mount_thickness*5, d=inner_mount_servo_mounting_diameter, center = true);
        translate([-inner_mount_servo_mounting_distance_1, 0, 0])
            cylinder(h = mount_thickness*5, d=inner_mount_servo_mounting_diameter, center = true);
        translate([-inner_mount_servo_mounting_distance_2, 0, 0])
            cylinder(h = mount_thickness*5, d=inner_mount_servo_mounting_diameter, center = true);
        
    }
}

module bearing_inner_mount() {
    //difference() {
        sub_bearing_inner_mount_solid();
        //wire_semi_circle_cutout();
    //}
}

module standoff() {
    difference() {
        cylinder(h = standoff_height, d=standoff_od, center = true);
        cylinder(h = standoff_height+2, d=standoff_id, center = true);
    }
}

module standoffs() {
    union() {
        // Ear holes
        for (angle = [0:90:270]) {
            translate([mounting_ear_distance * cos(angle),
                      mounting_ear_distance * sin(angle),
                      0])
            standoff();
        }

        // Inner Mount 
        for (angle = [45:90:315]) {
            translate([inner_mount_mounting_distance * cos(angle),
                      inner_mount_mounting_distance * sin(angle),
                      0])
            standoff();
        }
    }
}

module full_bearing_mount_assembly() {
    bearing_outer_mount();
    bearing_inner_mount();
    
    bearing_spacing = bearing_thickness/2 + mount_thickness/2;
    outer_mount_spacing = mount_thickness + bearing_thickness;
    inner_mount_spacing = outer_mount_spacing;
    standoff_spacing = bearing_spacing;
    
    translate([0, 0, bearing_spacing])
        bearing();
    
    translate([0, 0, outer_mount_spacing])
        bearing_outer_mount();
    
    translate([0, 0, inner_mount_spacing])
        bearing_inner_mount();
    
    translate([0, 0, standoff_spacing])
        standoffs();
    
}


module printable_set() {
    mount_spacing = mount_thickness/2;
    standoff_spacing = standoff_height/2;
    
    translate([0, 0, mount_spacing])
        bearing_outer_mount();
    translate([75, 0, mount_spacing])
        bearing_outer_mount();
    
    translate([0, 0, mount_spacing])
        bearing_inner_mount();
    translate([75, 0, mount_spacing])
        bearing_inner_mount();
    
    translate([5, -10, standoff_spacing])
        standoff();    
    translate([-5, -10, standoff_spacing])
        standoff();    
    translate([10, -10, standoff_spacing])
        standoff();    
    translate([-10, -10, standoff_spacing])
        standoff();    
    translate([0, -10, standoff_spacing])
        standoff();    
    translate([0, -15, standoff_spacing])
        standoff();    
    translate([5, -15, standoff_spacing])
        standoff();    
    translate([-5, -15, standoff_spacing])
        standoff();    
}

module printable_set_plates() {
    mount_spacing = mount_thickness/2;
    bearing_outer_mount_spacing = mount_thickness/4;
    standoff_spacing = standoff_height/2;
    
    translate([0, 0, bearing_outer_mount_spacing])
        bearing_outer_mount();
    translate([0, 0, 0])
        bearing_inner_mount();
    
    translate([75, 0, bearing_outer_mount_spacing])
        bearing_outer_mount();
    
    //translate([0, 0, mount_spacing+mount_thickness/2])
        //#bearing();
    
}

//full_bearing_mount_assembly();
//printable_set();

printable_set_plates();

    
