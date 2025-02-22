quality = 100;

horn_diameter = 5.8;
horn_height = 2;
gear_diameter = 20;

screw_hole_diameter = 3;

horn_ring_height = 2.75;
horn_ring_diameter = 9;
horn_ring_cutout_diameter = 6;
horn_ring_screw_hole_diameter = 3;
horn_ring_screw_hole_height = 5;

module metal_horn_holes() {
    
    translate([7, 0, horn_height+.38])
        cylinder(h=horn_height*4, d=screw_hole_diameter, center=true, $fn = quality);
    
    translate([0, 7, horn_height+.38])
        cylinder(h=horn_height*4, d=screw_hole_diameter, center=true, $fn = quality);
    
    translate([-7, 0, horn_height+.38])
        cylinder(h=horn_height*4, d=screw_hole_diameter, center=true, $fn = quality);
    
    translate([0, -7, horn_height+.38])
        cylinder(h=horn_height*4, d=screw_hole_diameter, center=true, $fn = quality);
    
}

module sub_metal_horn(horn_ring=true) {
    if (horn_ring) {
        difference() {
            
            // Horn Ring
            translate([0, 0, horn_height+.38])
                cylinder(h=horn_ring_height, d=horn_ring_diameter, center=true, $fn = quality);
            
            // Horn Ring Cutout
            translate([0, 0, horn_height])
                cylinder(h=horn_ring_screw_hole_height,
                            d=horn_ring_cutout_diameter,
                            center=true, $fn = quality);
        }
    }
    
    difference() {
        // Cap 
        cylinder(h=horn_height, d=gear_diameter, center=true, $fn = quality);
        
        // Screw Holes in Cap
        metal_horn_holes();
        
        // Horn Ring Screw Hole
        translate([0, 0, 0])
            cylinder(h=10, d=horn_ring_screw_hole_diameter, center=true, $fn = quality);
        
        if (horn_ring) {
            // Horn Ring Cutout
            translate([0, 0, horn_height])
                cylinder(h=horn_ring_screw_hole_diameter,
                            d=horn_ring_cutout_diameter,
                            center=true, $fn = quality);
        }
    }

}

module metal_horn(horn_ring=true) {
    translate([0, 0, horn_height/2])
        sub_metal_horn(horn_ring);
}


show_printables = false;
if (show_printables) {
    metal_horn();
}
