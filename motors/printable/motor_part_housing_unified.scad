
// Motor Housing
module housing() {
    difference() {
        union() {
            // Main cylinder
            difference() {
                cylinder(h=28.0, r=39.0, 
                        center=false, $fn=100);
                translate([0, 0, -2.0])
                cylinder(h=26.0, 
                        r=37.0, 
                        center=false, $fn=100);
            }

            // Bottom mounting points
            
            rotate([0, 0, 0.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 30.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 60.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 90.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 120.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 150.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 180.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 210.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 240.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 270.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 300.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);

            rotate([0, 0, 330.0])
            translate([33.15, 0, 0])
            cylinder(h=2.0, r=5.85, 
                    center=false, $fn=100);
        }

        // Shaft hole
        cylinder(h=29.0, r=12.2,
                center=false, $fn=100);
    }
}

housing();