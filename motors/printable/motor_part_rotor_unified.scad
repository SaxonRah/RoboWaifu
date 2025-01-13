
// Printable Rotor Plate
module rotor() {
    difference() {
        // Main plate
        cylinder(h=7.2, r=35.0, 
                center=false, $fn=100);

        // Center hole
        translate([0, 0, -1])
        cylinder(h=9.2, r=15.0, 
                center=false, $fn=100);

        // Magnet recesses
        union() {
            
                rotate([0, 0, 0.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 30.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 60.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 90.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 120.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 150.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 180.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 210.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 240.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 270.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 300.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);

                rotate([0, 0, 330.0])
                translate([25.0, 0, 5.0])
                cylinder(h=(2.2+1), d=10.671975511965979, 
                        center=false, $fn=100);
        }
    }
}

rotor();