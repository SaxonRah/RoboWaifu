
// Printable Rotor Plate
module rotor() {
    difference() {
        // Main plate
        cylinder(h=8.2, r=35.0, 
                center=false, $fn=100);

        // Center hole
        translate([0, 0, -1])
        cylinder(h=10.2, r=15.0, 
                center=false, $fn=100);

        // Magnet recesses
        union() {
            
                rotate([0, 0, 0.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 30.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 60.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 90.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 120.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 150.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 180.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 210.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 240.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 270.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 300.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);

                rotate([0, 0, 330.0])
                translate([25.0, 0, 4.999999999999999])
                cylinder(h=(3.2+1), d=10.2, 
                        center=false, $fn=100);
        }
    }
}

rotor();