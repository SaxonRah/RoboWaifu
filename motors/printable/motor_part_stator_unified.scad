
// Printable Stator Plate
module stator() {
    difference() {
        // Main plate
        cylinder(h=10.0, r=35.0, 
                center=false, $fn=100);

        // Center hole
        translate([0, 0, -1])
        cylinder(h=12.0, r=15.0, 
                center=false, $fn=100);

        // Coil slots
        union() {
            
                rotate([0, 0, 0.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 20.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 40.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 60.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 80.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 100.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 120.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 140.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 160.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 180.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 200.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 220.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 240.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 260.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 280.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 300.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 320.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);

                rotate([0, 0, 340.0])
                translate([25.0, 0, 10.0-(13.2-1)])
                cylinder(h=13.2, r=3.1333333333333333, 
                        center=false, $fn=100);
        }
    }
}

stator();