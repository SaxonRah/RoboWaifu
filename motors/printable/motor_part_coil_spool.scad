
// Coil Winding Spool
module coil_spool() {
    difference() {
        union() {
            // Center core
            cylinder(h=12.0, r=3.25, center=true, $fn=100);

            // Bottom flange
            translate([0, 0, -6.0])
            cylinder(h=2.0, r=6.5, center=false, $fn=100);

            // Top flange
            translate([0, 0, 4.0])
            cylinder(h=2.0, r=6.5, center=false, $fn=100);
        }

        // Center hole for mounting
        cylinder(h=13.0, r=1.625, center=true, $fn=100);
    }
}

// Define copper coil appearance
module copper_coil() {
    color("Orange", 0.8)
    coil_spool();
}

copper_coil();
