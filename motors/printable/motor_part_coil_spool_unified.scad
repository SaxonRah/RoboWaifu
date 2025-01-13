
// Main Coil Spool Body
module spool_body() {
    difference() {
        union() {
            // Center core
            cylinder(h=10.066666666666666, r=3.25, 
                    center=false, $fn=100);

            // Bottom flange only
            cylinder(h=2.0, r=6.5, 
                    center=false, $fn=100);
        }

        // Center hole for mounting
        cylinder(h=11.066666666666666, r=1.625, 
                center=false, $fn=100);
    }
}

// Simple Cap (matching core diameter)
module spool_cap() {
    difference() {
        cylinder(h=2.0, r=6.5, 
                center=false, $fn=100);

        // Center hole matching core
        cylinder(h=3.0, r=3.25, 
                center=false, $fn=100);
    }
}

// Define copper coil appearance for assembly
module copper_coil() {
    color("Orange", 0.8) {
        spool_body();
        translate([0, 0, 10.066666666666666])
        spool_cap();
    }
}

// For individual part export
if ($preview) {
    // Preview assembled
    copper_coil();
} else {
    // Export separate parts
    spool_body();
    translate([16.25, 0, 0])
    spool_cap();
}