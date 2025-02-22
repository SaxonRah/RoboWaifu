
// Motor Parameters
d_motor = 200.0;
t_motor = 40;
stator_radius = 90.0;
magnet_width = 39.269908169872416;
coil_width = 35.34291735288517;
air_gap = 1.5;

// Generate Rotor
module rotor() {
    cylinder(h = t_motor/2, r = d_motor/2, center = true);
}

// Generate Stator
module stator() {
    cylinder(h = t_motor/2, r = stator_radius, center = true);
}

// Generate Magnets
module magnet() {
    cube([magnet_width, 5, 3], center=true);
}

// Generate Coils
module coil() {
    cube([coil_width, 6, 4], center=true);
}

// Assembling the Motor
translate([0, 0, -t_motor/4]) rotor();
translate([0, 0, t_motor/4 + air_gap]) stator();
    