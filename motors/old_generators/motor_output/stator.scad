
// Stator with coils
$fn = 100;

module coil() {
    // Coil assembly
    color("Silver") // Coil former color
    difference() {
        cylinder(h=95.57820939942327, 
                d=20.833333333333332);

        translate([0, 0, -1])
            cylinder(h=97.57820939942327, 
                    d=12.499999999999998);
    }

    // Wire winding visualization
    color("PeachPuff")
    translate([0, 0, 0.5])  // Slight offset for visibility
    difference() {
        cylinder(h=94.57820939942327, 
                d=19.833333333333332);

        translate([0, 0, -1])
            cylinder(h=97.57820939942327, 
                    d=13.499999999999998);
    }
}

module stator() {
    // Base stator disk
    difference() {
        cylinder(h=103.49487606608994, d=200);

        // Center hole
        translate([0, 0, -1])
            cylinder(h=105.49487606608994, d=10);
    }

    // Coils
    
    // Coil 1
    translate([87.53521870054244, 0.0, 7.916666666666671])
        coil();

    // Coil 2
    translate([84.55252845271576, 22.65578171691471, 7.916666666666671])
        coil();

    // Coil 3
    translate([75.80772312049642, 43.767609350271215, 7.916666666666671])
        coil();

    // Coil 4
    translate([61.89674673580105, 61.89674673580105, 7.916666666666671])
        coil();

    // Coil 5
    translate([43.76760935027123, 75.8077231204964, 7.916666666666671])
        coil();

    // Coil 6
    translate([22.65578171691471, 84.55252845271576, 7.916666666666671])
        coil();

    // Coil 7
    translate([5.359986269714142e-15, 87.53521870054244, 7.916666666666671])
        coil();

    // Coil 8
    translate([-22.65578171691472, 84.55252845271576, 7.916666666666671])
        coil();

    // Coil 9
    translate([-43.7676093502712, 75.80772312049642, 7.916666666666671])
        coil();

    // Coil 10
    translate([-61.896746735801045, 61.89674673580105, 7.916666666666671])
        coil();

    // Coil 11
    translate([-75.80772312049642, 43.767609350271215, 7.916666666666671])
        coil();

    // Coil 12
    translate([-84.55252845271575, 22.655781716914735, 7.916666666666671])
        coil();

    // Coil 13
    translate([-87.53521870054244, 1.0719972539428283e-14, 7.916666666666671])
        coil();

    // Coil 14
    translate([-84.55252845271576, -22.655781716914717, 7.916666666666671])
        coil();

    // Coil 15
    translate([-75.8077231204964, -43.76760935027123, 7.916666666666671])
        coil();

    // Coil 16
    translate([-61.89674673580106, -61.896746735801045, 7.916666666666671])
        coil();

    // Coil 17
    translate([-43.76760935027126, -75.8077231204964, 7.916666666666671])
        coil();

    // Coil 18
    translate([-22.655781716914703, -84.55252845271576, 7.916666666666671])
        coil();

    // Coil 19
    translate([-1.6079958809142425e-14, -87.53521870054244, 7.916666666666671])
        coil();

    // Coil 20
    translate([22.655781716914674, -84.55252845271578, 7.916666666666671])
        coil();

    // Coil 21
    translate([43.76760935027123, -75.8077231204964, 7.916666666666671])
        coil();

    // Coil 22
    translate([61.89674673580103, -61.89674673580106, 7.916666666666671])
        coil();

    // Coil 23
    translate([75.80772312049639, -43.76760935027126, 7.916666666666671])
        coil();

    // Coil 24
    translate([84.55252845271576, -22.655781716914706, 7.916666666666671])
        coil();
}

stator();
