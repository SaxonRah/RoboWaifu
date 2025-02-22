/**
 * Servo outline library
 *
 * Authors:
 *   - Eero 'rambo' af Heurlin 2010-
 *
 * License: LGPL 2.1
 */

use <MCAD/triangles.scad>

/**
 * TowerPro SG90 servo
 *
 * @param vector position The position vector
 * @param vector rotation The rotation vector
 * @param boolean screws If defined then "screws" will be added and when the module is differenced() from something if will have holes for the screws
 * @param boolean cables If defined then "cables" output will be added and when the module is differenced() from something if will have holes for the cables output
 * @param number axle_length If defined this will draw a red indicator for the main axle
 */
module towerprosg90(position=undef, rotation=undef, screws = 0, axle_length = 0, cables=0)
{
	translate(position) rotate(rotation) {
        difference(){
            union()
            {
                translate([-5.9,-11.8/2,0]) cube([22.5,11.8,22.7]);
                translate([0,0,22.7-0.1]){
                    cylinder(d=11.8,h=4+0.1);
                    hull(){
                        translate([8.8-5/2,0,0]) cylinder(d=5,h=4+0.1);
                        cylinder(d=5,h=4+0.1);
                    }
                    translate([0,0,4]) cylinder(d=4.6,h=3.2);
                }
                translate([-4.7-5.9,-11.8/2,15.9]) cube([22.5+4.7*2, 11.8, 2.5]); 
            }
            //screw holes
            translate([-2.3-5.9,0,15.9+1.25]) cylinder(d=2,h=5, center=true);
            translate([-2.3-5.9-2,0,15.9+1.25]) cube([3,1.3,5], center=true);
            translate([2.3+22.5-5.9,0,15.9+1.25]) cylinder(d=2,h=5, center=true);
            translate([2.3+22.5-5.9+2,0,15.9+1.25]) cube([3,1.3,5], center=true);
        }
        if (axle_length > 0) {
            color("red", 0.3) translate([0,0,29.9/2]) cylinder(r=1, h=29.9+axle_length, center=true);
        }
        if (cables > 0) color("red", 0.3) translate([-12.4,-1.8,4.5]) cube([10,3.6,1.2]);
        if(screws > 0) color("red", 0.3) {
            translate([-2.3-5.9,0,15.9+1.25]) cylinder(d=2,h=10, center=true);
            translate([2.3+22.5-5.9,0,15.9+1.25]) cylinder(d=2,h=10, center=true);
        }
    }
    
}

/**
 * Futaba S3003 servo
 *
 * @param vector position The position vector
 * @param vector rotation The rotation vector
 */
module futabas3003(position, rotation)
{
	translate(position)
	{
		rotate(rotation)
	    {
			union()
			{
				// Box and ears
				translate([0,0,0])
				{
					cube([20.1, 39.9, 36.1], false);
					translate([1.1, -7.6, 26.6])
					{
                        difference() {
						    cube([18, 7.6, 2]);
                            translate([4, 3.5, -0.1])
                                cylinder(100, 2);
                            translate([14, 3.5, -0.1])
                                cylinder(100, 2);
                        }
					}

					translate([1.1, 39.9, 26.6])
					{
                        difference() {
                            cube([18, 7.6, 2.5]);
                            translate([4, 4.5, -0.1])
                                cylinder(100, 2);
                            translate([14, 4.5, -0.1])
                                cylinder(100, 2);
                        }
                    }
				}

				// Main axle
				translate([10, 30, 36.1])
				{
					cylinder(r=6, h=0.4, $fn=30);
					cylinder(r=2.5, h=4.9, $fn=20);
				}
			}
		}
	}
}


translate([0, 0, 0]) 
futabas3003(position=[0, 0, 0], rotation=[0, 0, 0]);

translate([50, 16.6, 0]) 
rotate([0, 0, 270])
towerprosg90(position=[0, 0, 0], rotation=[0, 0, 0]);
