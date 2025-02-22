import json
import numpy as np


def load_motor_config(filename):
    """Loads motor configurations from the given file."""
    with open(filename, "r") as file:
        lines = file.readlines()
        configurations = [json.loads(line.strip().replace("'", '"')) for line in lines[1:] if line.strip()]
    return configurations


def calculate_motor_dimensions(config):
    """Calculates dimensions for each motor component based on the configuration."""
    D_motor = config["D_motor (mm)"]  # Overall motor diameter
    T_motor = config["T_motor (mm)"]  # Motor thickness
    pole_pairs = config["Pole Pairs (count)"]
    magnet_count = pole_pairs * 2  # Each pole pair has 2 magnets
    coil_count = magnet_count  # Assuming one coil per magnet

    # Rotor dimensions
    rotor_outer_dia = D_motor
    rotor_inner_dia = D_motor * 0.3  # Example: Inner cutout for mounting
    rotor_thickness = T_motor * 0.5  # Half the total thickness

    # Stator dimensions
    stator_outer_dia = D_motor * 0.95  # Slightly smaller than rotor
    stator_inner_dia = D_motor * 0.4  # Example mounting cutout
    stator_thickness = T_motor * 0.5  # Half the total thickness

    # Magnet dimensions
    magnet_arc_angle = 360 / magnet_count  # Angle for each magnet
    magnet_width = (np.pi * D_motor / magnet_count) * 0.8  # Spacing factor
    magnet_height = rotor_thickness * 0.8  # Example height ratio

    # Coil dimensions
    coil_slot_width = magnet_width * 0.9  # Slightly smaller than magnet width
    coil_slot_depth = stator_thickness * 0.8  # Example depth ratio

    # Housing dimensions
    housing_outer_dia = D_motor * 1.1  # Slightly larger than motor
    housing_thickness = T_motor * 0.2  # Example housing thickness

    # Mounting dimensions
    mounting_hole_radius = rotor_inner_dia * 0.5  # Example: Half inner diameter
    mounting_hole_count = 4  # Standard four mounting holes

    return {
        "Rotor": {
            "outer_diameter": rotor_outer_dia,
            "inner_diameter": rotor_inner_dia,
            "thickness": rotor_thickness,
            "magnet_cutouts": magnet_count
        },
        "Stator": {
            "outer_diameter": stator_outer_dia,
            "inner_diameter": stator_inner_dia,
            "thickness": stator_thickness,
            "coil_slots": coil_count
        },
        "Magnets": {
            "count": magnet_count,
            "arc_angle": magnet_arc_angle,
            "width": magnet_width,
            "height": magnet_height
        },
        "Coils": {
            "count": coil_count,
            "slot_width": coil_slot_width,
            "slot_depth": coil_slot_depth
        },
        "Housing": {
            "outer_diameter": housing_outer_dia,
            "thickness": housing_thickness
        },
        "Mounting": {
            "hole_radius": mounting_hole_radius,
            "hole_count": mounting_hole_count
        }
    }


def save_dimensions_to_json(dimensions, output_filename):
    """Saves the calculated dimensions as a JSON file."""
    with open(output_filename, "w") as json_file:
        json.dump(dimensions, json_file, indent=4)


def main():
    input_filename = "valid_motor_configurations.txt"
    output_filename = "motor_dimensions.json"

    configs = load_motor_config(input_filename)
    if not configs:
        print("No valid configurations found.")
        return

    selected_config = configs[0]  # Pick the first valid config
    dimensions = calculate_motor_dimensions(selected_config)
    save_dimensions_to_json(dimensions, output_filename)

    print(f"Motor dimensions saved to {output_filename}")


if __name__ == "__main__":
    main()