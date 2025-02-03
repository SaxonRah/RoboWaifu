import numpy as np
from itertools import product

"""
Required:
    D_motor: Overall motor diameter (m)
    T_motor: Motor thickness (m)
    T_required: Required torque (Nm)
    V_supply: Available supply voltage (V)
    RPM_target: Target operating speed (RPM)

Optional:    
    Efficiency target (η_target) (default 90%)
    Max current limit (I_max)
"""


def calculate_torque(B, A_coil, N_turns, I, r_avg, p):
    return (B * A_coil * N_turns * I * r_avg * p) / 2


def calculate_power(T_required, rpm_target, efficiency=0.9):
    omega = (2 * np.pi * rpm_target) / 60  # Convert RPM to rad/s
    return (T_required * omega) / efficiency


def calculate_voltage(B, A_coil, N_turns, p, rpm_target):
    omega = (2 * np.pi * rpm_target) / 60  # Convert RPM to rad/s
    f = (p * omega) / (2 * np.pi)  # Electrical frequency
    return B * A_coil * N_turns * f


def search_motor_configuration(D_motor_range, T_motor_range, T_required_range, V_supply_range, rpm_target_range,
                               I_max_range):
    valid_configurations = []
    best_config = None
    best_efficiency = 0

    for D_motor, T_motor, T_required, V_supply, rpm_target, I_max in product(D_motor_range, T_motor_range,
                                                                             T_required_range, V_supply_range,
                                                                             rpm_target_range, I_max_range):
        r_avg = (D_motor / 2 - 10) / 1000  # Convert mm to meters, assume 10mm coil gap
        A_coil = np.pi * (r_avg ** 2) * 0.6  # Assume 60% coil coverage

        B_values = [1.2, 1.3, 1.4]  # Tesla, different magnet strengths
        pole_counts = [8, 10, 12, 14, 16, 18, 20]  # Possible pole counts
        turn_counts = range(5, 100, 5)  # Possible turns per coil

        for B, p, N_turns in product(B_values, pole_counts, turn_counts):
            for I in np.linspace(0.1, I_max, 10):
                T = calculate_torque(B, A_coil, N_turns, I, r_avg, p)
                if T < T_required:
                    continue

                V = calculate_voltage(B, A_coil, N_turns, p, rpm_target)
                if V > V_supply:
                    continue

                efficiency = T_required / T  # Approximate efficiency

                config = {
                    "D_motor (mm)": D_motor,
                    "T_motor (mm)": T_motor,
                    "T_required (Nm)": T_required,
                    "V_supply (V)": V_supply,
                    "RPM_target": rpm_target,
                    "I_max (A)": I_max,
                    "B (T)": B,
                    "Pole Pairs (count)": p,
                    "Turns per Coil (count)": N_turns,
                    "Current Draw (A)": I,
                    "Voltage (V)": V,
                    "Torque (Nm)": T,
                    "Efficiency (%)": efficiency * 100
                }
                valid_configurations.append(config)

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_config = config

    return valid_configurations, best_config


def save_valid_configurations_to_file(valid_configurations, filename: str = "valid_motor_configurations.txt"):
    with open(filename, "w") as file:
        file.write(f"Found {len(valid_configurations)} valid motor configurations.\n")
        for config in valid_configurations:
            file.write(str(config) + "\n")


# Output
text_file_name = "valid_motor_configurations.txt"

# Example Parameter Ranges
D_motor_range = [180, 200, 220]  # 180mm to 220mm motor diameter
T_motor_range = [40, 50, 60]  # 40mm to 60mm motor thickness
T_required_range = [5, 8, 10]  # Torque range
V_supply_range = [6, 12, 24]  # Voltage supply range
rpm_target_range = [100, 500]  # RPM target range
I_max_range = [40, 50, 60]  # Max current limit range

valid_motors, best_motor = search_motor_configuration(
    D_motor_range, T_motor_range, T_required_range, V_supply_range,
    rpm_target_range, I_max_range)

save_valid_configurations_to_file(valid_motors)

print(f"Found {len(valid_motors)} valid motor configurations.")
for motor in valid_motors:
    print(motor)

print("\nBest Motor Configuration:")
print(best_motor)
