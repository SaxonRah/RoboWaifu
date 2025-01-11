import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class GearType(Enum):
    SPUR = 1
    PLANETARY = 2
    HARMONIC = 3


@dataclass
class ThermalConstraints:
    max_temperature: float  # Celsius
    ambient_temperature: float  # Celsius
    thermal_resistance: float  # °C/W
    copper_resistivity: float  # Ω⋅m
    temperature_coefficient: float  # per °C


@dataclass
class GearParameters:
    type: GearType
    ratio: float
    base_efficiency: float
    efficiency_per_stage: List[float]  # Stage-specific efficiencies
    max_ratio_per_stage: float
    min_ratio_per_stage: float
    max_stages: int
    size_factor: float  # Relative size compared to motor diameter
    backlash: float  # degrees


@dataclass
class MotorVariant:
    outer_diameter: float  # meters
    inner_diameter: float  # meters
    poles: int
    magnets: int
    turns_per_coil: int
    wire_diameter: float  # meters
    stack_length: float  # meters
    max_current_density: float  # A/m^2
    flux_density: float  # Tesla
    air_gap: float  # meters
    thermal_constraints: ThermalConstraints


class SortCriteria(Enum):
    SIZE = "size"
    EFFICIENCY = "efficiency"
    TORQUE_MARGIN = "torque_margin"
    TEMPERATURE = "temperature"


# Define thermal constraints for different motor sizes
THERMAL_CONSTRAINTS = {
    'small': ThermalConstraints(
        max_temperature=85.0,
        ambient_temperature=25.0,
        thermal_resistance=2.5,
        copper_resistivity=1.68e-8,
        temperature_coefficient=0.00393
    ),
    'medium': ThermalConstraints(
        max_temperature=100.0,
        ambient_temperature=25.0,
        thermal_resistance=2.0,
        copper_resistivity=1.68e-8,
        temperature_coefficient=0.00393
    ),
    'large': ThermalConstraints(
        max_temperature=120.0,
        ambient_temperature=25.0,
        thermal_resistance=1.5,
        copper_resistivity=1.68e-8,
        temperature_coefficient=0.00393
    )
}

# Define gear parameters with stage-specific efficiencies
GEAR_TYPES = {
    GearType.SPUR: GearParameters(
        type=GearType.SPUR,
        ratio=5.0,
        base_efficiency=0.98,
        efficiency_per_stage=[0.98, 0.97, 0.96],  # Efficiency decreases with each stage
        max_ratio_per_stage=8.0,
        min_ratio_per_stage=1.5,
        max_stages=3,
        size_factor=1.2,
        backlash=0.1
    ),
    GearType.PLANETARY: GearParameters(
        type=GearType.PLANETARY,
        ratio=7.0,
        base_efficiency=0.95,
        efficiency_per_stage=[0.95, 0.94, 0.93],
        max_ratio_per_stage=10.0,
        min_ratio_per_stage=3.0,
        max_stages=3,
        size_factor=1.1,
        backlash=0.05
    ),
    GearType.HARMONIC: GearParameters(
        type=GearType.HARMONIC,
        ratio=100.0,
        base_efficiency=0.80,
        efficiency_per_stage=[0.80],  # Single stage only
        max_ratio_per_stage=160.0,
        min_ratio_per_stage=30.0,
        max_stages=1,
        size_factor=1.3,
        backlash=0.02
    )
}

# Define motor variants with thermal constraints
MOTOR_VARIANTS = [
    MotorVariant(
        outer_diameter=0.12,
        inner_diameter=0.05,
        poles=12,
        magnets=14,
        turns_per_coil=150,
        wire_diameter=0.8e-3,
        stack_length=0.08,
        max_current_density=6e6,
        flux_density=1.4,
        air_gap=0.5e-3,
        thermal_constraints=THERMAL_CONSTRAINTS['medium']
    ),
    MotorVariant(
        outer_diameter=0.08,
        inner_diameter=0.03,
        poles=8,
        magnets=10,
        turns_per_coil=120,
        wire_diameter=0.6e-3,
        stack_length=0.06,
        max_current_density=7e6,  # Higher current density for smaller motor
        flux_density=1.4,
        air_gap=0.4e-3,
        thermal_constraints=THERMAL_CONSTRAINTS['small']
    ),
    MotorVariant(
        outer_diameter=0.15,
        inner_diameter=0.06,
        poles=16,
        magnets=18,
        turns_per_coil=200,
        wire_diameter=1.0e-3,
        stack_length=0.10,
        max_current_density=5e6,  # Lower current density for better thermal management
        flux_density=1.4,
        air_gap=0.6e-3,
        thermal_constraints=THERMAL_CONSTRAINTS['large']
    ),
    MotorVariant(
        outer_diameter=0.18,
        inner_diameter=0.08,
        poles=20,
        magnets=24,
        turns_per_coil=250,
        wire_diameter=1.2e-3,
        stack_length=0.12,
        max_current_density=4.5e6,  # Even lower current density for largest motor
        flux_density=1.4,
        air_gap=0.7e-3,
        thermal_constraints=THERMAL_CONSTRAINTS['large']
    )
]


def validate_inputs(load_mass_kg: float, desired_precision_deg: float, safety_factor: float) -> None:
    """Validate input parameters."""
    if load_mass_kg <= 0:
        raise ValidationError(f"Load mass must be positive, got {load_mass_kg}")
    if desired_precision_deg <= 0:
        raise ValidationError(f"Desired precision must be positive, got {desired_precision_deg}")
    if safety_factor < 1:
        raise ValidationError(f"Safety factor must be >= 1, got {safety_factor}")


def calculate_thermal_limits(
        motor: MotorVariant,
        current: float,
        wire_length: float
) -> Tuple[float, float]:
    """
    Calculate temperature rise and power dissipation with temperature-dependent resistivity.

    Args:
        motor: MotorVariant object containing motor specifications
        current: Operating current in Amperes
        wire_length: Total length of wire in meters

    Returns:
        Tuple[float, float]: (final_temperature, power_dissipation)
    """
    if motor.wire_diameter <= 0:
        raise ValidationError("Wire diameter must be positive")
    if current < 0:
        raise ValidationError("Current cannot be negative")
    if wire_length <= 0:
        raise ValidationError("Wire length must be positive")

    tc = motor.thermal_constraints
    wire_area = math.pi * (motor.wire_diameter / 2) ** 2

    # Iterative solution for temperature-dependent resistance
    temp = tc.ambient_temperature
    max_iterations = 20
    tolerance = 0.1  # °C
    total_power_loss = 0

    for _ in range(max_iterations):
        # Calculate resistance at current temperature
        resistance = tc.copper_resistivity * (1 + tc.temperature_coefficient * (temp - 20)) * wire_length / wire_area

        # Calculate power dissipation
        power_dissipation = current ** 2 * resistance

        # Add core losses (simplified approximation)
        core_loss_coefficient = 2.5  # W/kg for typical electrical steel at 1T, 50Hz
        core_volume = math.pi * (motor.outer_diameter ** 2 - motor.inner_diameter ** 2) / 4 * motor.stack_length
        core_density = 7650  # kg/m^3 for electrical steel
        core_mass = core_volume * core_density
        core_losses = core_loss_coefficient * core_mass

        total_power_loss = power_dissipation + core_losses

        # Calculate new temperature
        new_temp = tc.ambient_temperature + total_power_loss * tc.thermal_resistance

        # Check convergence
        if abs(new_temp - temp) < tolerance:
            return new_temp, total_power_loss

        temp = new_temp

    warnings.warn("Thermal calculation did not converge")
    return temp, total_power_loss


def calculate_gear_stages(
        required_ratio: float,
        gear_params: GearParameters,
        precision: int = 2,
        ratio_tolerance: float = 0.02  # Allow 2% deviation from target ratio
) -> Tuple[List[float], float, str]:
    """
    Calculate optimal gear stages with improved ratio matching and error handling.

    Args:
        required_ratio: Target gear ratio
        gear_params: GearParameters object containing gear specifications
        precision: Decimal places for ratio rounding
        ratio_tolerance: Acceptable deviation from target ratio (as fraction)

    Returns:
        Tuple[List[float], float, str]: (stage_ratios, actual_ratio, status_message)
    """

    def is_ratio_acceptable(actual: float, target: float, tolerance: float) -> bool:
        """Check if actual ratio is within tolerance of target ratio."""
        return abs(actual - target) / target <= tolerance

    def get_stage_options(ratio: float, max_ratio: float) -> List[int]:
        """Calculate possible number of stages for achieving target ratio."""
        return list(range(1, gear_params.max_stages + 1))

    # Input validation
    if required_ratio <= 0:
        return [], 0.0, "Invalid required ratio: must be positive"

    # Check if single stage is possible
    if gear_params.min_ratio_per_stage <= required_ratio <= gear_params.max_ratio_per_stage:
        return [round(required_ratio, precision)], required_ratio, "Single stage solution found"

    # Try different numbers of stages
    best_solution = None
    min_error = float('inf')
    status = "No valid solution found"

    for num_stages in get_stage_options(required_ratio, gear_params.max_ratio_per_stage):
        # Calculate ideal ratio per stage
        ratio_per_stage = required_ratio ** (1 / num_stages)

        if not (gear_params.min_ratio_per_stage <= ratio_per_stage <= gear_params.max_ratio_per_stage):
            continue

        # Try different roundings of the ratio_per_stage
        for round_precision in range(precision, precision + 2):
            rounded_ratio = round(ratio_per_stage, round_precision)

            # Calculate last stage separately to minimize error
            if num_stages > 1:
                intermediate_stages = [rounded_ratio] * (num_stages - 1)
                intermediate_product = math.prod(intermediate_stages)
                last_stage = required_ratio / intermediate_product

                if not (gear_params.min_ratio_per_stage <= last_stage <= gear_params.max_ratio_per_stage):
                    continue

                stages = intermediate_stages + [round(last_stage, precision)]
            else:
                stages = [rounded_ratio]

            actual_ratio = math.prod(stages)
            error = abs(actual_ratio - required_ratio)

            if error < min_error:
                min_error = error
                best_solution = stages
                status = "Optimal solution found"

    if best_solution is None:
        # Try to find nearest achievable ratio if exact solution impossible
        min_achievable = gear_params.min_ratio_per_stage ** gear_params.max_stages
        max_achievable = gear_params.max_ratio_per_stage ** gear_params.max_stages

        if required_ratio < min_achievable:
            status = f"Required ratio too low. Minimum achievable: {min_achievable:.2f}"
        elif required_ratio > max_achievable:
            status = f"Required ratio too high. Maximum achievable: {max_achievable:.2f}"

        return [], 0.0, status

    actual_ratio = math.prod(best_solution)
    if is_ratio_acceptable(actual_ratio, required_ratio, ratio_tolerance):
        return best_solution, actual_ratio, status
    else:
        return [], 0.0, f"Best solution {actual_ratio:.2f}:1 exceeds tolerance of {ratio_tolerance * 100}%"


def validate_gear_solution(
        stages: List[float],
        actual_ratio: float,
        required_ratio: float,
        gear_params: GearParameters
) -> Dict[str, Any]:
    """
    Validate a gear stage solution and provide detailed diagnostics.

    Args:
        stages: List of gear ratios for each stage
        actual_ratio: Achieved total ratio
        required_ratio: Target ratio
        gear_params: GearParameters object

    Returns:
        Dict containing validation results and diagnostics
    """
    diagnostics = {
        "valid": True,
        "issues": [],
        "stage_analysis": [],
        "error_percentage": abs(actual_ratio - required_ratio) / required_ratio * 100
    }

    # Validate number of stages
    if len(stages) > gear_params.max_stages:
        diagnostics["valid"] = False
        diagnostics["issues"].append(f"Too many stages: {len(stages)} > {gear_params.max_stages}")

    # Analyze each stage
    for i, ratio in enumerate(stages, 1):
        stage_info = {
            "stage": i,
            "ratio": ratio,
            "within_limits": True,
            "issues": []
        }

        if ratio < gear_params.min_ratio_per_stage:
            stage_info["within_limits"] = False
            stage_info["issues"].append(f"Ratio {ratio:.2f} below minimum {gear_params.min_ratio_per_stage}")

        if ratio > gear_params.max_ratio_per_stage:
            stage_info["within_limits"] = False
            stage_info["issues"].append(f"Ratio {ratio:.2f} above maximum {gear_params.max_ratio_per_stage}")

        if not stage_info["within_limits"]:
            diagnostics["valid"] = False

        diagnostics["stage_analysis"].append(stage_info)

    return diagnostics


def calculate_motor_specs_with_gearing(
        load_mass_kg: float,
        desired_precision_deg: float,
        motor_variant: MotorVariant,
        gear_type: GearType,
        safety_factor: float = 1.5
) -> Dict[str, Any]:
    """Calculate motor specifications with enhanced validation and thermal considerations."""

    try:
        validate_inputs(load_mass_kg, desired_precision_deg, safety_factor)

        # Basic torque calculations
        required_torque = load_mass_kg * 9.81 * 0.1
        design_torque = required_torque * safety_factor
        gear_params = GEAR_TYPES[gear_type]

        if desired_precision_deg <= gear_params.backlash:
            raise ValidationError(
                f"Desired precision {desired_precision_deg} must be greater than gear backlash {gear_params.backlash}."
            )

        # Calculate minimum gear ratio with backlash consideration
        electrical_steps = 360 / (motor_variant.poles * 3)
        min_gear_ratio = math.ceil(
            electrical_steps / (desired_precision_deg - gear_params.backlash)
        )

        # Calculate gear stages
        gear_stages, actual_ratio, status = calculate_gear_stages(min_gear_ratio, gear_params)
        if not gear_stages:
            raise ValidationError(f"Gear stage calculation failed: {status}")

        validation = validate_gear_solution(gear_stages, actual_ratio, min_gear_ratio, gear_params)
        if not validation["valid"]:
            raise ValidationError(f"Invalid gear solution: {validation['issues']}")

        total_gear_ratio = math.prod(gear_stages)

        # Calculate stage-specific efficiencies
        stage_efficiencies = gear_params.efficiency_per_stage[:len(gear_stages)]
        total_efficiency = math.prod(stage_efficiencies)

        # Calculate wire length and resistance
        mean_turn_length = math.pi * (motor_variant.outer_diameter + motor_variant.inner_diameter) / 2
        total_wire_length = mean_turn_length * motor_variant.turns_per_coil * motor_variant.poles

        # Current calculation with thermal validation
        wire_area = math.pi * (motor_variant.wire_diameter / 2) ** 2
        max_current = motor_variant.max_current_density * wire_area

        final_temp, power_loss = calculate_thermal_limits(motor_variant, max_current, total_wire_length)

        if final_temp > motor_variant.thermal_constraints.max_temperature:
            # Reduce current to meet thermal constraints
            temp_factor = math.sqrt(
                (motor_variant.thermal_constraints.max_temperature -
                 motor_variant.thermal_constraints.ambient_temperature) /
                (final_temp - motor_variant.thermal_constraints.ambient_temperature)
            )
            max_current *= temp_factor

        # Torque calculation
        torque_constant = (
                (motor_variant.poles * motor_variant.turns_per_coil * motor_variant.flux_density * math.pi *
                 (motor_variant.outer_diameter ** 2 - motor_variant.inner_diameter ** 2) / 4 *
                 (motor_variant.outer_diameter + motor_variant.inner_diameter) / 4 * motor_variant.stack_length
                 ) / (2 * math.pi)
        )

        motor_torque = torque_constant * max_current
        output_torque = motor_torque * total_gear_ratio * total_efficiency

        return {
            "Motor Specifications": {
                "Outer Diameter (mm)": motor_variant.outer_diameter * 1000,
                "Inner Diameter (mm)": motor_variant.inner_diameter * 1000,
                "Stack Length (mm)": motor_variant.stack_length * 1000,
                "Stator Poles": motor_variant.poles,
                "Rotor Magnets": motor_variant.magnets,
                "Air Gap (mm)": motor_variant.air_gap * 1000,
                "Turns per Coil": motor_variant.turns_per_coil,
                "Wire Diameter (mm)": motor_variant.wire_diameter * 1000,
                "Operating Temperature (°C)": final_temp,
                "Power Loss (W)": power_loss
            },
            "Gear Specifications": {
                "Type": gear_type.name,
                "Number of Stages": len(gear_stages),
                "Stage Ratios": [f"{ratio:.2f}:1" for ratio in gear_stages],
                "Stage Efficiencies": [f"{eff:.1%}" for eff in stage_efficiencies],
                "Total Ratio": f"{total_gear_ratio:.2f}:1",
                "Total Efficiency": f"{total_efficiency:.1%}",
                "Backlash (deg)": gear_params.backlash,
                "Overall Size Factor": gear_params.size_factor * motor_variant.outer_diameter * 1000
            },
            "Performance": {
                "Required Torque (N·m)": required_torque,
                "Motor Torque (N·m)": motor_torque,
                "Output Torque (N·m)": output_torque,
                "Torque Margin": f"{(output_torque / design_torque - 1):.1%}",
                "Position Resolution (deg)": electrical_steps / total_gear_ratio + gear_params.backlash,
                "Current (A)": max_current
            },
            "Validation": {
                "Meets Torque Requirement": output_torque >= design_torque,
                "Meets Precision Requirement": (
                                                       electrical_steps / total_gear_ratio + gear_params.backlash
                                               ) <= desired_precision_deg,
                "Meets Thermal Requirement": final_temp <= motor_variant.thermal_constraints.max_temperature
            }
        }
    except Exception as e:
        raise ValidationError(f"Error in motor calculations: {str(e)}")


def process_combination(args):
    motor_variant, gear_type, load_mass_kg, desired_precision_deg, safety_factor = args
    try:
        return calculate_motor_specs_with_gearing(
            load_mass_kg,
            desired_precision_deg,
            motor_variant,
            gear_type,
            safety_factor
        )
    except ValidationError:
        return None


def find_optimal_combinations(
        load_mass_kg: float,
        desired_precision_deg: float,
        sort_by: SortCriteria = SortCriteria.SIZE,
        safety_factor: float = 1.5,
        max_workers: int = 4
) -> List[Dict]:
    """Find optimal motor and gear combinations using parallel processing."""

    # Generate all possible combinations
    combo_args = [
        (motor, gear_type, load_mass_kg, desired_precision_deg, safety_factor)
        for motor in MOTOR_VARIANTS
        for gear_type in GearType
    ]

    # Process combinations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_combination, combo_args)

    # Filter valid combinations
    # combinations = [
    #     result for result in results
    #     if result is not None and all(result["Validation"].values())
    # ]

    combinations = [
        result for result in results
        if result is not None and all(result["Validation"].values()) and result["Gear Specifications"][
            "Stage Ratios"] != []
    ]

    if not combinations:
        print("No valid combinations found.")
        print("Diagnosing possible issues...")
        # for motor, gear_type in combo_args:
        for motor, gear_type, load_mass_kg, desired_precision_deg, safety_factor in combo_args:
            print(f"Testing Motor: {motor.outer_diameter * 1000:.1f}mm, Gear Type: {gear_type.name}")
            try:
                spec = calculate_motor_specs_with_gearing(
                    load_mass_kg,
                    desired_precision_deg,
                    motor,
                    gear_type,
                    safety_factor
                )
                print("Result:", spec["Validation"])
            except ValidationError as e:
                print("Validation Error:", str(e))

    # Sort based on criteria
    sort_keys = {
        SortCriteria.SIZE: lambda x: x["Gear Specifications"]["Overall Size Factor"],
        SortCriteria.EFFICIENCY: lambda x: -float(x["Gear Specifications"]["Total Efficiency"].rstrip('%')),
        SortCriteria.TORQUE_MARGIN: lambda x: -float(x["Performance"]["Torque Margin"].rstrip('%')),
        SortCriteria.TEMPERATURE: lambda x: x["Motor Specifications"]["Operating Temperature (°C)"]
    }

    return sorted(combinations, key=sort_keys[sort_by])


def print_results(combinations: List[Dict], format_type: str = "table"):
    """Print results in a formatted manner."""
    if not combinations:
        print("No valid combinations found.")
        return

    if format_type == "table":
        headers = ["Option", "Motor Size", "Gear Type", "Total Ratio", "Efficiency", "Temperature", "Torque Margin"]
        rows = []

        for i, combo in enumerate(combinations, 1):
            rows.append([
                i,
                f"{combo['Motor Specifications']['Outer Diameter (mm)']:.1f}mm",
                combo['Gear Specifications']['Type'],
                combo['Gear Specifications']['Total Ratio'],
                combo['Gear Specifications']['Total Efficiency'],
                f"{combo['Motor Specifications']['Operating Temperature (°C)']:.1f}°C",
                combo['Performance']['Torque Margin']
            ])

        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        for i, combo in enumerate(combinations, 1):
            print(f"\nOption {i}:")
            for category, values in combo.items():
                print(f"\n{category}:")
                for key, value in values.items():
                    print(f"{key}: {value}")
            print("\n" + "=" * 50)


# Example usage
if __name__ == "__main__":
    try:
        optimal_combinations = find_optimal_combinations(
            load_mass_kg=20,
            desired_precision_deg=0.11,
            sort_by=SortCriteria.EFFICIENCY
        )
        print_results(optimal_combinations, format_type="table")
    except ValidationError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
