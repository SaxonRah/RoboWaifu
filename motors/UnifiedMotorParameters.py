from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class UnifiedMotorParameters:
    # Electrical Parameters
    poles: int
    coils: int
    turns_per_coil: int
    wire_diameter: float  # mm
    voltage: float  # V
    max_current: float  # A

    # Magnetic Parameters
    magnet_type: str  # 'circle' or 'square'
    magnet_width: float  # mm
    magnet_length: float  # mm
    magnet_thickness: float  # mm
    magnet_br: float  # Tesla

    # Mechanical Parameters
    outer_radius: float  # mm
    inner_radius: float  # mm
    air_gap: float  # mm
    stator_thickness: float  # mm
    rotor_thickness: float  # mm
    target_diameter: float  # mm

    # Performance Parameters
    torque: float  # Nm
    efficiency: float  # percentage
    resistance: float  # ohms
    current: float  # amps
    target_torque: float  # Nm
    tolerance: float  # ±20%
    estimated_torque: Optional[float] = None

    # Derived Geometric Parameters (calculated)
    coil_width: Optional[float] = None  # mm
    coil_height: Optional[float] = None  # mm
    total_height: Optional[float] = None  # mm

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate all parameters for physical and electrical feasibility."""
        errors = []

        # Validate geometric constraints
        if self.inner_radius >= self.outer_radius:
            errors.append("Inner radius must be less than outer radius")

        if self.air_gap <= 0:
            errors.append("Air gap must be positive")

        if self.stator_thickness <= 0:
            errors.append("Stator thickness must be positive")

        # Validate electrical parameters
        if self.voltage <= 0:
            errors.append("Voltage must be positive")

        if self.max_current <= 0:
            errors.append("Maximum current must be positive")

        # Validate magnetic parameters
        if self.magnet_br <= 0:
            errors.append("Magnet remanence must be positive")

        if self.magnet_type not in ['circle', 'square']:
            errors.append("Magnet type must be 'circle' or 'square'")

        # Validate counts
        if self.poles <= 0 or self.poles % 2 != 0:
            errors.append("Number of poles must be positive and even")

        if self.coils <= 0:
            errors.append("Number of coils must be positive")

        return len(errors) == 0, errors
