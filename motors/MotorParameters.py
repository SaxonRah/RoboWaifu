from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class MotorParameters:
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


def create_default_parameters() -> MotorParameters:
    """Create default parameters for testing"""
    return MotorParameters(
        poles=4,  # Will be varied in calculations
        coils=6,  # Will be varied in calculations
        turns_per_coil=100,  # Will be varied in calculations
        wire_diameter=0.65,  # mm
        voltage=12.0,  # V
        max_current=10.0,  # A
        magnet_type="circle",
        magnet_width=10.0,  # mm
        magnet_length=10.0,  # mm
        magnet_thickness=3.0,  # mm
        magnet_br=1.2,  # Tesla (N42 NdFeB)
        outer_radius=50.0,  # mm
        inner_radius=10.0,  # mm
        air_gap=1.0,  # mm
        stator_thickness=15.0,  # mm
        rotor_thickness=5.0,  # mm
        target_diameter=50,  # mm
        torque=0,  # Nm
        target_torque=0.1,  # Nm
        estimated_torque=0.0,
        tolerance=0.2,  # ±20%
        efficiency=0.0,
        resistance=0.0,
        current=0.0,
        coil_width=None,
        coil_height=None,
        total_height=None
    )
