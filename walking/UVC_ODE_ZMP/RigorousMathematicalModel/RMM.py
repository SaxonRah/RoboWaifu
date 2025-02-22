import pybullet as p
import pybullet_data
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import math
from typing import List, Tuple, TypeVar, Union, Dict
import time


# Type aliases for better readability
ArrayType = TypeVar('ArrayType', bound=npt.NDArray[Union[np.float64, np.int64]])
BlendedControls = Dict[str, ArrayType]


@dataclass
class RobotState:
    orientation: np.ndarray  # Roll, pitch, yaw
    angular_velocity: np.ndarray
    com_position: np.ndarray
    com_velocity: np.ndarray
    foot_forces: np.ndarray  # Shape (2,6) for [left,right] x [fx,fy,fz,mx,my,mz]
    joint_positions: np.ndarray  # Order: [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]
    joint_velocities: np.ndarray  # Same order as joint_positions

    def get_foot_normal_forces(self) -> np.ndarray:
        """Returns just the normal forces for both feet"""
        return self.foot_forces[:, 2]

    def get_foot_lateral_forces(self) -> np.ndarray:
        """Returns the lateral forces for both feet"""
        return self.foot_forces[:, :2]

    def get_foot_moments(self) -> np.ndarray:
        """Returns the moments for both feet"""
        return self.foot_forces[:, 3:]


class AdaptiveCPG:
    def __init__(self, num_oscillators=2):
        # Oscillator parameters
        self.num_oscillators = num_oscillators
        self.phase = np.zeros(num_oscillators)
        self.amplitude = np.ones(num_oscillators)
        self.frequency = np.ones(num_oscillators)

        # Adaptive parameters
        self.base_frequency = 1.0
        self.target_amplitude = 1.0
        self.adaptation_rate = 0.1

        # Coupling parameters
        self.coupling_weights = np.array([[0, 0.5],
                                          [0.5, 0]])
        self.phase_bias = np.array([[0, np.pi],
                                    [-np.pi, 0]])

        # Sensory feedback parameters
        self.feedback_gain = 0.2
        self.phase_reset_threshold = 0.5

        self.timestep = 0.01

    def adapt_parameters(self, robot_state: RobotState):
        """Adapt CPG parameters based on robot state"""
        # Adapt frequency based on forward velocity
        desired_velocity = 0.5  # target walking speed
        velocity_error = desired_velocity - np.linalg.norm(robot_state.com_velocity[:2])
        self.base_frequency += self.adaptation_rate * velocity_error
        self.base_frequency = np.clip(self.base_frequency, 0.5, 2.0)  # limit frequency range

        # Adapt amplitude based on stability
        stability_measure = np.abs(robot_state.orientation[1])  # pitch angle
        if stability_measure > 0.1:  # if robot is tilting
            self.target_amplitude *= 0.9  # reduce step size
        else:
            self.target_amplitude = min(1.0, self.target_amplitude * 1.1)  # gradually increase

        # Update oscillator frequencies
        self.frequency = self.base_frequency * np.ones(self.num_oscillators)

    def phase_reset(self, foot_forces: np.ndarray):
        """Reset phase based on foot contact (only vertical force fz)."""
        for i in range(self.num_oscillators):
            fz = foot_forces[i, 2]  # Extract normal force (fz)

            if fz > self.phase_reset_threshold:  # Use scalar comparison
                target_phase = 0 if i == 0 else np.pi
                phase_error = target_phase - self.phase[i]
                self.phase[i] += 0.5 * phase_error  # Partial phase reset

    def update_coupling(self, robot_state: RobotState):
        """Update coupling weights based on robot state"""
        stability = np.exp(-np.sum(np.abs(robot_state.orientation)))
        self.coupling_weights *= stability  # reduce coupling when unstable

    def step(self, robot_state: RobotState) -> np.ndarray:
        """Update oscillator states with feedback"""
        # Adapt parameters based on robot state
        self.adapt_parameters(robot_state)

        # Phase resetting based on foot contact
        self.phase_reset(robot_state.foot_forces)

        # Update coupling based on stability
        self.update_coupling(robot_state)

        # Update phase dynamics
        for i in range(self.num_oscillators):
            # Natural frequency evolution
            self.phase[i] += 2 * np.pi * self.frequency[i] * self.timestep

            # Coupling effects
            for j in range(self.num_oscillators):
                coupling = self.coupling_weights[i, j] * \
                           np.sin(self.phase[j] - self.phase[i] - self.phase_bias[i, j])
                self.phase[i] += coupling * self.timestep

            # Sensory feedback influence
            feedback = self.compute_feedback(robot_state, i)
            self.phase[i] += self.feedback_gain * feedback * self.timestep

        # Keep phase in [0, 2π]
        self.phase = np.mod(self.phase, 2 * np.pi)

        # Update amplitude with smoothing
        self.amplitude += (self.target_amplitude - self.amplitude) * 0.1

        return self.get_output()

    def compute_feedback(self, robot_state: RobotState, oscillator_idx: int) -> float:
        """Compute sensory feedback for phase modification."""

        # Extract vertical force (fz) from foot forces
        fz = robot_state.foot_forces[oscillator_idx, 2]

        # Stability feedback from orientation
        pitch = robot_state.orientation[1]  # Pitch angle
        roll = robot_state.orientation[0]  # Roll angle

        # Compute feedback as a single scalar value
        feedback = (
                -0.1 * float(fz) * np.sin(float(self.phase[oscillator_idx]))  # Ensure fz is a scalar
                - 0.2 * float(pitch) * np.cos(float(self.phase[oscillator_idx]))  # Ensure pitch is a scalar
                - 0.1 * float(roll)  # Ensure roll is a scalar
        )

        return feedback  # This is now a single float, not an array

    def get_output(self) -> np.ndarray:
        """Get oscillator outputs"""
        return self.amplitude * np.sin(self.phase)


class GaitGenerator:
    """Generate walking gait patterns"""

    def __init__(self, robot_params):
        self.params = robot_params
        self.cpg = AdaptiveCPG(num_oscillators=2)

        # Gait parameters
        self.stride_length = 0.3
        self.step_height = 0.1
        self.stance_duration = 0.6  # percentage of cycle
        self.lateral_width = 0.1

        # Phase states
        self.is_left_stance = True
        self.gait_phase = 0.0

    def compute_foot_trajectories(self, robot_state: RobotState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute desired foot positions based on gait phase
        Returns: (left_foot_pos, right_foot_pos)
        """
        cpg_output = self.cpg.step(robot_state)

        # Use CPG outputs to modulate stride and step height
        left_cpg = cpg_output[0]
        right_cpg = cpg_output[1]

        # Forward motion based on CPG phase
        left_x = self.stride_length * left_cpg
        right_x = self.stride_length * right_cpg

        # Vertical trajectory modulated by CPG amplitude
        left_z = self.step_height * np.maximum(0, left_cpg)
        right_z = self.step_height * np.maximum(0, right_cpg)

        # Lateral position with CPG influence for balance
        left_y = self.lateral_width + 0.02 * left_cpg
        right_y = -self.lateral_width + 0.02 * right_cpg

        # Create foot position vectors
        left_pos = np.array([left_x, left_y, left_z])
        right_pos = np.array([right_x, right_y, right_z])

        # Add stability adjustments based on robot state
        if np.abs(robot_state.orientation[0]) > 0.1:  # Roll adjustment
            roll_compensation = 0.05 * np.sign(robot_state.orientation[0])
            left_pos[1] += roll_compensation
            right_pos[1] -= roll_compensation

        if np.abs(robot_state.orientation[1]) > 0.1:  # Pitch adjustment
            pitch_compensation = 0.05 * np.sign(robot_state.orientation[1])
            left_pos[0] += pitch_compensation
            right_pos[0] += pitch_compensation

        return left_pos, right_pos


class UVCBipedRobot:
    def __init__(self):
        self.verbose = True

        # Physics setup
        self.robot_id = None
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        # Simulation parameters
        self.simulation_timestep = 1.0/240.0  # 240 Hz simulation

        # Robot parameters
        self.leg_length = 0.5
        self.foot_width = 0.1
        self.foot_length = 0.2
        self.hip_width = 0.2
        self.mass = 50.0  # kg

        self.knee_joints = []
        self.hip_joints = []
        self.ankle_joints = []

        # Control parameters
        self.Kp = np.array([100.0, 100.0, 100.0])  # For roll, pitch, yaw
        self.Ki = np.array([1.0, 1.0, 1.0])
        self.Kd = np.array([10.0, 10.0, 10.0])

        # State variables
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)
        self.phase = 0  # 0: double support, 1: left support, 2: right support

        # Walking state variables
        self.walking_enabled = False
        self.gait_phase = 0.0
        self.gait_frequency = 1.0  # Hz

        # Load ground plane and create robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.create_robot()

        # Add gait control components
        self.gait_generator = GaitGenerator(robot_params={
            'leg_length': self.leg_length,
            'hip_width': self.hip_width
        })

        # Add new attributes needed for transitions
        self.in_transition = False
        self.transition_start = 0.0
        self.transition_phase = 0.0

        # Add control gains for standing and walking
        self.Kp_stand = 100.0
        self.Kd_stand = 10.0
        self.Kp_walk = 80.0
        self.Kd_walk = 8.0
        self.Kp_pos = 0
        self.Kd_pos = 0

    def handle_walking_transition(self):
        """Handle transitions between standing and walking"""
        TRANS_TIME = 1.0  # Transition time in seconds

        if self.walking_enabled and not self.in_transition:
            # Start transition to walking
            self.transition_start = time.time()
            self.in_transition = True
            self.transition_phase = 0.0

        if self.in_transition:
            # Update transition phase
            current_time = time.time()
            self.transition_phase = min(1.0,
                                        (current_time - self.transition_start) / TRANS_TIME)

            # Modify control gains during transition
            self.Kp_pos = self.lerp(self.Kp_stand, self.Kp_walk, self.transition_phase)
            self.Kd_pos = self.lerp(self.Kd_stand, self.Kd_walk, self.transition_phase)

            if self.transition_phase >= 1.0:
                self.in_transition = False

    def lerp(self, start: float, end: float, t: float) -> float:
        """Linear interpolation between start and end values"""
        return start + t * (end - start)

    def compute_walking_pose(self, state: RobotState) -> Tuple[np.ndarray, np.ndarray]:
        """Compute desired leg poses for walking"""
        left_foot, right_foot = self.gait_generator.compute_foot_trajectories(state)

        # Compute inverse kinematics for both legs
        left_joints = self.inverse_kinematics(left_foot, is_left=True)
        right_joints = self.inverse_kinematics(right_foot, is_left=False)

        return left_joints, right_joints

    def inverse_kinematics(self, target_pos: np.ndarray, is_left: bool) -> np.ndarray:
        """
        Analytical inverse kinematics for left or right leg.
        Returns joint angles for hip and knee.

        Args:
            target_pos (np.ndarray): The desired foot position in the world frame [x, y, z].
            is_left (bool): Whether computing for the left leg (True) or right leg (False).

        Returns:
            np.ndarray: Joint angles [hip_roll, hip_pitch, hip_yaw, knee_angle]
        """

        x, y, z = target_pos
        leg_length = self.leg_length

        # Apply lateral flipping for left/right leg
        if not is_left:
            y = -y  # Flip y-direction for right leg

        # Compute distance from hip to target foot position
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # Ensure foot does not extend beyond physical limits
        max_reach = 2 * leg_length - 0.01  # Slight tolerance
        if d > max_reach:
            scaling_factor = max_reach / d
            x *= scaling_factor
            y *= scaling_factor
            z *= scaling_factor
            d = max_reach  # Set to max allowable reach

        # Compute knee angle using the law of cosines
        knee_angle = np.arccos((d ** 2 - 2 * leg_length ** 2) / (-2 * leg_length ** 2))

        # Flip knee bending direction
        if is_left:
            knee_angle = -knee_angle  # Ensure both knees bend in the same direction

        # Compute hip pitch
        hip_pitch = np.arctan2(x, z) - np.arccos(d / (2 * leg_length))

        # Compute hip roll
        hip_roll = np.arctan2(y, z)

        # Compute hip yaw (yaw is typically minimal but can be adjusted)
        hip_yaw = np.arctan2(y, x) * (1 if is_left else -1)  # Flip for right leg

        return np.array([hip_roll, hip_pitch, hip_yaw, knee_angle])

    def get_slip_velocity(self, state: RobotState) -> np.ndarray:
        """Calculate slip velocity for each foot"""
        slip_velocities = np.zeros((2, 2))  # [left,right] x [x,y]

        for i, foot in enumerate(self.ankle_joints):
            # Get foot velocity in world frame
            link_state = p.getLinkState(self.robot_id, foot, computeLinkVelocity=1)
            world_vel = np.array(link_state[6])  # Linear velocity

            # Only consider x-y components
            slip_velocities[i] = world_vel[:2]

        return slip_velocities

    def get_slip_direction(self, state: RobotState) -> List[str]:
        """Determine direction of slip for each foot"""
        directions = []
        slip_velocities = self.get_slip_velocity(state)

        for foot_vel in slip_velocities:
            magnitude = np.linalg.norm(foot_vel)
            if magnitude > 0.01:  # Threshold for significant motion
                angle = np.arctan2(foot_vel[1], foot_vel[0])
                # Convert angle to cardinal direction
                direction = self.angle_to_direction(angle)
            else:
                direction = "none"
            directions.append(direction)

        return directions

    def angle_to_direction(self, angle: float) -> str:
        """Convert angle to cardinal direction"""
        # Convert angle to degrees and shift to 0-360 range
        degrees = np.degrees(angle) % 360

        if 45 <= degrees < 135:
            return "forward"
        elif 135 <= degrees < 225:
            return "left"
        elif 225 <= degrees < 315:
            return "backward"
        else:
            return "right"

    def prevent_slip(self, state: RobotState):
        """Implement slip prevention strategy"""
        left_slip, right_slip = self.detect_slip(state)
        slip_directions = self.get_slip_direction(state)

        for i, (is_slipping, direction) in enumerate(zip([left_slip, right_slip], slip_directions)):
            if is_slipping:
                foot = "left" if i == 0 else "right"

                # Adjust foot force based on slip direction
                if direction == "forward":
                    self.reduce_forward_force(foot)
                elif direction == "backward":
                    self.reduce_backward_force(foot)

                # Increase normal force to increase friction
                self.increase_normal_force(foot)

                # Adjust foot orientation to improve grip
                self.adjust_foot_orientation(foot)

    def handle_slip_event(self, foot: str):
        """
        Handle detected slip event by adjusting forces and posture
        Args:
            foot: "left" or "right" to indicate which foot is slipping
        """
        # Get current state
        state = self.get_state()

        # Determine which joint indices to use - fixed version
        if foot == "left":
            hip_joint = self.hip_joints[0]
            knee_joint = self.knee_joints[0]
            ankle_joint = self.ankle_joints[0]
        else:
            hip_joint = self.hip_joints[1]
            knee_joint = self.knee_joints[1]
            ankle_joint = self.ankle_joints[1]

        # 1. Reduce forward/backward forces by reducing hip torque
        current_hip_torque = p.getJointState(self.robot_id, hip_joint)[3]  # Get applied torque
        reduced_torque = current_hip_torque * 0.5  # Reduce by 50%
        p.setJointMotorControl2(
            self.robot_id,
            hip_joint,
            p.TORQUE_CONTROL,
            force=reduced_torque
        )

        # 2. Increase normal force by adjusting knee angle
        current_knee_angle = p.getJointState(self.robot_id, knee_joint)[0]
        new_knee_angle = current_knee_angle + 0.1  # Increase bend by 0.1 radians
        p.setJointMotorControl2(
            self.robot_id,
            knee_joint,
            p.POSITION_CONTROL,
            targetPosition=new_knee_angle,
            maxVelocity=5.0
        )

        # 3. Adjust foot orientation by modifying ankle angle
        current_ankle_angle = p.getJointState(self.robot_id, ankle_joint)[0]
        new_ankle_angle = current_ankle_angle - 0.05  # Adjust by 0.05 radians
        p.setJointMotorControl2(
            self.robot_id,
            ankle_joint,
            p.POSITION_CONTROL,
            targetPosition=new_ankle_angle,
            maxVelocity=5.0
        )

        # 4. Shift body weight to stable foot
        com_pos = state.com_position
        if foot == "left":
            target_shift = np.array([0, -0.02, 0])  # Shift right
        else:
            target_shift = np.array([0, 0.02, 0])  # Shift left

        # 4. Shift body weight to stable foot
        other_hip = self.hip_joints[1] if foot == "left" else self.hip_joints[0]
        p.setJointMotorControl2(
            self.robot_id,
            other_hip,
            p.POSITION_CONTROL,
            targetPosition=0.1 if foot == "left" else -0.1,  # Lean toward stable foot
            maxVelocity=2.0
        )

        # Helper methods that were previously undefined
        self.reduce_forward_force(foot)
        self.reduce_backward_force(foot)
        self.increase_normal_force(foot)
        self.adjust_foot_orientation(foot)

    def reduce_forward_force(self, foot: str):
        """Reduce forward force on specified foot"""
        joint_index = self.hip_joints[0] if foot == "left" else self.hip_joints[1]
        current_velocity = p.getJointState(self.robot_id, joint_index)[1]

        if current_velocity > 0:  # If moving forward
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.VELOCITY_CONTROL,
                targetVelocity=current_velocity * 0.5,  # Reduce velocity by half
                force=10.0
            )

    def reduce_backward_force(self, foot: str):
        """Reduce backward force on specified foot"""
        joint_index = self.hip_joints[0] if foot == "left" else self.hip_joints[1]
        current_velocity = p.getJointState(self.robot_id, joint_index)[1]

        if current_velocity < 0:  # If moving backward
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.VELOCITY_CONTROL,
                targetVelocity=current_velocity * 0.5,  # Reduce velocity by half
                force=10.0
            )

    def increase_normal_force(self, foot: str):
        """Increase normal force on specified foot"""
        knee_index = self.knee_joints[0] if foot == "left" else self.knee_joints[1]
        current_angle = p.getJointState(self.robot_id, knee_index)[0]

        # Slightly bend knee to increase ground pressure
        target_angle = current_angle + 0.1
        p.setJointMotorControl2(
            self.robot_id,
            knee_index,
            p.POSITION_CONTROL,
            targetPosition=target_angle,
            maxVelocity=2.0,
            force=50.0
        )

    def adjust_foot_orientation(self, foot: str):
        """Adjust foot orientation to improve ground contact"""
        ankle_index = self.ankle_joints[0] if foot == "left" else self.ankle_joints[1]
        current_angle = p.getJointState(self.robot_id, ankle_index)[0]

        # Adjust ankle angle to improve ground contact
        p.setJointMotorControl2(
            self.robot_id,
            ankle_index,
            p.POSITION_CONTROL,
            targetPosition=0.0,  # Try to keep foot parallel to ground
            maxVelocity=5.0,
            force=20.0
        )

    def step_simulation(self):
        """Perform one step of simulation"""
        state = self.get_state()

        # Detect slip
        left_slip, right_slip = self.detect_slip(state)

        if left_slip or right_slip:
            # Could modify control strategy here when slip is detected
            # For example:
            # - Reduce walking speed
            # - Adjust foot placement
            # - Increase friction coefficient in simulation
            # - Modify stepping pattern
            if left_slip:
                self.handle_slip_event("left")
            if right_slip:
                self.handle_slip_event("right")

        # Handle walking transitions
        self.handle_walking_transition()

        if self.walking_enabled:
            # Update gait phase
            self.gait_phase += 2 * np.pi * self.gait_frequency * self.simulation_timestep
            self.gait_phase = np.mod(self.gait_phase, 2 * np.pi)

            # Get desired leg poses using state feedback
            left_joints, right_joints = self.compute_walking_pose(state)

            # Combine into single gait command array
            gait_command = np.concatenate([left_joints, right_joints])

            # Get UVC stabilizing torques
            uvc_command = self.uvc_control(state)

            # Calculate support phase based on gait phase (0 to 1)
            support_phase = (np.cos(self.gait_phase) + 1) / 2

            try:
                # Blend walking and stabilizing controls
                blended = self.blend_controls(gait_command, uvc_command, support_phase)

                # Apply blended controls
                for i, joint in enumerate(self.hip_joints + self.knee_joints + self.ankle_joints):
                    p.setJointMotorControl2(
                        bodyUniqueId=self.robot_id,
                        jointIndex=joint,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=blended['position'][i],
                        force=blended['torque'][i],
                        positionGain=self.Kp_pos,
                        velocityGain=self.Kd_pos
                    )
            except ValueError as e:
                print(f"Error in control blending: {e}")
                # Fall back to basic UVC control
                self.apply_joint_torques(uvc_command)

        else:
            # Just apply UVC when standing
            hip_torques = self.uvc_control(state)
            self.apply_joint_torques(hip_torques)

        p.stepSimulation()

    def apply_hip_torques(self, hip_torques: np.ndarray):
        """Apply torques to hip joints only"""
        for i, joint in enumerate(self.hip_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint,
                controlMode=p.TORQUE_CONTROL,
                force=hip_torques[i],
                positionGain=0.0,
                velocityGain=0.0
            )

    def apply_joint_torques(self, hip_torques: np.ndarray):
        """
        Apply computed torques to hip joints

        Args:
            hip_torques: numpy array of shape (6,) containing torques for all hip DOFs
                        [left_roll, left_pitch, left_yaw, right_roll, right_pitch, right_yaw]
        """
        # Apply torques to each hip joint
        for i, joint in enumerate(self.hip_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint,
                controlMode=p.TORQUE_CONTROL,
                force=hip_torques[i],
                # Disable the built-in position constraint motor
                positionGain=0.0,
                velocityGain=0.0
            )

            # Add some damping to prevent unwanted oscillations
            current_vel = p.getJointState(self.robot_id, joint)[1]
            damping_torque = -0.1 * current_vel  # damping coefficient = 0.1

            p.applyExternalTorque(
                objectUniqueId=self.robot_id,
                linkIndex=joint,
                torqueObj=[0, 0, damping_torque],
                flags=p.LINK_FRAME
            )

    def apply_joint_control(self, left_joints: np.ndarray, right_joints: np.ndarray,
                            stabilizing_torques: np.ndarray):
        """Apply combined position and torque control to joints"""
        # Position control gains
        Kp_pos = 100.0
        Kd_pos = 10.0

        # Apply position control with feed-forward torque from UVC
        for i, joint in enumerate(self.hip_joints[:3]):  # Left hip
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=left_joints[i],
                positionGain=Kp_pos,
                velocityGain=Kd_pos,
                force=stabilizing_torques[i]
            )

        for i, joint in enumerate(self.hip_joints[3:]):  # Right hip
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=right_joints[i],
                positionGain=Kp_pos,
                velocityGain=Kd_pos,
                force=stabilizing_torques[i + 3]
            )

    def create_robot(self):
        """Create robot URDF on the fly"""
        # Link masses
        torso_mass = 5
        upper_leg_mass = 5.0
        lower_leg_mass = 5.0
        foot_mass = 1

        # Link Extents
        torso_extents = [0.1, 0.15, 0.2]
        upper_leg_extents = [0.05, 0.05, self.leg_length / 4]
        lower_leg_extents = [0.03, 0.03, self.leg_length / 4]
        foot_extents = [0.1, 0.05, 0.02]

        # Create collision shapes - origin at TOP of each shape
        torso_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=torso_extents,
            collisionFramePosition=[0, 0, -torso_extents[2]]  # Shift down from origin
        )
        upper_leg_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=upper_leg_extents,
            collisionFramePosition=[0, 0, -upper_leg_extents[2]]  # Shift down from origin
        )
        lower_leg_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=lower_leg_extents,
            collisionFramePosition=[0, 0, -lower_leg_extents[2]]  # Shift down from origin
        )
        foot_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=foot_extents,
            collisionFramePosition=[0, 0, -foot_extents[2]]  # Shift down from origin
        )

        # Create matching visual shapes
        torso_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=torso_extents,
            rgbaColor=[0.8, 0.2, 0.2, 1],
            visualFramePosition=[0, 0, -torso_extents[2]]
        )
        upper_leg_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=upper_leg_extents,
            rgbaColor=[0.2, 0.2, 0.8, 1],
            visualFramePosition=[0, 0, -upper_leg_extents[2]]
        )
        lower_leg_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=lower_leg_extents,
            rgbaColor=[0.2, 0.8, 0.2, 1],
            visualFramePosition=[0, 0, -lower_leg_extents[2]]
        )
        foot_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=foot_extents,
            rgbaColor=[0.8, 0.8, 0.2, 1],
            visualFramePosition=[0, 0, -foot_extents[2]]
        )

        # Joint positions - each at the TOP of the child link
        left_hip_pos = [0, self.hip_width / 2, -(torso_extents[2]*2)]
        right_hip_pos = [0, -self.hip_width / 2, -(torso_extents[2]*2)]

        basePosition = [0, 0, 1.25]
        baseOrientation = [0, 0, 0, 1]

        # Position each joint at the end of the previous link
        knee_pos = [0, 0, -2 * upper_leg_extents[2]]  # Full length of upper leg down from hip
        ankle_pos = [0, 0, -2 * lower_leg_extents[2]]  # Full length of lower leg down from knee

        # Joint axes
        hip_pitch_axis = [0, 1, 0]  # Around Y - forward-back
        knee_axis = [0, 1, 0]  # Around Y - knee bend
        ankle_axis = [0, 1, 0]  # Around Y - foot tilt

        linkJointAxis = [
            hip_pitch_axis,  # Hip joints rotate around Y axis (pitch)
            knee_axis,  # Knee joints rotate around Y axis
            ankle_axis,  # Ankle joints rotate around Y axis
            hip_pitch_axis,  # Repeated for right leg
            knee_axis,
            ankle_axis
        ]

        if self.verbose:
            print("Debugging link array lengths:")
            print("Masses:",
                  len([torso_mass, upper_leg_mass, lower_leg_mass, foot_mass, upper_leg_mass, lower_leg_mass, foot_mass]))
            print("Collision Shapes:",
                  len([torso_shape, upper_leg_shape, lower_leg_shape, foot_shape, upper_leg_shape, lower_leg_shape,
                       foot_shape]))
            print("Visual Shapes:",
                  len([torso_visual, upper_leg_visual, lower_leg_visual, foot_visual, upper_leg_visual, lower_leg_visual,
                       foot_visual]))
            print("Link Positions:", len([left_hip_pos, knee_pos, ankle_pos, right_hip_pos, knee_pos, ankle_pos]))
            print("Parent Indices:", len([0, 1, 2, 0, 4, 5]))
            print("Joint Types:", len([p.JOINT_REVOLUTE] * 6))
            print("Joint Axes:", len([hip_pitch_axis, knee_axis, ankle_axis, hip_pitch_axis, knee_axis, ankle_axis]))

            # Create the base (torso)
            self.robot_id = p.createMultiBody(
                baseMass=self.mass / 2,  # Half mass to torso
                baseCollisionShapeIndex=torso_shape,
                baseVisualShapeIndex=torso_visual,
                basePosition=basePosition,
                baseOrientation=baseOrientation,
                linkMasses=[
                    self.mass / 10, self.mass / 10, self.mass / 10,  # Left leg
                    self.mass / 10, self.mass / 10, self.mass / 10  # Right leg
                ],
                linkCollisionShapeIndices=[
                    upper_leg_shape, lower_leg_shape, foot_shape,  # Left leg
                    upper_leg_shape, lower_leg_shape, foot_shape  # Right leg
                ],
                linkVisualShapeIndices=[
                    upper_leg_visual, lower_leg_visual, foot_visual,  # Left leg
                    upper_leg_visual, lower_leg_visual, foot_visual  # Right leg
                ],
                linkPositions=[
                    left_hip_pos, knee_pos, ankle_pos,  # Left leg
                    right_hip_pos, knee_pos, ankle_pos  # Right leg
                ],
                linkOrientations=[[0, 0, 0, 1]] * 6,
                linkInertialFramePositions=[[0, 0, 0]] * 6,
                linkInertialFrameOrientations=[[0, 0, 0, 1]] * 6,
                linkParentIndices=[0, 1, 2, 0, 4, 5],
                linkJointTypes=[p.JOINT_REVOLUTE] * 6,
                linkJointAxis=linkJointAxis
            )

        # Store joint indices
        self.hip_joints = [0, 3]  # Left and right hip joints
        self.knee_joints = [1, 4]  # Left and right knee joints
        self.ankle_joints = [2, 5]  # Left and right ankle joints

        for joint in range(6):
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=50
            )

            # Apply damping to all joints
            p.changeDynamics(self.robot_id, joint, linearDamping=0.04, angularDamping=0.04)

            # Set joint limits
            if joint in self.hip_joints:
                # joint_limit = math.pi / 2  # 90 degrees
                joint_limit = math.pi / 4  # 45 degrees
            elif joint in self.knee_joints:
                # joint_limit = math.pi / 3  # 60 degrees
                joint_limit = math.pi / 4  # 45 degrees
            else:  # ankle joints
                joint_limit = math.pi / 4  # 45 degrees


            p.changeDynamics(
                self.robot_id,
                joint,
                jointLowerLimit=-joint_limit,
                jointUpperLimit=joint_limit
            )

            # Initial positions
            target_pos = 0.2 if joint in self.knee_joints else 0
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                positionGain=0.1,
                velocityGain=0.1,
                maxVelocity=5,
                force=50
            )

    def get_state(self) -> RobotState:
        """Get current robot state"""
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)

        # Number of joints per leg
        joints_per_leg = 3  # hip, knee, ankle
        num_legs = 2  # left and right
        num_joints = joints_per_leg * num_legs

        joint_positions = np.zeros(num_joints)
        joint_velocities = np.zeros(num_joints)

        # Get states in left leg -> right leg order
        joint_indices = [
            (self.hip_joints[0], 0),  # Left hip
            (self.knee_joints[0], 1),  # Left knee
            (self.ankle_joints[0], 2),  # Left ankle
            (self.hip_joints[1], 3),  # Right hip
            (self.knee_joints[1], 4),  # Right knee
            (self.ankle_joints[1], 5),  # Right ankle
        ]

        # Fill arrays maintaining leg-wise ordering
        for joint_idx, array_idx in joint_indices:
            state = p.getJointState(self.robot_id, joint_idx)
            joint_positions[array_idx] = state[0]
            joint_velocities[array_idx] = state[1]

        # Initialize foot_forces array with correct dimensions
        foot_forces = np.zeros((2, 6))  # [left, right] x [fx, fy, fz, mx, my, mz]

        for i, foot in enumerate(self.ankle_joints):
            try:
                contact_points = p.getContactPoints(self.robot_id, self.plane_id, foot)
                if contact_points:
                    contact = contact_points[0]  # Take first contact point

                    # Extract forces - contact normal force is a scalar
                    normal_force = contact[9]

                    # Get applied force in world coordinates
                    force_applied = contact[13]  # This is a 3-component vector

                    # Store forces
                    foot_forces[i][0] = force_applied[0]  # x component
                    foot_forces[i][1] = force_applied[1]  # y component
                    foot_forces[i][2] = normal_force  # z component (normal force)

                    # Get positions for moment calculation
                    pos_on_a = contact[5]  # Position on first body (robot)
                    pos_on_b = contact[6]  # Position on second body (ground)

                    # Calculate moment arm
                    r = np.array(pos_on_a) - np.array(pos_on_b)
                    force = np.array([force_applied[0], force_applied[1], normal_force])
                    moment = np.cross(r, force)
                    foot_forces[i][3:] = moment
            except Exception as e:
                print(f"Error getting contact points for foot {i}: {e}")
                foot_forces[i] = np.zeros(6)

        return RobotState(
            orientation=p.getEulerFromQuaternion(ori),
            angular_velocity=ang_vel,
            com_position=pos,
            com_velocity=lin_vel,
            foot_forces=foot_forces,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )

    def detect_slip(self, state: RobotState, friction_coefficient: float = 0.6) -> Tuple[bool, bool]:
        """
        Detect if either foot is slipping using Coulomb friction model

        Args:
            state: Current robot state
            friction_coefficient: Static friction coefficient between feet and ground (typically 0.6-0.8)

        Returns:
            Tuple of (left_foot_slip, right_foot_slip) booleans
        """
        left_slip = False
        right_slip = False

        for i in range(2):
            # Extract forces for this foot
            normal_force = abs(state.foot_forces[i][2])  # Vertical force component
            lateral_forces = state.foot_forces[i][:2]  # X,Y force components

            # Calculate magnitude of lateral force
            lateral_force_magnitude = np.linalg.norm(lateral_forces)

            # Check for slip using Coulomb friction
            if normal_force > 0.1:  # Only check if foot has significant contact
                # If lateral force exceeds friction force, slip is occurring
                if lateral_force_magnitude > friction_coefficient * normal_force:
                    if i == 0:
                        left_slip = True
                    else:
                        right_slip = True

                    if self.verbose:
                        print(f"Slip detected on {'left' if i == 0 else 'right'} foot!")
                        print(f"Normal force: {normal_force:.2f} N")
                        print(f"Lateral force: {lateral_force_magnitude:.2f} N")
                        print(f"Max static friction: {friction_coefficient * normal_force:.2f} N")

        return left_slip, right_slip

    def uvc_control_full(self, state: RobotState) -> np.ndarray:
        """
        Implement UVC control law using the Jacobian for torque conversion
        """
        # Calculate orientation error
        desired_orientation = np.zeros(3)  # We want the torso vertical
        error = desired_orientation - state.orientation

        # Update integral and derivative terms
        self.integral_error += error * self.simulation_timestep
        derivative_error = (error - self.last_error) / self.simulation_timestep
        self.last_error = error

        # Compute desired torso angular acceleration
        torso_acc = (self.Kp * error +
                     self.Ki * self.integral_error +
                     self.Kd * derivative_error)

        # Get Jacobian
        J = self.compute_jacobian(state)

        # Compute required joint torques using the Jacobian transpose
        # τ = J^T * I * α
        # where I is the inertia tensor and α is angular acceleration
        I = np.eye(3) * self.mass  # Simplified inertia tensor
        all_torques = np.dot(J.T, np.dot(I, torso_acc))

        # Split torques for different joint groups
        hip_torques = all_torques[:6]
        knee_torques = all_torques[6:8]
        ankle_torques = all_torques[8:]

        # Apply torque limits
        max_torque = 100.0  # Nm
        hip_torques = np.clip(hip_torques, -max_torque, max_torque)

        # Apply appropriate torque limits for each joint type
        hip_torques = np.clip(hip_torques, -max_torque, max_torque)  # Higher limits for hips
        knee_torques = np.clip(knee_torques, -max_torque/2, max_torque/2)  # Lower limits for knees
        ankle_torques = np.clip(ankle_torques, -max_torque/3, max_torque/3)  # Even lower for ankles

        return np.concatenate([hip_torques, knee_torques, ankle_torques])

    def uvc_control(self, state: RobotState) -> np.ndarray:
        """
        Implement UVC control law using the Jacobian for torque conversion
        """
        # Calculate orientation error
        desired_orientation = np.zeros(3)  # We want the torso vertical
        error = desired_orientation - state.orientation

        # Update integral and derivative terms
        self.integral_error += error * self.simulation_timestep
        derivative_error = (error - self.last_error) / self.simulation_timestep
        self.last_error = error

        # Compute desired torso angular acceleration
        torso_acc = (self.Kp * error +
                     self.Ki * self.integral_error +
                     self.Kd * derivative_error)

        # Get Jacobian
        J = self.compute_jacobian(state)

        # Compute required joint torques using the Jacobian transpose
        I = np.eye(3) * self.mass  # Simplified inertia tensor
        all_torques = np.dot(J.T, np.dot(I, torso_acc))

        # Initialize torques array with correct dimensions (8 total joints)
        full_torques = np.zeros(8)  # 4 joints per leg

        # Apply hip torques to the first joint of each leg
        full_torques[0] = all_torques[0]  # Left hip
        full_torques[4] = all_torques[3]  # Right hip

        # Add smaller stabilizing torques for other joints
        knee_torque = 0.5  # Small constant torque for stability
        ankle_torque = 0.3

        # Add knee and ankle torques
        full_torques[1] = knee_torque  # Left knee
        full_torques[2] = ankle_torque  # Left ankle
        full_torques[5] = knee_torque  # Right knee
        full_torques[6] = ankle_torque  # Right ankle

        # Apply torque limits
        max_torque = 100.0  # Nm
        full_torques = np.clip(full_torques, -max_torque, max_torque)

        return full_torques

    def compute_hip_torques(self, tau: np.ndarray, state: RobotState) -> np.ndarray:
        """Convert desired torso torques to hip joint torques"""
        J = self.compute_jacobian(state)
        hip_torques = np.dot(J.T, tau)
        return hip_torques

    @staticmethod
    def blend_controls(gait_command: np.ndarray, uvc_command: np.ndarray,
                       support_phase: float) -> BlendedControls:
        """
        Blend gait generator commands with UVC stabilizing commands

        Args:
            gait_command: Joint positions from gait generator (n_joints,)
            uvc_command: Stabilizing torques from UVC (n_joints,)
            support_phase: 0-1 indicating progression through support phase

        Returns:
            Dictionary containing:
                'position': Array of target joint positions
                'torque': Array of scaled stabilizing torques

        Raises:
            ValueError: If input dimensions don't match or support_phase is invalid
        """
        # Validate inputs
        if gait_command is None or uvc_command is None:
            raise ValueError("Commands cannot be None")

        if len(gait_command.shape) != 1 or len(uvc_command.shape) != 1:
            raise ValueError("Commands must be 1D arrays")

        if gait_command.shape != uvc_command.shape:
            raise ValueError(f"Command dimensions must match: got {gait_command.shape} and {uvc_command.shape}")

        if not 0 <= support_phase <= 1:
            raise ValueError(f"Support phase must be between 0 and 1, got {support_phase}")

        n_joints = len(gait_command)
        n_joints_per_leg = n_joints // 2

        # Weight factors for blending
        stance_weight = np.exp(-4 * support_phase)  # Higher weight during early stance
        swing_weight = 1 - stance_weight

        # Create scaling array with proper dimensions
        uvc_scaling = np.ones_like(uvc_command)

        # Scale each leg's joints
        uvc_scaling[:n_joints_per_leg] *= stance_weight  # Left leg
        uvc_scaling[n_joints_per_leg:] *= swing_weight  # Right leg

        try:
            blended = {
                'position': gait_command.copy(),  # Prevent modifying original
                'torque': uvc_command * uvc_scaling
            }
            return blended
        except Exception as e:
            raise ValueError(f"Error blending controls: {str(e)}")

    def compute_jacobian_hip(self, state: RobotState) -> np.ndarray:
        """
        Compute the Jacobian matrix relating hip joint velocities to torso angular velocities
        Only considers pitch DOF for each hip joint
        """
        # Initialize Jacobian matrix (3x2 for single-DOF hip joints)
        J = np.zeros((3, 2))  # Changed from (3,6) to (3,2)

        # Get hip angles - only pitch angles
        hip_angles = state.joint_positions[:2]  # Only take first 2 angles for hip pitch

        # For each hip (left and right)
        for side in range(2):
            hip_pos = np.array([0, self.hip_width / 2 if side == 0 else -self.hip_width / 2, 0])

            # Only compute pitch contribution
            pitch_angle = hip_angles[side]

            # Pitch axis is around Y
            pitch_axis = np.array([0, 1, 0])

            # Set Jacobian column for this hip's pitch DOF
            J[:, side] = pitch_axis * np.linalg.norm(hip_pos)

        return J

    def compute_jacobian(self, state: RobotState) -> np.ndarray:
        """
        Compute the Jacobian matrix relating hip joint velocities to torso angular velocities
        """
        # Initialize Jacobian matrix (3x6 for hip joints only)
        J = np.zeros((3, 6))

        # Get hip angles - ensure we only get hip joint angles
        hip_angles = state.joint_positions[:6]  # Only take first 6 angles for hips

        # Validate we have the correct number of angles
        if len(hip_angles) < 6:
            if self.verbose:
                print(f"Warning: Not enough hip angles ({len(hip_angles)}), padding with zeros")
            hip_angles = np.pad(hip_angles, (0, 6 - len(hip_angles)))

        # For each hip (left and right)
        for side in range(2):
            hip_pos = np.array([0, self.hip_width / 2 if side == 0 else -self.hip_width / 2, 0])
            offset = side * 3  # Each hip has 3 DOF

            try:
                # Get rotation matrices for current joint configuration
                R_x = self.rotation_matrix_x(float(hip_angles[offset]))
                R_y = self.rotation_matrix_y(float(hip_angles[offset + 1]))
                R_z = self.rotation_matrix_z(float(hip_angles[offset + 2]))

                # For X rotation (roll)
                J[0, offset] = 1.0
                J[1, offset] = 0.0
                J[2, offset] = 0.0

                # For Y rotation (pitch)
                pitch_axis = np.dot(R_x, np.array([0, 1, 0]))
                J[0:3, offset + 1] = pitch_axis

                # For Z rotation (yaw)
                yaw_axis = np.dot(R_x @ R_y, np.array([0, 0, 1]))
                J[0:3, offset + 2] = yaw_axis

                # Scale columns based on hip position
                for i in range(3):
                    J[:, offset + i] *= np.linalg.norm(hip_pos)

            except IndexError as e:
                print(f"Error accessing joint angles at offset {offset}: {e}")
                # Fill remaining columns with zeros
                J[:, offset:offset + 3] = 0

        return J

    def rotation_matrix_x(self, angle: float) -> np.ndarray:
        """Create rotation matrix for rotation around x-axis"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])

    def rotation_matrix_y(self, angle: float) -> np.ndarray:
        """Create rotation matrix for rotation around y-axis"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    def rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Create rotation matrix for rotation around z-axis"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])



def main():
    robot = UVCBipedRobot()

    try:
        # Wait for robot to stabilize
        for _ in range(100):
            robot.step_simulation()
            time.sleep(1 / 240)

        # Enable walking
        robot.walking_enabled = True

        # Main simulation loop
        while True:
            robot.step_simulation()
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        p.disconnect()


if __name__ == "__main__":
    main()
