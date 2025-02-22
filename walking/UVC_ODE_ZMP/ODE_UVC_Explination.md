
## **Main Concept**
The walking system integrates **three primary control mechanisms**:
1. **Basic Motion** – Generates stable walking motions without external forces.
2. **Postural Reflex (UVC - Upper Vertical Control)** – Ensures the robot's upper body remains vertical, similar to human postural balance.
3. **Timing Control (CPG - Central Pattern Generator)** – Synchronizes walking rhythm and periodic movements.

---

## **Core Components & File Highlights**
### **1. Walking Control Core**
- **Files:** `uvc_core.cpp`, `uvc_core.h`, `ode_core.cpp`, `ode_core.h`
- **Functionality:**  
  - Handles fundamental **walking behavior** and incorporates **UVC (Upper Vertical Control)**.
  - Key variables like `K0W`, `K1W`, `HW`, `A0W`, and `A1W` manage joint angles for the **hip, knee, and ankle**.
  - Includes **experimental PID-like control** for adjusting stride length to keep the **upper body vertical**.

---

### **2. Bipedal Robot Main Routine**
- **Files:** `uvc_biped.cpp`, `uvc_biped.h`, `ode_biped.cpp`, `ode_biped.h`
- **Functionality:**  
  - Responsible for **creating the 3D robot model** and applying walking control inputs.
  - Uses **ODE (Open Dynamics Engine)** for physics simulation and kinematics.
  - **Experimental Features:**  
    - Walking initiation by detecting body inclination.
    - Walking on slopes and steps (adjusts knee bending if swing leg doesn’t detect ground contact).
    - Applying external forces (e.g., simulating a push).

---

### **3. Universal Vertical Control (UVC) Mechanics**
- **File:** `ODE_Explination.md`
- **Core Principle:**  
  - Simulates **postural reflexes in humans**, keeping the upper body upright **without complex mechanical calculations**.
  - Adjusts **hip joint movement on the grounded side** to compensate for upper body tilt.
  - **UVC vs. ZMP:**  
    - Unlike **Zero Moment Point (ZMP) control**, which involves detailed force calculations, **UVC only uses postural corrections** for simplicity.
    - **No gyroscope-based stabilization** (common in other robots).

---

### **4. Simulation and Execution**
- **Files:** `uvc.c`, `ode_2_uvc.md`
- **Functionality:**  
  - `uvc.c` contains low-level **servo control and communication code** (specific to the KHR robot).
  - `ode_2_uvc.md` provides **setup instructions** for running the **ODE-based simulation**.
  - Includes **keyboard commands** for real-time control:
    - `w` → Start walking
    - `u` → Enable/disable UVC (robot body turns **blue** when active)
    - `r` → Reset
    - `1-5` → Apply different walking experiments (slopes, steps, external force)

---

### **5. Addressing Key Research Questions**
- **Mathematical Model**:  
  - The researcher **hasn’t fully defined a mathematical model** but confirms UVC’s effectiveness through physics simulation.
  - No detailed ZMP calculations; relies on **postural reflex adjustments**.

- **Joint Variables & Walking Parameters**:  
  - `dxi` and `dyi` serve as integral terms, modifying **stride length** based on upper body tilt.
  - `autoH` and `autoHs` define leg height to **prevent singularities** when touching the ground.

- **Experimental Coefficients**:  
  - **Empirically chosen values** (e.g., `1.5 * 193 * sin(lrRad)`) optimize walking stability.

- **Alternation Between Support & Free Leg**:  
  - `jikuasi` (軸足) → Pivot foot (support leg: **0 = right, 1 = left**).
  - Alternates between **supporting** and **swinging** foot for balance.

---

### **6. Summary**
This **bipedal walking control system** leverages **UVC for balance** and **CPG for rhythm**, avoiding complex ZMP calculations. It’s implemented in **ODE for simulation** and supports **real-time keyboard control** for testing. The **code structure is modular**, with clear separation between:
- **Core walking control** (`uvc_core`)
- **ODE physics simulation** (`ode_biped`)
- **UVC mechanics & experiments** (`ODE_Explination.md`)

---

Key Components:

1. UVC (Upper Body Vertical Control):
- The core innovation is UVC - a balance control method that keeps the robot's upper body vertical
- Rather than using complex ZMP (Zero Moment Point) calculations, it uses simple postural reflexes
- When the upper body tilts forward, it spreads the legs to push back
- When leaning back, it narrows the stance and moves legs forward
- Similar compensation occurs for side-to-side tilting

2. Walking Control Integration:
Three main elements are combined:
- Basic walking motion (predefined stable gait pattern)
- Postural reflex (UVC)
- Timing control (CPG - Central Pattern Generator, though simplified in this implementation)

3. Key Files Structure:
- ode_biped.cpp/uvc_biped.cpp: Main ODE simulation routines
- ode_core.cpp/uvc_core.cpp: Walking control implementation
- Associated header files (.h) contain structure definitions

4. Control Parameters:
- fbRad: Forward/backward tilt angle (前後角度)
- lrRad: Left/right tilt angle (左右角度) 
- fbAV: Forward/backward angular velocity (前後角速度)
- lrAV: Left/right angular velocity (左右角速度)
- asiPress_r/l: Right/left foot pressure (足圧力)

5. Robot Structure:
- Modeled with various body parts (頭/head, 胴/torso, 腰/hip joints, etc.)
- Uses hinge joints, slider joints, and fixed joints
- Includes force feedback sensors in the feet

6. Interactive Controls:
- 'w': Start walking (歩行開始)
- 'u': Toggle UVC on/off
- 'r': Reset position (初期位置)
- 'q': Quit
- Additional experimental modes in uvc_biped.cpp for testing different scenarios

7. Core Walking Algorithm:
- Manages stride length, foot height, and timing
- Integrates UVC feedback for balance
- Handles transition between support leg and swing leg
- Includes ground contact detection and force response

The code represents a unique approach to bipedal walking that relies more on reflexive responses than traditional ZMP-based control. The author notes that while effective, this approach is similar to how a decapitated chicken can still walk using spinal reflexes - it works but has limitations compared to full humanoid control systems.

The implementation is noted as experimental and the author encourages others to improve upon it, particularly in developing more rigorous mathematical models of the UVC approach.