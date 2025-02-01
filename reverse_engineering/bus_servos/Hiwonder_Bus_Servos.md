# Reverse Engineering of Hiwonder LX-224 and HTD-45H Bus Servos

Both the **LX-224** and **HTD-45H** servos share nearly identical hardware architectures, with minor variations in MOSFET driver versions and some IC markings. 

### **Hiwonder LX-224** 20kg serial bus servo.
- PCB Version: 1.4 Number: 2337
- One cpu HL400 (nuvoton mini54) (ARM: HL004 322GC 2320B079-80)
- One 25cz (dual bus buffer gate 74lvc2g125)
- Two j3y (s8050 NPN transistor)
- Two 4606 GA3H2X (MIC4606 Full Bridge MOSFET Driver ) I think
- Two 106 tantalum capacitors
- One VIVC (might be VLVC, or V1VC) 3 pin ic not sure what it is.
- One Crystal (for cpu)
- Nine SMD Resistors (Two 512, Two 331, One 303, One 220, One 105, One 010, One Unknown value)
- Six small SMD capacitors, probably power filtering,

### **Hiwonder HTD-45H** 45kg high-voltage serial bus servo.
- PCB Version: 1.4 Number: 2429
- One cpu HL400 (nuvoton mini54) (ARM: HL004 431GC 2424B030-ZZ)
- One 25cz (dual bus buffer gate 74lvc2g125)
- Two j3y (s8050 NPN transistor)
- Two 4606 GA4F3N (MIC4606 Full Bridge MOSFET Driver ) I think
- Two 106 tantalum capacitors
- One VAXQ (might be VAXO, or VAX0) 3 pin ic not sure what it is.
- One Crystal (for cpu)
- Nine SMD Resistors (Two 512, Two 331, One 303, One 220, One 105, One 010, One Unknown value)
- Six small SMD capacitors, probably power filtering
---

### **Identified Components & Functions**
#### **Microcontroller (HL400 - Nuvoton Mini54)**
- This is the **main CPU** controlling the servo. 
- It likely **processes UART commands** from the bus and **generates PWM** signals to drive the motor.
- The **crystal** (likely 12MHz or 16MHz) provides clock timing.

#### **Bus Interface (25CZ - 74LVC2G125)**
- This is a **dual bus buffer gate**.
- Used for **signal isolation** and **level shifting** between the servo's logic and the external UART bus.
- It may **protect the microcontroller** from high voltages.

#### **Transistors (J3Y - S8050 NPN)**
- Most likely **signal amplifiers or switches**.
- Could be used for **enabling/disabling the driver IC (MIC4606) or controlling MOSFETs**.

#### **Motor Driver (4606 - MIC4606 Full-Bridge MOSFET Driver)**
- This confirms an **H-bridge circuit** for the **DC motor drive**.
- It likely takes **PWM signals** from the microcontroller and drives the **MOSFETs**.

#### **Tantalum Capacitors (106 - 10µF, 10V or 16V)**
- These are **power decoupling capacitors**.
- Likely used to **stabilize the voltage supply** to the **MIC4606 motor driver** and **MCU**.

#### **Unknown 3-Pin IC (VIVC, VLVC, or V1VC)**
- Possible candidates:
  1. **Voltage regulator (3.3V or 5V LDO)** — If this chip is near the microcontroller, it could be a small **SOT-23 LDO voltage regulator** (like AMS1117-3.3).
  2. **Hall Effect sensor / current sensor** — If it’s near motor connections, it could be **monitoring current flow**.
  3. **MOSFET** — If it’s near the power path, it might be a **power-switching FET**.

---

### **Resistor Values & Possible Functions**
- **512 (5.1kΩ), 331 (330Ω), 303 (30kΩ), 220 (22Ω), 105 (1MΩ), 010 (0.1Ω)**  
  These are likely used for:
  - **Current sensing** (low values like **0.1Ω**).
  - **Pull-up or pull-down resistors** for logic signals.
  - **Gate resistors** for MOSFETs (22Ω sounds like a good candidate).

---

### **Possible Layout & Working Mechanism**
1. **UART Command Handling**
   - The servo likely communicates via **TTL UART** using the **HL400 (Mini54)**.
   - The **74LVC2G125** bus buffer isolates and protects the microcontroller.

2. **Motor Control**
   - The **Mini54 MCU** reads the servo position via an **internal potentiometer (not listed but assumed)**.
   - It then **calculates PWM output** and sends signals to the **MIC4606 H-Bridge driver**.
   - The **MIC4606 drives the motor** via MOSFETs, enabling forward/reverse motion.

3. **Feedback & Power Management**
   - The **unknown 3-pin IC** could be:
     - A **voltage regulator** (stepping down from the high voltage bus).
     - A **Hall sensor or current sensor** for torque control.

---

### **Next Steps for Reverse Engineering**
- **Pinout Identification**
  - Find power, signal, and debugging points.
  - Probe TX/RX lines with an oscilloscope or logic analyzer.
- **Check the UART protocol** (it might be standard **half-duplex RS485 or TTL UART**).
  - Determine if the protocol is UART-based, RS485, or a proprietary half-duplex system.
  - Identify baud rate, packet structure, and command set.
- **Check the voltage levels** around the unknown 3-pin IC.
  - Possibly a linear voltage regulator or MOSFET gate driver.
  - Could provide 3.3V or 5V regulation for logic circuits.
- **Use a logic analyzer** on the bus buffer (74LVC2G125) 
  - See **how the Mini54 communicates** with external controllers.
- **Trace the connections** of the MIC4606 motor driver to confirm the H-bridge.
- **PCB Recreation** Design and Develop Raspberry Pi Pico 2 based drop in replacement PCBs.
