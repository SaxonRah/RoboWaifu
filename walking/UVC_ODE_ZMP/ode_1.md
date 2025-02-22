ODE source code
Download the following 4 files and add them to your project using ODE's Solution Explorer,
it will work.

ode_biped.cpp download (Updated on June 1,2021)
This is the ODE main routine. Here, a 3D object is generated,
and the 3D robot is driven by the joint drive amount specified by the walking control unit below.

ode_biped.h download
ODE main routine header file.

ode_core.cpp download (Updated on June 1,2021)
This is the main body of the walking control unit. The three elements of walking,
① basic motion, ② postural reflex (UVC), and ③ timing control (CPG) are integrated
to generate walking motion.

ode_core.h download (Updated on June 1,2021)
Header file of the walking control unit.

Explain the operation.
The following operations can be performed by key input.
w:　Start walking
u:　Specify UVC (upper body vertical control) enable / disable as an alternative.
　　When the body is blue, UVC is enabled,and when it is red, UVC is disabled.
r:　Reset (return to the initial position)
q:　Finished

