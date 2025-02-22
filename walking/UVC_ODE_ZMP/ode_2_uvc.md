ODE source code 2（UVC application）
Download the following 4 files and add them to your project using ODE's Solution Explorer,
it will work.

uvc_biped.cpp download (Updated on June 1,2021)
This is the ODE main routine. Here, a 3D object is generated,
and the 3D robot is driven by the joint drive amount specified by the walking control unit below.

uvc_biped.h download
ODE main routine header file.

uvc_core.cpp download (Updated on June 1,2021)
This is the main body of the walking control unit. The three elements of walking,
① basic motion, ② postural reflex (UVC), and ③ timing control (CPG) are integrated
to generate walking motion.

uvc_core.h download (Updated on June 1,2021)
Header file of the walking control unit.

Explain the operation.
The following operations can be performed by key input.

w:　Start walking

1:　Experiment of applying external force
Set the experimental environment to push the robot from behind.

2:　Experiment to start walking naturally
Tilt the robot's upper body forward and start walking naturally.

3:　Experiment to walk on a slope (provisional specification)
The integrated value of the tilt angle of the robot is superimposed on the upper body angle to enable walking on slopes.

4:　Experiment to traverse steps (provisional specifications)
If the ground contact of the swing leg cannot be detected, the knee of the support leg is bent.

5:　Apply all algorithms (provisional specifications)

u:　Specify UVC (upper body vertical control) enable / disable as an alternative.
　　When the body is blue, UVC is enabled,and when it is red, UVC is disabled.

r:　Reset (return to the initial position)

q:　Finished

※3-4 is a provisional specification, and there is still room for improvement.
