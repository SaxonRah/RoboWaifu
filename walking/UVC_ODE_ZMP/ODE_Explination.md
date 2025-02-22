```UVC (upper vertical control) description

■Overview
Posture reflection ( like humans without complex mechanics *1 Balance only by ).

■principle
The postural reflection used here is to move the hip joint of the landing side of the leg,
Control that the upper body is always vertical.
For example, when the upper body leans forward, it is returned back by spreading the crotch more.
On the contrary, when leaning back, narrow your crotch ⁇ , and when walking, shake your legs
Slow down and put the upper body back in front. In a similar way if you lean sideways Keep the upper body vertically.

How the upper body angle changes when the hip is moved on the grounded side Confirmed by physical simulation. Less than 
[http://www.youtube.com/v/1Wc7Idd7vjU]
weightless ⁇ 3 sticks each with blue on the upper body and white on the legs Fix one of the white l sticks
to the ground with a free joint, Spread the hip joint (the white and blue bar connections) to the right
The upper body leans to the left, and if spread in the opposite direction, the upper body leans to the right,
Upper body vertical control (UVC) Use this effect to keep the upper body always vertical.

For this robot, the balance control is only UVC, ZMP ( *2 )etc. No mechanical calculations are made aware of.
Gyro ( for vibration deterrence often used in hobby robots *3 ) Not applied.

■walking control
UVC is not available for walking control and is not available alone,
Achieve walking behavior by synthesizing three elements (hereinafter) that are suitable for walking.

[uvc_prin.png]

①walking motion
　Stable that can be walked by itself if no external force is applied Orbital reproduction walking motion (fixed motion *Four )

②Posture reflection
　UVC (upper vertical control), the cornerstone of balance control.

③Sync (CPG *Five )
　Generation of walking rhythms (periods) that synchronize with the specific frequency of the robot.
  However, this simulation has a fixed period (simplified specification).

The three elements of the above are integrated to achieve a stable course of action.

■simulation program
For the entire program, in the ODE main part and in the walking control department Greatly separated,
the main part of ODE is the generation of 3D objects and the walking control part
Move the object (robot) according to the instructions.
On the other hand, the walking control department is the main part of the walking process that
produces the walking pattern, ① walking motion, ② posture reflection, ③ integrated synchronization,
Generate the joint drive amount of the robot in real time.
(The walking control department is made up of a short program of 100 steps)
(Publish source code) There is no detailed optimization, so there is room for improvement and improvement,
Free to post and redistribute books. It is also free to apply and publish UVC to non-commercial items.

■About robot movement
In the video, the robot contacts the bar and pushes it back,
There is a scene where you stumble on a sphere and step on it,
This is not specifically embedded in such behavior Generated irregularly and voluntarily due to the effect of UVC.

■Challenge
Walking control using only UVC is such that decapitated chickens are spinal reflections
only It is like walking and cannot handle various external conditions. ( *6 )
To be a versatile humanoid robot, Human kinematic function (simple behavior),
kinetic coalition function (set of behavior) Cerebellar function (correction of error) and
frontal lobe function (construction and operation of external model) Central system functions are also required.
It's not just one-way information processing like deep learning, but also processing for itself
(Self-reflection, image manipulation) Build a outside world model (It requires a body with access
to the real world) Ability to manipulate (think) it Robot in preparation Must evolve into.


------------------------ Supplementary explanation ------------------------
*1 : For humans, bounce reflexes (keeping the head and trunk in the correct position in space,
Function to correct the relative relationship between the head and trunk),
parachute reflection (Stap back when pressed from front), extension reflex, tension maze reflex,
There are some reflections related to upper vertical retention, such as tension cervical reflexes,
Maintain a walking balance with them.

*2 : Center of pressure applied to the soles, usually called ZMP
( ⁇ also serves gravity and acceleration of the center of gravity) Come to the center of the supporting polygon
that contacts the ground (for example, an equilateral triangle for tripods) Decide to carry your legs.
Motion equations and various calculations that require it.

*3 : The output (acceleration) of the rate gyro is usually overlapped with the
joint drive of the roll and pitch axes of the ankle, Deter the shaking of the aircraft
by making a trampling movement in the tilted direction.

*4 : In the case of humans, if the trunk moves to the left side, the legs on the
right side are reflexively raised, etc. The basic walking movements themselves are also supplemented by reflections.

*5 : A pattern generator (central pattern generator) in the spinal cord,
Generate a walking rhythm. Use some feedback signal Adjust the output pattern and
period so that the walking rhythm does not collapse.

*6 : Humanoid robot development is stopped or shelved There are many cases.
Robots just walk in good condition or do some show ( The current level ) may not find much meaning.

```

```ODE source code

biped.cpp download (Updated on June 1,2021)
This is the ODE main routine. Here, a 3D object is generated,
and the 3D robot is driven by the joint drive amount specified by the walking control unit below.

biped.h download
ODE main routine header file.

core.cpp download (Updated on June 1,2021)
This is the main body of the walking control unit. The three elements of walking,
① basic motion, ② postural reflex (UVC), and ③ timing control (CPG) are integrated
to generate walking motion.

core.h download (Updated on June 1,2021)
Header file of the walking control unit.

Explain the operation.
The following operations can be performed by key input.
w:　Start walking
u:　Specify UVC (upper body vertical control) enable / disable as an alternative.
　　When the body is blue, UVC is enabled,and when it is red, UVC is disabled.
r:　Reset (return to the initial position)
q:　Finished
```

```UVC Question and Answer
Q.1
We see that you have published the source code,
but we’re having trouble deciphering the mathematical model and governing equations of the system. Specifically,
many of the variable explanations get lost in translation.
We were wondering if you would be comfortable sharing any labeled models or equations that you used for this project.
A.1
Currently, I am in the stage of verifying the mechanical image by physics simulation.
They have demonstrated the effectiveness of UVC, but a clear mathematical model has not yet been constructed.
I hope that other research institutes will pursue the mathematical model of UVC.
Also, since it is in the experimental stage and trial and error is repeated,
the comments on variables etc. are very rough.


Q.2
We attempted to model your joint lengths and angles,
but failed to understand angle k (K0 knee bending angle) and angle x (K0 leg swing angle).
Are these approximations or is there something that we are perhaps misunderstanding?
A.2
① There was an error in the calculation of x（K0 leg swing angle).
x=asin(x/LEG)　⇒　x = asin(x/k)

② Another. There was a discrepancy between the ODE object settings and the parameter settings.
#define LEG 190.0　⇒　#define LEG 180.0

It doesn't matter so much in operation,
but it is mathematically inconsistent, so I fixed it.
Updated core.cpp.
Please refer to the following memo for the explanation about the
calculation of the hip joint angle and the knee joint angle.
[qa.jpg]


Q.3
It seems that K1W and K2W are both used for hip joint lateral writing (for hip joints, for lateral writing).
Was this done intentionally? If so, would you be willing to explain the purpose of that to us?
A.3
K2W is for hip yaw axis and is not used in this simulation.
I deleted it so that there is no misunderstanding.
Updated core.h and biped.cpp.


Q.4
We’ve observed that dxi and dyi are used frequently in the UVC section of your code.
Even after thorough investigation, we were unable to figure out what specifically these variables represent.
Is this something you would be able to show us?
A.4
dxi is equivalent to I in PID control, and is for converting the tilt angle in the pitch direction
into the distance in the front-back direction and integrating it. By superimposing dxi on the stride length,
the front-back direction of the upper body is kept vertical. dyi has the same idea as dxi,
and keeps it vertical with respect to the lateral direction of the upper body. The basic idea of UVC is to superimpose
the inclination of the upper body on the hip joint angle and keep the upper body vertical at all times.
In this simulation, the same effect is obtained by converting the tilt angle into the stride for convenience.


Q.5
Your code heavily relies on the alternation between support leg and free leg.
Many variables are arrays of 2 that house only 0 and 1.
In some instances, 0 indicates support leg and 1 indicates the free leg.
However, we are not sure if that is the case for all variables of this type,
and we theorize that some of them indicate the left and right legs. Is this assumption correct?
A.5
jikuasi is a Japanese translation of the pivot foot and represents the foot on the grounded side.
When jikuasi is 0, the right foot is in contact with the ground, and when it is 1,
the left foot is in contact with the ground.


Q.6
In this video when the hand pushes the robot (in the experiment), how does this trigger the movement in the code?
A.6
In the actual machine, the walking motion is started when a certain inclination is detected,
and when the walking speed exceeds a certain speed, the walking speed is maintained.
These are convenient means only for the actual machine and are not implemented in the ODE simulation software.
UVC can be used not only for maintaining balance,
but also for various applications such as walking start control and constant-velocity walking.


Q.7
You are using head angular movement and rotation,to adjust movement of the body.

Rot = dBodyGetRotation(HEADT.b);
fbRad = asin(Rot[8]);
lrRad = asin(Rot[9]);
Rot = dBodyGetAngularVel(HEADT.b);
fbAV = Rot[1];
lrAV = Rot[0];

What if make invisible sphere (or 2 invisible circles) with the center in loin,
and use it as calculation parameter for torso movement?
A.7
fbAV and lrAV are used in the footCont(float x,float y,float h,int s) function,
but they are currently disabled.
In other words, at present, only the angle(fbRad and lrRad) is valid,
so even if you measure the angle with a sphere centered on the waist or calculate with the current head object,
the behavior of the robot will not change.


Q.8
You mention dxi is like the integral (I) term in PID. In my own code,
I have renamed dxi to I_pitch and dyi to I_roll (relative to the hip). Would this be correct?
A.8
It is correct. For programming reasons, I control the robot's stride to keep its upper body vertical,
but in principle it is more direct and easier to evaluate by superimposing deviations
on the pitch angle and roll angle as you did. I think.


Q.9
Are you considering a proportional term or differential term?
A.9
UVC does not try to maintain any equilibrium point like the general PID control target,
but merely keeps the upper body vertical. 
Therefore, in UVC, using the differential term or proportional term may be rather unstable. 
In my past productions, tightrope walking robots, all the terms of PID (maintaining the equilibrium point) are applied.


Q.10
You mention: "By superimposing dxi on the stride length, the front-back direction of the upper body is kept vertical."
Would you be able to clarify what you mean by this?
A.10
The current calculation method may have been too rough for how to keep the upper body angle vertical
by controlling the stride length. For the time being, it worked, so it remains as it is,
but it is a considerable omission. As mentioned above, the method of superimposing the deviation of the
upper body angle on the pitch angle and roll angle of the supporting leg hip joint is a little complicated,
but I think that better results can be obtained.


Q.11
Are you essentially trying to cancel out the angle of the upper body relative the waist
by producing joint angles in "the opposite direction"?
A.11
That's right. The basis of UVC is to superimpose (integrate) the inclination (deviation) of the
upper body with respect to the vertical on the hip joint angle and try to rebuild the upper body vertically.


Q.12
Would you be able to tell me how you determined values for certain variables? In particular,
I am a bit uncertain about where to get the following variables for our robot: -autoH -autoHs -adjFr
A.12
autoHs is a variable that stores the maximum length from the robot's hip to ankle.
(Maximum length is 180) autoHs is a variable that stores the standard length from the robot's hip to ankle.
(Standard length is 170) The standard length is slightly shorter than the maximum length so that the
knee is slightly bent so that no singularity occurs when the swing leg touches the ground.
Also, when the robot is started, the knees are slowly bent by subtracting autoHs little by little to
gently shift to the idle state. adjFr is for storing the value to correct the position of the
center of gravity of the robot that is slightly shifted forward in the idle state.


Q.13
I am assuming fh and fhMax stands for foot height, but I am not certain
A.13
fh is a variable that represents the height of the foot and is a sine curve. The peak value of the sine curve is fhMax.


Q.14
Additionally, would you mind explaining where you got these values (or weights) in UVC: 1.5, 193, 1.5, 130?
k = 1.5 * 193 * sin(lrRad); //// Left-right displacement ////
k = 1.5 * 130 * sin(fbRad); //// Forward-backward displacement ////
A.14
1.5 is the value obtained experimentally (generally the optimum value).
It doesn't make much sense to explain the background of 193 and 130, and when 1.5 * 193=289.5 and 1.5*130=195
are set as the coefficients, the movement becomes relatively stable.


Q.15
Please let me know if I can clarify anything. If there are any issues, please let me know.
A.15
The sample code is just one example of how UVC works, it's not optimized and it's very rough.
So it may not make much sense to dig into the details.
Rather, I hope that you will understand the basic concept of UVC, and based on that,
you will be able to try out your own ideas and take on the challenge of applying it to actual machines.

```
