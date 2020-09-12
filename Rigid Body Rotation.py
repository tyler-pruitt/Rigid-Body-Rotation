#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:14:41 2020

@author: tylerpruitt
"""


#import packages
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

#get moments of interita and products of inertia to form inertia tensor
inertia_tensor = np.zeros((3, 3), dtype=float)
I_xx = float(input('Enter I_xx: '))
inertia_tensor[0][0] = I_xx
I_xy = float(input('Enter I_xy: '))
I_yx = I_xy
inertia_tensor[0][1] = I_xy
inertia_tensor[1][0] = inertia_tensor[0][1]
I_xz = float(input('Enter I_xz: '))
I_zx = I_xz
inertia_tensor[0][2] = I_xz
inertia_tensor[2][0] = inertia_tensor[0][2]
I_yy = float(input('Enter I_yy: '))
inertia_tensor[1][1] = I_yy
I_yz = float(input('Enter I_yz: '))
I_zy = I_yz
inertia_tensor[1][2] = I_yz
inertia_tensor[2][1] = inertia_tensor[1][2]
I_zz = float(input('Enter I_zz: '))
inertia_tensor[2][2] = I_zz

#convert list of lists into np.array (matrix)
inertia_tensor = np.array(inertia_tensor)

#print out a copy of the inertia tensor as entered
print()
print('Inertia Tensor:')
print(inertia_tensor)
print()

if inertia_tensor[0][1] != 0 or inertia_tensor[0][2] != 0 or inertia_tensor[1][2] != 0:
    (principle_moments, principle_axes) = la.eig(inertia_tensor)
    I_1 = float(principle_moments[0])
    I_2 = float(principle_moments[1])
    I_3 = float(principle_moments[2])
    e_1 = [principle_axes[0][0], principle_axes[1][0], principle_axes[2][0]]
    e_2 = [principle_axes[0][1], principle_axes[1][1], principle_axes[2][1]]
    e_3 = [principle_axes[0][2], principle_axes[1][2], principle_axes[2][2]]
    inertia_tensor = np.zeros((3,3), dtype=float)
    inertia_tensor[0][0] = I_1
    inertia_tensor[1][1] = I_2
    inertia_tensor[2][2] = I_3
    print('Inertia Tensor Along Principle Axes:')
    print(inertia_tensor)
    print()
    print('Principle Axes of Inertia:')
    print(principle_axes)
    print()
else:
    I_1 = float(inertia_tensor[0][0])
    I_2 = float(inertia_tensor[1][1])
    I_3 = float(inertia_tensor[2][2])

assert I_1 > 0, 'I_1 must be positive'
assert I_2 > 0, 'I_2 must be positive'
assert I_3 > 0, 'I_3 must be positive'

omega_1_0 = float(input('Enter inital \u03C9_1: '))
omega_2_0 = float(input('Enter inital \u03C9_2: '))
omega_3_0 = float(input('Enter inital \u03C9_3: '))

def EOM_torque(duration, Gamma_1, Gamma_2, Gamma_3, omega_1, omega_2, omega_3):
    """
    EOM for rigid_body motion with external torques:
    Gamma_1 = I_1 * omega_1_dot + (I_3 - I_2) * omega_2 * omega_3
    Gamma_2 = I_2 * omega_2_dot + (I_1 - I_3) * omega_3 * omega_1
    Gamma_3 = I_3 * omega_3_dot + (I_2 - I_1) * omega_1 * omega_2
    """
    omega_1_dot = (I_2 - I_3) * omega_2 * omega_3 + Gamma_1
    omega_2_dot = (I_3 - I_1) * omega_3 * omega_1 + Gamma_2
    omega_3_dot = (I_1 - I_2) * omega_1 * omega_2 + Gamma_3
    omega_dot_tq = np.array([omega_1_dot, omega_2_dot, omega_3_dot])
    return omega_dot_tq

def EOM_free(omega_1, omega_2, omega_3):
    """
    EOM for free rigid-body motion:
    omega_1_dot = ((I_2 - I_3) / I_1) * omega_3 * omega_2
    omega_2_dot = ((I_3 - I_1) / I_2) * omega_1 * omega_3
    omega_3_dot = ((I_1 - I_2) / I_3) * omega_2 * omega_1
    """
    omega_1_dot = ((I_2 - I_3) / I_1) * omega_3 * omega_2
    omega_2_dot = ((I_3 - I_1) / I_2) * omega_1 * omega_3
    omega_3_dot = ((I_1 - I_2) / I_3) * omega_2 * omega_1
    omega_dot = np.array([omega_1_dot, omega_2_dot, omega_3_dot])
    return omega_dot

motion_state = input("Is there an external torque? Reply either 'Y' for yes or 'N' for no. ")

if motion_state == 'Y':
    #EOM_torque
    duration = float(input('Enter duration of torque: '))
    Gamma_1 = float(input('Enter \u0393_1: '))
    Gamma_2 = float(input('Enter \u0393_2: '))
    Gamma_3 = float(input('Enter \u0393_3: '))
    # integration time step and number of time steps
    iterations=1000
    dt = duration / iterations
    t=np.arange(iterations,dtype=float)*dt
    omega_1=np.zeros(iterations)
    omega_2=np.zeros(iterations)
    omega_3=np.zeros(iterations)
    omega_1[0]=omega_1_0
    omega_2[0]=omega_2_0
    omega_3[0]=omega_3_0
    # integrate Euler equations of motion using RK4 with fixed time step dt
    k1=np.array(3,dtype=float)
    k2=np.array(3,dtype=float)
    k3=np.array(3,dtype=float)
    k4=np.array(3,dtype=float)
    for it in range(1,iterations):
        k1=dt*EOM_torque(duration, Gamma_1, Gamma_2, Gamma_3, omega_1[it-1],omega_2[it-1],omega_3[it-1])
        k2=(dt*EOM_torque(duration, Gamma_1, Gamma_2, Gamma_3, omega_1[it-1]+0.5*k1[0],omega_2[it-1]+0.5*k1[1],
                        omega_3[it-1]+0.5*k1[2]))
        k3=(dt*EOM_torque(duration, Gamma_1, Gamma_2, Gamma_3, omega_1[it-1]+0.5*k2[0],omega_2[it-1]+0.5*k2[1],
                        omega_3[it-1]+0.5*k2[2]))
        k4=(dt*EOM_torque(duration, Gamma_1, Gamma_2, Gamma_3, omega_1[it-1]+k3[0],omega_2[it-1]+k3[1],
                        omega_3[it-1]+k2[2]))
        omega_dot_tq=(k1+2.*k2+2.*k3+k4)/6.
        omega_1[it]=omega_1[it-1]+omega_dot_tq[0]
        omega_2[it]=omega_2[it-1]+omega_dot_tq[1]
        omega_3[it]=omega_3[it-1]+omega_dot_tq[2]
    plt.plot(t,omega_1,label='$\omega_1$')
    plt.plot(t,omega_2,label='$\omega_2$')
    plt.plot(t,omega_3,label='$\omega_3$')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.xlim(0, duration)
    plt.ylabel('$\omega$')
    plt.title('Angular Velocity About Principle Axes (Torque)')
    plt.savefig('rigid_body_rotation_torque_w:'+str(I_1)+','+str(I_2)+','+str(I_3)+'.jpeg',format='jpeg')
    plt.show()
    #see how kinetic energy T changes
    T = (1 / 2) * (I_1 * omega_1 ** 2 + I_2 * omega_2 ** 2 + I_3 * omega_3 ** 2)
    plt.plot(t,T)
    plt.xlabel('t')
    plt.xlim(0, duration)
    plt.ylabel('T')
    plt.title('Kinetic Energy (Torque)')
    plt.savefig('rigid_body_rotation_torque_T:'+str(I_1)+','+str(I_2)+','+str(I_3)+'.jpeg',format='jpeg')
    plt.show()
    #see angular momentum components  L_1, L_2, L_3 along principle axes e_1, e_2, e_3
    L_1 = I_1 * omega_1
    L_2 = I_2 * omega_2
    L_3 = I_3 * omega_3
    plt.plot(t, L_1, label = '$L_1$')
    plt.plot(t, L_2, label = '$L_2$')
    plt.plot(t, L_3, label = '$L_3$')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.xlim(0, duration)
    plt.ylabel('L')
    plt.title('Angular Momentum Along Principle Axes (Torque)')
    plt.savefig('rigid_body_rotation_torque_L:'+str(I_1)+','+str(I_2)+','+str(I_3)+'.jpeg,',format='jpeg')
    plt.show()
    omega_1_0 = omega_1[-1]
    omega_2_0 = omega_2[-1]
    omega_3_0 = omega_3[-1]
    print('Calculations Successful.')
    keep_going = input("Analyze motion past duration of torque? Reply either 'Y' for yes or 'N' of no. ")
    if keep_going == 'Y':
        #EOM_free
        # integration time step and number of time steps
        time = float(input('Enter duration of plot: '))
        dt=0.1/np.amax([omega_1_0,omega_2_0,omega_3_0])
        iterations=1000
        t=np.arange(iterations,dtype=float)*dt
        omega_1=np.zeros(iterations)
        omega_2=np.zeros(iterations)
        omega_3=np.zeros(iterations)
        omega_1[0]=omega_1_0
        omega_2[0]=omega_2_0
        omega_3[0]=omega_3_0
        # integrate Euler equations of motion using RK4 with fixed time step dt
        k1=np.array(3,dtype=float)
        k2=np.array(3,dtype=float)
        k3=np.array(3,dtype=float)
        k4=np.array(3,dtype=float)
        for it in range(1,iterations):
            k1=dt*EOM_free(omega_1[it-1],omega_2[it-1],omega_3[it-1])
            k2=(dt*EOM_free(omega_1[it-1]+0.5*k1[0],omega_2[it-1]+0.5*k1[1],
                        omega_3[it-1]+0.5*k1[2]))
            k3=(dt*EOM_free(omega_1[it-1]+0.5*k2[0],omega_2[it-1]+0.5*k2[1],
                        omega_3[it-1]+0.5*k2[2]))
            k4=(dt*EOM_free(omega_1[it-1]+k3[0],omega_2[it-1]+k3[1],
                        omega_3[it-1]+k2[2]))
            omega_dot=(k1+2.*k2+2.*k3+k4)/6.
            omega_1[it]=omega_1[it-1]+omega_dot[0]
            omega_2[it]=omega_2[it-1]+omega_dot[1]
            omega_3[it]=omega_3[it-1]+omega_dot[2]
        plt.plot(t,omega_1,label='$\omega_1$')
        plt.plot(t,omega_2,label='$\omega_2$')
        plt.plot(t,omega_3,label='$\omega_3$')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.xlim(0, time)
        plt.ylabel('$\omega$')
        plt.title('Angular Velocity About Principle Axes (Free)')
        plt.savefig('rigid_body_rotation_free_w:'+str(I_1)+','+str(I_2)+','+str(I_3)+'.jpeg',format='jpeg')
        plt.show()
        #see if kinetic energy T is conserved
        T = (1 / 2) * (I_1 * omega_1 ** 2 + I_2 * omega_2 ** 2 + I_3 * omega_3 ** 2)
        plt.plot(t,T)
        plt.xlabel('t')
        plt.xlim(0, time)
        plt.ylabel('T')
        plt.title('Kinetic Energy (Free)')
        plt.savefig('rigid_body_rotation_free_T:'+str(I_1)+','+str(I_2)+','+str(I_3)+'.jpeg',format='jpeg')
        plt.show()
        #see angular momentum components  L_1, L_2, L_3 along principle axes e_1, e_2, e_3
        L_1 = I_1 * omega_1
        L_2 = I_2 * omega_2
        L_3 = I_3 * omega_3
        plt.plot(t, L_1, label = '$L_1$')
        plt.plot(t, L_2, label = '$L_2$')
        plt.plot(t, L_3, label = '$L_3$')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.xlim(0, time)
        plt.ylabel('L')
        plt.title('Angular Momentum Along Principle Axes (Free)')
        plt.savefig('rigid_body_rotation_free_L:'+str(I_1)+','+str(I_2)+','+str(I_3)+'.jpeg',format='jpeg')
        plt.show()
        print('Calculations Successful.')
    print('End.')
elif motion_state == 'N':
    #EOM_free
    # integration time step and number of time steps
    time = float(input('Enter duration of plot: '))
    dt=0.1/np.amax([omega_1_0,omega_2_0,omega_3_0])
    iterations=1000
    t=np.arange(iterations,dtype=float)*dt
    omega_1=np.zeros(iterations)
    omega_2=np.zeros(iterations)
    omega_3=np.zeros(iterations)
    omega_1[0]=omega_1_0
    omega_2[0]=omega_2_0
    omega_3[0]=omega_3_0
    # integrate Euler equations of motion using RK4 with fixed time step dt
    k1=np.array(3,dtype=float)
    k2=np.array(3,dtype=float)
    k3=np.array(3,dtype=float)
    k4=np.array(3,dtype=float)
    for it in range(1,iterations):
        k1=dt*EOM_free(omega_1[it-1],omega_2[it-1],omega_3[it-1])
        k2=(dt*EOM_free(omega_1[it-1]+0.5*k1[0],omega_2[it-1]+0.5*k1[1],
                        omega_3[it-1]+0.5*k1[2]))
        k3=(dt*EOM_free(omega_1[it-1]+0.5*k2[0],omega_2[it-1]+0.5*k2[1],
                        omega_3[it-1]+0.5*k2[2]))
        k4=(dt*EOM_free(omega_1[it-1]+k3[0],omega_2[it-1]+k3[1],
                        omega_3[it-1]+k2[2]))
        omega_dot=(k1+2.*k2+2.*k3+k4)/6.
        omega_1[it]=omega_1[it-1]+omega_dot[0]
        omega_2[it]=omega_2[it-1]+omega_dot[1]
        omega_3[it]=omega_3[it-1]+omega_dot[2]
    plt.plot(t,omega_1,label='$\omega_1$')
    plt.plot(t,omega_2,label='$\omega_2$')
    plt.plot(t,omega_3,label='$\omega_3$')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.xlim(0, time)
    plt.ylabel('$\omega$')
    plt.title('Angular Velocity About Principle Axes (Free)')
    plt.savefig('rigid_body_rotation_free_w:'+str(I_1)+','+str(I_2)+','+str(I_3)+'.jpeg',format='jpeg')
    plt.show()
    #see if kinetic energy T is conserved
    T = (1 / 2) * (I_1 * omega_1 ** 2 + I_2 * omega_2 ** 2 + I_3 * omega_3 ** 2)
    plt.plot(t,T)
    plt.xlabel('t')
    plt.xlim(0, time)
    plt.ylabel('T')
    plt.title('Kinetic Energy (Free)')
    plt.savefig('rigid_body_rotation_free_T:(I_1,I_2,I_3)=('+str(I_1)+','+str(I_2)+','+str(I_3)+'.jpeg',format='jpeg')
    plt.show()
    #see angular momentum components  L_1, L_2, L_3 along principle axes e_1, e_2, e_3
    L_1 = I_1 * omega_1
    L_2 = I_2 * omega_2
    L_3 = I_3 * omega_3
    plt.plot(t, L_1, label = '$L_1$')
    plt.plot(t, L_2, label = '$L_2$')
    plt.plot(t, L_3, label = '$L_3$')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.xlim(0, time)
    plt.ylabel('L')
    plt.title('Angular Momentum Along Principle Axes (Free)')
    plt.savefig('rigid_body_rotation_free_L:'+str(I_1)+','+str(I_2)+','+str(I_3)+'.jpeg',format='jpeg')
    plt.show()
    print('Calculations Successful.')
    print('End.')
else:
    print("Response must be either 'Y' or 'N'.")
    print('Calculations Unsuccessful.')
    print('End.')