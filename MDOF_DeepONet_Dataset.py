## Import Libaries -------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt 
from time import time
import tensorflow as tf
import scipy.io as sp

## Class Training Dataset -------------------------------------------------------

def Training_Dataset(samples, min_t, max_t, dt, ft, am_start, am_end, fr_start, fr_end, M1, M2, M3, M4, M5, alpha, C1, C2, C3, C4, C5, K1, K2, K3, K4, K5, x_init, x_dot_init):

    # time array
    t = np.arange(min_t, max_t, dt)
    lt = t.shape[0]

    # Runge-Kutta Method
    def rkm(y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, f, t):

        F1 = lambda Y: Y[6]
        F2 = lambda Y: Y[7]
        F3 = lambda Y: Y[8]
        F4 = lambda Y: Y[9]
        F5 = lambda Y: Y[10]
        F6 = lambda Y: (1/M1)*(-M1*f-(C1*Y[5+1]+K1*Y[1]+C2*(Y[5+1]-Y[5+2])+K2*(Y[1]-Y[2])+alpha*Y[1]**3))
        F7 = lambda Y: (1/M2)*(-M2*f-(C2*(Y[5+2]-Y[5+1])+K2*(Y[2]-Y[1])+C3*(Y[5+2]-Y[5+3])+K3*(Y[2]-Y[3])))
        F8 = lambda Y: (1/M3)*(-M3*f-(C3*(Y[5+3]-Y[5+2])+K3*(Y[3]-Y[2])+C4*(Y[5+3]-Y[5+4])+K5*(Y[3]-Y[4])))
        F9 = lambda Y: (1/M4)*(-M4*f-(C4*(Y[5+4]-Y[5+3])+K4*(Y[4]-Y[3])+C5*(Y[5+4]-Y[5+5])+K5*(Y[4]-Y[5])))
        F10 = lambda Y: (1/M5)*(-M5*f-(C5*(Y[5+5]-Y[5+4])+K5*(Y[5]-Y[4])))

        k0 = dt*F1([t, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]);
        l0 = dt*F2([t, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]);
        m0 = dt*F3([t, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]);
        n0 = dt*F4([t, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]);
        o0 = dt*F5([t, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]);
        p0 = dt*F6([t, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]);
        q0 = dt*F7([t, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]);
        r0 = dt*F8([t, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]);
        s0 = dt*F9([t, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]);
        t0 = dt*F10([t, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]);

        k1 = dt*F1([t+0.5*dt, y1+0.5*k0, y2+0.5*l0, y3+0.5*m0, y4+0.5*n0, y5+0.5*o0, y6+0.5*p0, y7+0.5*q0, y8+0.5*r0, y9+0.5*s0, y10+0.5*t0]);
        l1 = dt*F2([t+0.5*dt, y1+0.5*k0, y2+0.5*l0, y3+0.5*m0, y4+0.5*n0, y5+0.5*o0, y6+0.5*p0, y7+0.5*q0, y8+0.5*r0, y9+0.5*s0, y10+0.5*t0]);
        m1 = dt*F3([t+0.5*dt, y1+0.5*k0, y2+0.5*l0, y3+0.5*m0, y4+0.5*n0, y5+0.5*o0, y6+0.5*p0, y7+0.5*q0, y8+0.5*r0, y9+0.5*s0, y10+0.5*t0]);
        n1 = dt*F4([t+0.5*dt, y1+0.5*k0, y2+0.5*l0, y3+0.5*m0, y4+0.5*n0, y5+0.5*o0, y6+0.5*p0, y7+0.5*q0, y8+0.5*r0, y9+0.5*s0, y10+0.5*t0]);
        o1 = dt*F5([t+0.5*dt, y1+0.5*k0, y2+0.5*l0, y3+0.5*m0, y4+0.5*n0, y5+0.5*o0, y6+0.5*p0, y7+0.5*q0, y8+0.5*r0, y9+0.5*s0, y10+0.5*t0]);
        p1 = dt*F6([t+0.5*dt, y1+0.5*k0, y2+0.5*l0, y3+0.5*m0, y4+0.5*n0, y5+0.5*o0, y6+0.5*p0, y7+0.5*q0, y8+0.5*r0, y9+0.5*s0, y10+0.5*t0]);
        q1 = dt*F7([t+0.5*dt, y1+0.5*k0, y2+0.5*l0, y3+0.5*m0, y4+0.5*n0, y5+0.5*o0, y6+0.5*p0, y7+0.5*q0, y8+0.5*r0, y9+0.5*s0, y10+0.5*t0]);
        r1 = dt*F8([t+0.5*dt, y1+0.5*k0, y2+0.5*l0, y3+0.5*m0, y4+0.5*n0, y5+0.5*o0, y6+0.5*p0, y7+0.5*q0, y8+0.5*r0, y9+0.5*s0, y10+0.5*t0]);
        s1 = dt*F9([t+0.5*dt, y1+0.5*k0, y2+0.5*l0, y3+0.5*m0, y4+0.5*n0, y5+0.5*o0, y6+0.5*p0, y7+0.5*q0, y8+0.5*r0, y9+0.5*s0, y10+0.5*t0]);
        t1 = dt*F10([t+0.5*dt, y1+0.5*k0, y2+0.5*l0, y3+0.5*m0, y4+0.5*n0, y5+0.5*o0, y6+0.5*p0, y7+0.5*q0, y8+0.5*r0, y9+0.5*s0, y10+0.5*t0]);
    
        k2 = dt*F1([t+0.5*dt, y1+0.5*k1, y2+0.5*l1, y3+0.5*m1, y4+0.5*n1, y5+0.5*o1, y6+0.5*p1, y7+0.5*q1, y8+0.5*r1, y9+0.5*s1, y10+0.5*t1]);
        l2 = dt*F2([t+0.5*dt, y1+0.5*k1, y2+0.5*l1, y3+0.5*m1, y4+0.5*n1, y5+0.5*o1, y6+0.5*p1, y7+0.5*q1, y8+0.5*r1, y9+0.5*s1, y10+0.5*t1]);
        m2 = dt*F3([t+0.5*dt, y1+0.5*k1, y2+0.5*l1, y3+0.5*m1, y4+0.5*n1, y5+0.5*o1, y6+0.5*p1, y7+0.5*q1, y8+0.5*r1, y9+0.5*s1, y10+0.5*t1]);
        n2 = dt*F4([t+0.5*dt, y1+0.5*k1, y2+0.5*l1, y3+0.5*m1, y4+0.5*n1, y5+0.5*o1, y6+0.5*p1, y7+0.5*q1, y8+0.5*r1, y9+0.5*s1, y10+0.5*t1]);
        o2 = dt*F5([t+0.5*dt, y1+0.5*k1, y2+0.5*l1, y3+0.5*m1, y4+0.5*n1, y5+0.5*o1, y6+0.5*p1, y7+0.5*q1, y8+0.5*r1, y9+0.5*s1, y10+0.5*t1]);
        p2 = dt*F6([t+0.5*dt, y1+0.5*k1, y2+0.5*l1, y3+0.5*m1, y4+0.5*n1, y5+0.5*o1, y6+0.5*p1, y7+0.5*q1, y8+0.5*r1, y9+0.5*s1, y10+0.5*t1]);
        q2 = dt*F7([t+0.5*dt, y1+0.5*k1, y2+0.5*l1, y3+0.5*m1, y4+0.5*n1, y5+0.5*o1, y6+0.5*p1, y7+0.5*q1, y8+0.5*r1, y9+0.5*s1, y10+0.5*t1]);
        r2 = dt*F8([t+0.5*dt, y1+0.5*k1, y2+0.5*l1, y3+0.5*m1, y4+0.5*n1, y5+0.5*o1, y6+0.5*p1, y7+0.5*q1, y8+0.5*r1, y9+0.5*s1, y10+0.5*t1]);
        s2 = dt*F9([t+0.5*dt, y1+0.5*k1, y2+0.5*l1, y3+0.5*m1, y4+0.5*n1, y5+0.5*o1, y6+0.5*p1, y7+0.5*q1, y8+0.5*r1, y9+0.5*s1, y10+0.5*t1]);
        t2 = dt*F10([t+0.5*dt, y1+0.5*k1, y2+0.5*l1, y3+0.5*m1, y4+0.5*n1, y5+0.5*o1, y6+0.5*p1, y7+0.5*q1, y8+0.5*r1, y9+0.5*s1, y10+0.5*t1]);

        k3 = dt*F1([t+dt, y1+k2, y2+l2, y3+m2, y4+n2, y5+o2, y6+p2, y7+q2, y8+r2, y9+s2, y10+t2]);
        l3 = dt*F2([t+dt, y1+k2, y2+l2, y3+m2, y4+n2, y5+o2, y6+p2, y7+q2, y8+r2, y9+s2, y10+t2]);
        m3 = dt*F3([t+dt, y1+k2, y2+l2, y3+m2, y4+n2, y5+o2, y6+p2, y7+q2, y8+r2, y9+s2, y10+t2]);
        n3 = dt*F4([t+dt, y1+k2, y2+l2, y3+m2, y4+n2, y5+o2, y6+p2, y7+q2, y8+r2, y9+s2, y10+t2]);
        o3 = dt*F5([t+dt, y1+k2, y2+l2, y3+m2, y4+n2, y5+o2, y6+p2, y7+q2, y8+r2, y9+s2, y10+t2]);
        p3 = dt*F6([t+dt, y1+k2, y2+l2, y3+m2, y4+n2, y5+o2, y6+p2, y7+q2, y8+r2, y9+s2, y10+t2]);
        q3 = dt*F7([t+dt, y1+k2, y2+l2, y3+m2, y4+n2, y5+o2, y6+p2, y7+q2, y8+r2, y9+s2, y10+t2]);
        r3 = dt*F8([t+dt, y1+k2, y2+l2, y3+m2, y4+n2, y5+o2, y6+p2, y7+q2, y8+r2, y9+s2, y10+t2]);
        s3 = dt*F9([t+dt, y1+k2, y2+l2, y3+m2, y4+n2, y5+o2, y6+p2, y7+q2, y8+r2, y9+s2, y10+t2]);
        t3 = dt*F10([t+dt, y1+k2, y2+l2, y3+m2, y4+n2, y5+o2, y6+p2, y7+q2, y8+r2, y9+s2, y10+t2]);

        y1 = y1+(1/6)*(k0+2*k1+2*k2+k3);
        y2 = y2+(1/6)*(l0+2*l1+2*l2+l3);
        y3 = y3+(1/6)*(m0+2*m1+2*m2+m3);
        y4 = y4+(1/6)*(n0+2*n1+2*n2+n3);
        y5 = y5+(1/6)*(o0+2*o1+2*o2+o3);
        y6 = y6+(1/6)*(p0+2*p1+2*p2+p3);
        y7 = y7+(1/6)*(q0+2*q1+2*q2+q3);
        y8 = y8+(1/6)*(r0+2*r1+2*r2+r3);
        y9 = y9+(1/6)*(s0+2*s1+2*s2+s3);
        y10 = y10+(1/6)*(t0+2*t1+2*t2+t3);
    

        return y1, y2, y3, y4, y5, y6, y7, y8, y9, y10

    y1 = np.zeros([lt, samples])+x_init
    y2 = np.zeros([lt, samples])+x_init
    y3 = np.zeros([lt, samples])+x_init
    y4 = np.zeros([lt, samples])+x_init
    y5 = np.zeros([lt, samples])+x_init
    y6 = np.zeros([lt, samples])+x_dot_init
    y7 = np.zeros([lt, samples])+x_dot_init
    y8 = np.zeros([lt, samples])+x_dot_init
    y9 = np.zeros([lt, samples])+x_dot_init
    y10 = np.zeros([lt, samples])+x_dot_init

    am = np.random.uniform(am_start, am_end, [samples,ft])
    fr = np.random.uniform(fr_start, fr_end, [samples,ft])
    s1 = np.arange(0,ft,2)
    s2 = np.arange(1,ft,2)
    f = np.zeros([lt, samples])
    for i in range(0, samples):
        f[:, i] = (np.sum([am[i,j]*np.sin(fr[i,j]*t) for j in s1],0,keepdims=True) +
                np.sum([am[i,j]*np.cos(fr[i,j]*t) for j in s2],0,keepdims=True))

    for i in range(0, samples):
        if i%100 ==  0 or i == samples-1: print('sample number  = ', i) 
        for j in range(1, lt):
            y1[j, i], y2[j, i], y3[j, i], y4[j, i], y5[j, i], y6[j, i], y7[j, i], y8[j, i], y9[j, i], y10[j, i] = rkm(y1[j-1, i], y2[j-1, i], y3[j-1, i], y4[j-1, i], y5[j-1, i], y6[j-1, i], y7[j-1, i], y8[j-1, i], y9[j-1, i], y10[j-1, i], f[j-1, i], t[j-1])

    sp.savemat('data_5DOF_duffing_FT'+str(ft)+'_Samples'+str(samples)+'.mat',{'f': f.T, 'y1': y1.T, 'y2': y2.T, 'y3': y3.T, 'y4': y4.T, 'y5': y5.T, 'am': am, 'fr': fr, 't': t, 'dt': dt})