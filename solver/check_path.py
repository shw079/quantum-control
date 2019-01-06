import numpy as np
import solver.constants as const
import matplotlib.pyplot as plt
import solver.solvers as s

w1 = const.B/const.hbar
n = 100000
tf=100*const.hbar/const.B
dt=tf/n
t = np.linspace(0,tf,num=n)
Q = 0.25*tf;
B2 = 0.0002;
v = 0.2;

def cosphi(m):
    """Calculates for x component of path """
    cosphi_input=np.full((2*m),0.5)
    cosphi=np.diag(cosphi_input,k=1)+np.diag(cosphi_input,k=-1)
    return cosphi

def sinphi(m):
    """Calculates for y component of path """
    sinphi_input1=np.full((2*m),0.5j)
    sinphi_input2=np.full((2*m),-0.5j)
    sinphi=np.diag(sinphi_input1,k=1)+np.diag(sinphi_input2,k=-1)
    return sinphi

def init():
    w1 = const.B/const.hbar
    n = 100000
    tf=100*const.hbar/const.B
    dt=tf/n
    t = np.linspace(0,tf,num=n)
    Q = 0.25*tf;
    B2 = 0.0002;
    v = 0.2;

    state1 = np.zeros(int(2*const.m+1))
    state1[8] = 1
    state1 = state1.reshape(1,2*const.m+1)
    state2 = np.zeros(int(2*const.m+1))
    state2[8] = 1
    state2 = state2.reshape(2*const.m+1,1)

    sigmoid = 1/((1+Q*np.exp(-B2*t))**(1/v))
    O1 = 0.95*(t/tf)*np.sin(0.5*w1*t)*sigmoid
    O2 = 0.95*(t/tf)*np.cos(0.5*w1*t)*sigmoid;
    print(state1.shape)
    print(cosphi(const.m).shape)
    print(state2.shape)
    for i in range(len(O1)):
        O1[i] = O1[i] - (O1[0]-np.asscalar(state1@cosphi(const.m)@state2))
        O2[i] = O2[i] - (O2[0]-np.asscalar(state1@sinphi(const.m)@state2))
    return np.stack((O1,O2), axis=1)
        #print(O1[i],O2[i])

fsolver = s.PathToField(init(), dt=dt)
fsolver.solve()
time, fields, states = fsolver.export()

plt.plot(time,fields[:,0])
plt.plot(time,fields[:,1])

#print(len(t))
#print(len(O1))
#plt.plot(t[:],O1[:])
#plt.plot(t[:],O2[:])
plt.show()
