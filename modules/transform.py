import numpy as np
import scipy
from scipy.signal import savgol_filter as savitzky_golay
import math
import constants as const

def transform_path(path):
    
    TOL = 0.5;
    if (np.amax(math.abs(path[:,0]))>TOL or np.amax(math.abs(path[:,1]))>TOL):
        path[:,0] = path[:,0]*(TOL/np.amax(math.abs(path[:,0])))
        path[:,1] = path[:,1]*(TOL/np.amax(math.abs(path[:,1])))

    lengthxy=0
    for k in range(1,len(path)):
        lengthxy = math.sqrt((path[k,0]-path[k-1,0])**2+(path[k,1]-path[k-1,1])**2)+lengthxy

    dt = 1000
    Trot = 2*math.pi*const.hbar/const.B
    max_t = 10*lengthxy*Trot/(2*math.pi)
    t = np.linspace(0, max_t ,max_t/dt+1)

    t_sigmoid = np.linspace(-5,0,len(t))
    sigmoid = np.exp(t_sigmoid) @ (1/(np.exp(t_sigmoid)[:]+1))
    sigmoid = sigmoid*(1/(sigmoid[-1]-sigmoid[0]))
    sigmoid = t[-1]*(sigmoid[:]+(1-sigmoid[-1]));

    t_raw = np.linspace(0,t[-1],t[-1]*(1/(len(path))))
    Ox = np.interp(sigmoid, t_raw, path[:,0])
    Oy = np.interp(sigmoid, t_raw, path[:,1])

    Ox = savitzky_golay(Ox, 4*math.floor(len(Ox)/len(Oxraw))+1, 5)
    Oy = savitzky_golay(Oy, 4*math.floor(len(Ox)/len(Oxraw))+1, 5)

    new_path = np.stack((Ox Oy))

    return new_path, dt
