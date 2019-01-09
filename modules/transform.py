''' transform.py
'''

import numpy as np
import scipy
from scipy.signal import savgol_filter as savitzky_golay
import math
import constants as const

def transform_path(path):
    """!@brief Processes user defined input path by interpolating (in a manner that 
    creates a higher density of points at early times) and 
       smoothing using a savitzky-golay filter for compatibility with solver
       """


    ## Confine path to remain in circle with radius TOL (physical limit is 1)
    TOL = 0.5; 
    if (np.amax(abs(path[:,0]))>TOL or np.amax(abs(path[:,1]))>TOL):
        x = path[:,0]*(TOL/np.amax(abs(path[:,0])))
        y = path[:,1]*(TOL/np.amax(abs(path[:,1])))
        path = np.column_stack((x,y))

    ## Calculate approximate length of path
    lengthxy=0
    for k in range(1,len(path)):
        lengthxy = math.sqrt((path[k,0]-path[k-1,0])**2+(path[k,1]-path[k-1,1])**2)+lengthxy

    ## Create new time array based on length of path
    dt = 1000
    Trot = 2*math.pi*const.hbar/const.B
    max_t = 10*lengthxy*Trot/(2*math.pi)
    t = np.arange(0,max_t,dt)
    max_t = t[-1]

    ## Use sigmoid function to create higher density of points at early times
    t_sigmoid = np.linspace(-5,0,len(t))
    sigmoid = np.exp(t_sigmoid)[:]/(np.exp(t_sigmoid)[:]+1)
    sigmoid = sigmoid*(1/(sigmoid[-1]-sigmoid[0]))
    sigmoid = t[-1]*(sigmoid[:]+(1-sigmoid[-1]));

    ## Interpolate paths using sigmoid
    t_raw = np.linspace(0,1,len(path))
    t_raw = t_raw*max_t
    Ox = np.interp(sigmoid, t_raw, path[:,0])
    Oy = np.interp(sigmoid, t_raw, path[:,1])

    ## Smooth interpolated paths using savitzky-golay filter
    Ox = savitzky_golay(Ox, 8*math.floor(len(Ox)/len(path[:,0]))+1, 5)
    Oy = savitzky_golay(Oy, 8*math.floor(len(Ox)/len(path[:,0]))+1, 5)

    new_path = np.stack((Ox,Oy)).T

    return new_path, dt
