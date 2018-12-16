'''solver.py
'''

import numpy as np
import math
import functions as f
import constants as const
from state import State
import operators as op


class PathToField(object):
    def __init__(self, path_desired):
        self.path = path_desired
        self.path_predicted = np.zeros_like(path_desired)
        self.n = path_desired.shape[0]
        self.tf = (2*math.pi*const.hbar/const.B)*100/(2*math.pi)
        self.t, self.dt = np.linspace(0, self.tf,
                          num=self.n, dtype=float, retstep=True)
        
        self.states = [None] * self.n
        state = State(m=const.m)
        state.value[const.m] = 1.0
        self.states[0] = state

        self._ddpath = np.column_stack((f.d2dt2(self.path[:,0], self.dt),
                                       f.d2dt2(self.path[:,1], self.dt)))
        self._op1 = (f.cosphi(const.m)
                     + 4*f.sinphi(const.m)@f.ddphi(const.m)
                     - 4*f.cosphi(const.m)@f.d2dphi2(const.m))
        self._op2 = (f.sinphi(const.m)
                     - 4*f.cosphi(const.m)@f.ddphi(const.m)
                     - 4*f.sinphi(const.m)@f.d2dphi2(const.m))
        self._cosphi2 = f.cosphi(const.m) @ f.cosphi(const.m)
        self._sinphi2 = f.sinphi(const.m) @ f.sinphi(const.m)
        self._cosphi_sinphi = f.cosphi(const.m) @ f.sinphi(const.m)
        self._sinphi_cosphi = f.sinphi(const.m) @ f.cosphi(const.m)

        self.field = np.zeros((self.n,2))
        self.field[0,:] = (self._get_Ainv(state) @ self._get_b(0, state)).reshape((2,)).real
    
    def calc_state_and_field(self):
        state = self.states[0]
        field = self.field[0,:]

        for j in range(1,self.n):
            state = op.RotorH(field).evolve(self.states[j-1], self.dt)
            field = self._get_Ainv(state) @ self._get_b(j, state)
            self.states[j] = state
            self.field[j,:] = field.real.reshape((2,))

        field_const = 5.142 * 10**11 * 10**(-10)
        self.field = field_const * self.field

        self.path_predicted = f.states_to_path(self.states)
        self._velidate()


    def __call__(self):
        return self.path_obtained

    def _velidate(self):
        pass
        
    def _get_det(self, state):
        c = 4*const.B**2*const.mu**2/const.hbar**4
        det = (c * (state.get_expt(self._sinphi2)
                   * state.get_expt(self._cosphi2)
                   - state.get_expt(self._sinphi_cosphi)**2 ))
        return det

    def _get_Ainv(self, state):
        c = 2*const.B*const.mu/const.hbar**2
        a11 = c * state.get_expt(self._sinphi2)
        a12 = -c * state.get_expt(self._cosphi_sinphi)
        a21 = -c * state.get_expt(self._sinphi_cosphi)
        a22 = c * state.get_expt(self._cosphi2)
        det = self._get_det(state)
        A_inv = 1/det * np.array([[a22, -a12],
                                  [-a21, a11]])
        return A_inv

    def _get_b(self, i, state):
        c = const.B**2/const.hbar**2
        b1 = self._ddpath[i,0] + np.real(c*state.get_expt(self._op1))
        b2 = self._ddpath[i,1] + np.real(c*state.get_expt(self._op2))
        return np.array([b1,b2]).reshape((2,1))


class FieldToPath(object):
    pass