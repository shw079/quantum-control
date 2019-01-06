'''solvers.py

This module contains two solver classes to calculate (1) control
fields for a given path (class PathToField), and (2) resulting path
from a given field (class PathToField). 

'''

import numpy as np
import math
from . import functions as f
from . import constants as const
from .state import State
from .field import Field
from . import observable as obs
from .molecule import Rotor

class PathToField(object):
    """Solve control fields for a given path of dipole moment projection.

    """

    def __init__(self, path_desired, dt=1000, molecule=None):
        """Instantiate a PathToField object for solving a specified path.

        Parameters:

            path_desired: A desired path of molecule dipole moment 
            projection. Guaranteed to be a n-by-2 numpy.ndarray where `n`
            is the number of time points?

            dt: (Optional) Difference of time between two adjacent time 
            points. This is path-specific but currently needs to be 
            provided by user directly. Guaranteed to be a scalar?

            molecule: (Optional) System of interest. A molecule.Rotor 
            object with its quantum number `m` specified in constants.py
            will be used if this argument is not provided by the user.

        """
        # Create a Rotor object as the system of interest if not 
        # provided by the user
        if molecule is None:
            ## System of interest.
            ## Default value is a rotor (solver.molecule.Rotor)
            self.molecule = Rotor(const.m)
        else:
            self.molecule = rotor

        ## Path specified
        self.path = path_desired
        self._path_predicted = np.zeros_like(path_desired)
        ## Number of time points
        self.n = path_desired.shape[0]
        ## Delta t between two adjacent time points.
        self.dt = dt
        self._t_final = self.n * self.dt
        ## Time vector in unit of ?
        self.time = np.arange(self._t_final, step=self.dt, dtype=float)
        self._ddpath = np.stack((f.d2dt2(self.path[:,0], self.dt),
                                 f.d2dt2(self.path[:,1], self.dt)), axis=1)

        # operators used only by private methods within class instance
        m = self.molecule.m
        self._op1 = (f.cosphi(m)
                     + 4*f.sinphi(m)@f.ddphi(m)
                     - 4*f.cosphi(m)@f.d2dphi2(m))
        self._op2 = (f.sinphi(m)
                     - 4*f.cosphi(m)@f.ddphi(m)
                     - 4*f.sinphi(m)@f.d2dphi2(m))
        self._cosphi2 = f.cosphi(m) @ f.cosphi(m)
        self._sinphi2 = f.sinphi(m) @ f.sinphi(m)
        self._cosphi_sinphi = f.cosphi(m) @ f.sinphi(m)
        self._sinphi_cosphi = f.sinphi(m) @ f.cosphi(m)

        #calc and set initial field, but not using molecule.update_field()
        field = self._get_field(0, real=True)
        self.molecule.set_field(field)

    def solve(self):
        """Calculate the control field required for each time step.
        """

        for j in range(1,self.n):
            self.molecule.evolve(self.dt)
            field = self._get_field(j)
            self.molecule.update_field(field)

        # self._velidate()

    def export(self):
        """Export calculated time vector, fields, and states as np.ndarray.

        Return values:

            time: Time vector based on dt. In unit of picoseconds.
            Numpy.ndarray of shape (n,)

            fields: Control fields required for the path of interest.
            In unit of V/angstrom. Numpy.ndarray of shape (n,2)

            states: States of the system at every time point.
            Numpy.ndarray of shape (2m+1,n)
        """

        time = self.molecule.get_time_asarray()
        time = time * 2.418 * 10**(-17) * 10**12 #time in picoseconds

        states = self.molecule.get_states_asarray()

        fields = self.molecule.get_fields_asarray()
        field_const = 5.142 * 10**11 * 10**(-10) #amplitude in V/angstrom
        fields = fields * field_const

        path = np.zeros((self.n,2))
        oper_x = obs.DipoleX(self.molecule.m).operator
        oper_y = obs.DipoleY(self.molecule.m).operator
        states_list = self.molecule.history['state']
        for i in range(self.n):
            path[i,0] = states_list[i].get_expt(oper_x).real
            path[i,1] = states_list[i].get_expt(oper_y).real

        return time, fields, states
    
    def _get_field(self, j, real=False):
        """Calculate the required field for the next step.

        Parameters:

            j: System is at the j-th time point.

            real: (optional) If True, force the returned value to be only 
            the real part of a complex number.

        Return value:

            The Field object (solver.field.Field) for the next step.
        """

        value = self._get_Ainv() @ self._get_b(j)
        if real:
            value = value.real
        return Field(value)

    def _get_det(self):
        """Calculate determinant of matrix A"""
        state = self.molecule.state
        c = 4*const.B**2*const.mu**2/const.hbar**4
        det = (c * (state.get_expt(self._sinphi2)
                   * state.get_expt(self._cosphi2)
                   - state.get_expt(self._sinphi_cosphi)**2 ))
        return det

    def _get_Ainv(self):
        """Calculate inverse of matrix A"""
        state = self.molecule.state
        c = 2*const.B*const.mu/const.hbar**2
        a11 = c * state.get_expt(self._sinphi2)
        a12 = -c * state.get_expt(self._cosphi_sinphi)
        a21 = -c * state.get_expt(self._sinphi_cosphi)
        a22 = c * state.get_expt(self._cosphi2)
        det = self._get_det()
        A_inv = 1/det * np.array([[a22, -a12],
                                  [-a21, a11]])
        return A_inv

    def _get_b(self, i):
        """Calculate b vector"""
        state = self.molecule.state
        c = const.B**2/const.hbar**2
        b1 = self._ddpath[i,0] + np.real(c*state.get_expt(self._op1))
        b2 = self._ddpath[i,1] + np.real(c*state.get_expt(self._op2))
        return np.array([b1,b2])


class FieldToPath(object):
    """Calculate the resulting path from a given set of control fields"""

    def __init__(self, fields, dt=1000, molecule=None):
        """Instantiate a FieldToPath object for solving path for given fields.

        Parameters:

            fields: Control fields of interest. Guaranteed to be a n-by-2
            numpy.ndarray where `n` is the number of time points?

            dt: (Optional) Difference of time between two adjacent time 
            points. This is path-specific but currently needs to be 
            provided by user directly. Guaranteed to be a scalar?

            molecule: (Optional) System of interest. A molecule.Rotor 
            object with its quantum number `m` specified in constants.py
            will be used if this argument is not provided by the user.

        """

        # Create a Rotor object as the system of interest if not 
        # provided by the user
        if molecule is None:
            ## System of interest
            ## Default: a Rotor object with quantum number = const.m
            self.molecule = Rotor(const.m)
        else:
            self.molecule = rotor
        ## Number of time points
        self.n = fields.shape[0]
        ## An nx2 np.ndarray containing the given field.
        ## Each row is the x- and y-component of such field at a time point.
        self.fields = fields
        self._fields_list = [Field(fields[i,:]) for i in range(self.n)]
        ## The resulting path from the given fields.
        self.path = np.zeros((self.n,2))
        ## Time difference between two adjacent time points.
        self.dt = dt
        self._t_final = self.dt * self.n
        ## Time vector containing all time points.
        self.time = np.arange(self._t_final, step=self.dt, dtype=float)
        self.time_in_ps = self.time * 2.418e-5 #time in picoseconds

        #get and set initial field, but not using molecule.update_field()
        field = self._fields_list[0]
        self.molecule.set_field(field)

    def solve(self):
        """Calculate path of rotor dipole moment projection from given fields.
        """

        for i in range(1,self.n):
            self.molecule.evolve(self.dt)

    def export(self):
        """Export calculated time vector, fields, and states as np.ndarray.

        Return values:

            time: Time vector based on dt. In unit of picoseconds.
            Numpy.ndarray of shape (n,)

            path: Resulting path from the given fields.
            Numpy.ndarray of shape (n,2)

            states: States of the system at every time point.
            Numpy.ndarray of shape (2m+1,n)
        """

        time = self.molecule.get_time_asarray()
        # time = self.time
        states = self.molecule.get_states_asarray()
        states_list = self.molecule.history['state']
        path = np.zeros((self.n,2))
        oper_x = obs.DipoleX(self.molecule.m).operator
        oper_y = obs.DipoleY(self.molecule.m).operator
        for i in range(self.n):
            path[i,0] = states_list[i].get_expt(oper_x).real
            path[i,1] = states_list[i].get_expt(oper_y).real

        return time, path, states















