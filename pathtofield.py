'''path_to_field.py
'''

class PathToField(object):

    def __init__(self, path_desired):
        self.path_desired = path_desired
        self.field = np.zeros_like(path_desired)

    def calc_field():

        # initialize state
        self.state[const.m,t] = 1 # ground state initial condition

        # solve system of equations
        D,a1x,a2x,a1y,a2y = f.calcDandA(sinphi,cosphi,self.state[:,t])

        # initial step epsilon/field calculated
        self.field[t,:] = np.real(A_inv@[b1, b2])
        # convert field to appropriate units (V/angstrom)    
        field_const = 5.142*(10**11)*(10**-10)
        self.field = self.field*field_const

'''
        states = States()

        for i in range(1,n+1):
            H = RotorH(epsilon)
            my_state = H.evolve()
            states.set_list(i, my_state)
'''

    def calc_a
        # A^-1
        a1x = (2*const.B*const.mu/const.hbar**2)*state_conj@sinphi2@state
        a2x = (-2*const.B*const.mu/const.hbar**2)*state_conj@sinphi_cosphi@state
        a1y = (-2*const.B*const.mu/const.hbar**2)*state_conj@cosphi_sinphi@state
        a2y = (2*const.B*const.mu/const.hbar**2)*state_conj@cosphi2@state
        A_inv = (1/D)*np.array([[a2y, -a1y],[-a2x, a1x]])

    def calc_d
        D_consts = (4*const.B**2*const.mu**2/const.hbar**4)
        D_term1 = state_conj@sinphi2@state
        D_term2 = state_conj@cosphi2@state
        D_term3 = state_conj@sinphi_cosphi@state
        D_together = D_consts*(D_term1*D_term2-D_term3**2)
        D = D_consts*(D_term1*D_term2-D_term3**2)

    def calc_b
            def d2dt2(path,dt): # pass in 'path' vector, 2 components x and y, at a single time point
            """Calculate second derivative of path using centered finite differences, used for solving b-vector in pathtofield"""
            # initial step (finite differences method)
            n=len(path)
            d2x = np.zeros(n,dtype=complex)
            d2y = np.zeros(n,dtype=complex)
            d2x[0] = (path[2,0]-2*path[1,0]+path[0,0])/(dt**2)
            d2y[0] = (path[2,1]-2*path[1,1]+path[0,1])/(dt**2)
            for i in range(1,n-1):
                d2x[i] = ((path[i+1,0]-path[i,0])-(path[i,0]-path[i-1,0]))/(dt**2)
                d2y[i] = ((path[i+1,1]-path[i,1])-(path[i,1]-path[i-1,1]))/(dt**2)
            d2x[n-1] = (path[n-3,0]-2*path[n-2,0]+path[n-1,0])/(dt**2)
            d2y[n-1] = (path[n-3,1]-2*path[n-2,1]+path[n-1,1])/(dt**2)
            return d2x,d2y
        b1 = d2x[i]+np.real((const.B**2/const.hbar**2)*(state_conj@(cosphi+4*sinphi@ddphi-4*cosphi@d2dphi2)@self.state[:,i]))
        b2 = d2y[i]+np.real((const.B**2/const.hbar**2)*(state_conj@(sinphi-4*cosphi@ddphi-4*sinphi@d2dphi2)@self.state[:,i]))

    def get_field(self):
        return self.field


        ''' field_to_path 
'''
