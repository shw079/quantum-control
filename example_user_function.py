"""@package docstring`

This module exists to hold an example analytic function that a user might provide

"""

import numpy as np

def user_function():

	t_list = np.linspace(np.pi/2.0,10*np.pi,1000)
	coord_list = [ [np.cos(t)*t, np.sin(t)*t] for t in t_list ]
	return coord_list

if __name__ == "__main__":
    np.savetxt('example_data.dat', user_function())

