import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    """Class for visualizing data of a DataStore object.

    """

    def __init__(dat):
        self.t = dat.t
        self.field = dat.field
        self.sin_phi_actual = dat.path_actual[:,1]
        self.cos_phi_actual = dat.path_actual[:,0]
        self.sin_phi_desired = dat.path_desired[:,1]
        self.cos_phi_desired = dat.path_desired[:,0]

    def trajectory(self):
        """Plot trajectories from field <cos(phi)> vs t, 
        <sin(phi) vs t, <sin(phi)> vs <cos(phi)>,
        and amplitude vs t

        """
        pass

    def field(self):
        """Plot the field over time

        """
        plt.figure(figsize=(8,8))
        plt.plot(self.t, self.field[:,0], color="blue", lw=2, label="x field")
        plt.plot(self.t, self.field[:,1], color="red", lw=2, label="y field")
        plt.xlabel("Time [ps]")
        plt.ylabel("Amplitude [V/A]")
        plt.title("Control field over time")
        plt.legend(loc="upper right")
        plt.show()

    def density(self):
        """Plot probability density over time for 
        different rotational angles

        """
        pass
