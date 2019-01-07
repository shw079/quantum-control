"""!@package docstring
The visualization module uses data in a DataStore object for plotting.
This module allows plotting of:
    1. control fields amplitude over time for both x and y fields.
    2. actual and expected trajectory of the rigid rotor for dimensions
       x and y over time, together with a phase plot of x and y tracks.
    3. a heatmap of probability density for rotational angle over time.
"""


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np


class Visualization:
    """Class for visualizing data of a DataStore object.

    """
    def __init__(self, dat):
        """!
        Construct a Visualization object.
        
        @param dat the DataContainer object used for plotting.
        """
        ## time points
        self.t = t
        ## system state at each time point
        self.state = dat.state
        ## magnetic quantum number
        self.m = dat.Const.m
        ## control fields
        self.field = dat.field
        ## actual x track
        self.cos_phi_actual = dat.path_actual[:, 0]
        ## actual y track 
        self.sin_phi_actual = dat.path_actual[:, 1]
        ## desired x track
        self.cos_phi_desired = dat.path_desired[:, 0]
        ## desired y track
        self.sin_phi_desired = dat.path_desired[:, 1]

    def density(self, n_grid=100, out=None):
        """!
        Plot a probability density heatmap over time for different 
        rotational angles. 

        @param n_grid number of equally spaced rotational angles to 
                      calculate probability density between 0 and 
                      2&pi;. Default to 100.

        @param out the path to save the figure if not None. If None,
                   show the figure in the console. Default to None.

        @return a numpy ndarray for probability density for plotting, 
                with shape of (n_grid, self.t).
        """
        # calculate probability density
        # get equally spaced points in [0, 2 * pi)
        phi = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)

        # initialize arrays
        # energy to anuglar representation transformation
        # wave_trans of shape (2m+1, len(phi))
        wave_trans = np.empty((2 * self.m + 1, len(phi)),
                              dtype=np.complex64)

        for l in range(2 * self.m + 1):
            wave_trans[l, :] = 1 / np.sqrt(2 * np.pi) * \
                               np.exp(1j * (l - self.m - 1) * phi)

        # state of shape (2m+1, len(t))
        # proba of shape (len(phi), len(t))
        proba = np.flip(np.abs(np.dot(wave_trans.T, self.state)) ** 2,
                        axis = 0)

        # plot probability density
        plt.figure(figsize=(10, 10))

        # display matrix
        plt.imshow(proba, extent=[self.t.min(), self.t.max(), 
                   0, 2 * np.pi],
                   aspect = np.ptp(self.t) / (2 * np.pi),
                   cmap="jet", vmin=0, vmax=1,
                   interpolation="bilinear")

        plt.colorbar(fraction=0.0457, pad=0.04)
        plt.xlabel("Time [ps]", fontsize=14)
        plt.ylabel(u"Angle of rotation $\phi \in [0, 2\pi)$", fontsize=14)
        plt.title(u"Probability density $|<\phi|\psi(t)>|^2$", fontsize=20)

        # save or display figure
        if out:
            plt.savefig(out)
        else:
            plt.show()
        # return proba for unit testing
        return proba
    
    def trajectory(self, out=None):
        """!@brief Plot trajectories of the rigid rotor.

        Plot actual and expected trajectories of x (i.e. <cos(&phi;)>) 
        and y (i.e. <sin(&phi;)>) over time and also the phase plot of
        x and y.

        @param out the path to save the figure if not None. If None,
                   show the figure in the console. Default to None.
        """
        # set up grids
        fig = plt.figure(figsize=(20.5, 10))
        grid = plt.GridSpec(2, 4, wspace=0.5)

        # trajectories over time
        # x track
        ax1 = fig.add_subplot(grid[0, :2])

        ax1.plot(self.t, self.cos_phi_actual, color="blue", lw=2,
                 alpha=0.6, label="actual x track")
        ax1.plot(self.t, self.cos_phi_desired, color="blue", lw=2,
                 ls="--", label="expected x track")

        ax1.set_ylabel(u"<cos(${\phi}$)>", fontsize=14)
        ax1.set_ylim(-1,1)
        ax1.legend(loc="upper left", fontsize=12)

        # y track
        ax2 = fig.add_subplot(grid[1, :2])

        ax2.plot(self.t, self.sin_phi_actual, color="red", lw=2,
                 alpha=0.6, label="actual y track")
        ax2.plot(self.t, self.sin_phi_desired, color="red", lw=2,
                 ls="--", label="expected y track")

        ax2.set_ylabel(u"<sin(${\phi}$)>", fontsize=14)
        ax2.set_ylim(-1,1)
        ax2.set_xlabel("Time [ps]", fontsize=14)
        ax2.legend(loc="upper left", fontsize=12)

        # phase plot
        ax3 = fig.add_subplot(grid[:, 2:])

        ax3.plot(self.cos_phi_actual, self.sin_phi_actual,
                 color="black", lw=2, alpha=0.6, 
                 label="actual phase plot")
        ax3.plot(self.cos_phi_desired, self.sin_phi_desired,
                 color="black", lw=2, ls="--", 
                 label="expected phase plot")

        ax3.set_ylabel(u"<sin(${\phi}$)>", fontsize=14)
        ax3.set_xlabel(u"<cos(${\phi}$)>", fontsize=14)
        ax3.set_xlim(-1,1)
        ax3.set_ylim(-1,1)
        ax3.legend(loc="upper right", fontsize=12)

        fig.suptitle("Trajectory of the rigid rotor", fontsize=20)

        # save or display figure
        if out:
            plt.savefig(out)
        else:
            plt.show()

    def field(self, out=None):
        """!
        Plot contol fields amplitude over time for both x and y 
        dimensions.

        @param out the path to save the figure if not None. If None,
                   show the figure in the console. Default to None.
        """
        plt.figure(figsize=(10, 10))

        # plot the real part only
        plt.plot(self.t, np.real(self.field[:, 0]), color="blue", 
                 lw=2, label="x field")
        plt.plot(self.t, np.real(self.field[:, 1]), color="red", 
                 lw=2, label="y field")

        plt.xlabel("Time [ps]", fontsize=14)
        plt.ylabel("Amplitude [V/A]", fontsize=14)
        plt.title("Control fields over time", fontsize=20)
        plt.legend(loc="upper right", fontsize=12)

        # save or display figure
        if out:
            plt.savefig(out)
        else:
            plt.show()

