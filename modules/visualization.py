import matplotlib.pyplot as plt
import numpy as np

import constants


class Visualization:
    """Class for visualizing data of a DataStore object.

    The visualization module uses data in a DataContainer object for 
    plotting. This module allows plotting of:

    - control fields amplitude over time for both x and y fields.
    - actual and expected trajectory of the rigid rotor for dimensions
      x and y over time, together with a phase plot of x and y 
      trajectories.
    - a heatmap of probability density for rotational angle over time.

    Parameters
    ----------
    dat : object
        A DataContainer used for plotting.


    Attributes
    ----------
    t : numpy.array, shape=(n,).
        Time points.

    m : int
        Magnetic quantum number.

    state : numpy.array, shape=((2m+1, n))
        System state at each time point.

    field : numpy.array, shape=(n, 2)
        Control fields.

    cos_phi_actual : numpy.array, shape=(n,)
        Actual x trajectory.

    sin_phi_actual : numpy.array, shape=(n,)
        Actual y trajectory.

    cos_phi_desired : numpy.array, shape=(n,)
        Desired x trajectory.

    sin_phi_desired : numpy.array, shape=(n,)
        Desired y trajectory.

    mean_cos_phi_actual_noise : numpy.array, shape=(n,)
        Mean actual x trajectory with noise.

    mean_sin_phi_actual_noise : numpy.array, shape=(n,)
        Mean actual y trajectory with noise.

    noise_stat_var : numpy.array, shape=(n, 2)
        Variance of actual trajectorys with noise.

    """
    def __init__(self, dat):
        self.t = dat.t
        self.state = dat.state
        self.m = constants.m
        self.field = dat.field
        self.cos_phi_actual = dat.path_actual[:, 0]
        self.sin_phi_actual = dat.path_actual[:, 1]
        self.cos_phi_desired = dat.path_desired[:, 0]
        self.sin_phi_desired = dat.path_desired[:, 1]
        self.mean_cos_phi_actual_noise = dat.noise_stat_mean[:, 0]
        self.mean_sin_phi_actual_noise = dat.noise_stat_mean[:, 1]
        self.noise_stat_var = dat.noise_stat_var

    def density(self, n_grid=100, out=None):
        """Plot a probability density heatmap over time for different 
        rotational angles. 

        Parameters
        ----------
        n_grid : int, optional (default=100)
            Number of equally spaced rotational angles to calculate 
            probability density between 0 and 2:math:`pi`. Default 
            to 100.

        out : str, optional (default=None)
            The path to save the figure if not None. If None, show 
            the figure in the console. Default to None.

        Returns
        -------
        numpy.array, shape=(n_grid, n)
            A numpy ndarray for probability density for plotting.

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
            plt.switch_backend('agg')
            plt.savefig(out)
        else:
            plt.show()
        # return proba for unit testing
        return proba
    
    def trajectory(self, noise=True, out=None):
        """Plot trajectories of the rigid rotor.

        Plot actual and expected trajectories of x (i.e. 
        <cos(:math:`phi`)>) and y (i.e. <sin(:math:`phi`)>) over time 
        and also the phase plot of x and y.

        Parameters
        ----------
        noise : bool, optional (default=True)
            Whether to plot mean trajectories under noise over time.
            Default to True.

        out : str
            The path to save the figure if not None. If None, show 
            the figure in the console. Default to None.

        """
        # set up grids
        fig = plt.figure(figsize=(20.5, 10))
        grid = plt.GridSpec(2, 4, wspace=0.5)

        # trajectories over time
        # x trajectory
        ax1 = fig.add_subplot(grid[0, :2])

        ax1.plot(self.t, self.cos_phi_actual, color="blue", lw=2,
                 alpha=0.6, label="actual x trajectory" if not noise else \
                                  "actual x trajectory, w/o noise")

        if noise:
            ax1.plot(self.t, self.mean_cos_phi_actual_noise,
                     color="blue", lw=2, alpha=0.6, ls=":",
                     label="mean actual x trajectory, w/ noise")

        ax1.plot(self.t, self.cos_phi_desired, color="blue", lw=2,
                 ls="-.", label="expected x trajectory")

        ax1.set_ylabel(u"<cos(${\phi}$)>", fontsize=14)
        ax1.set_ylim(-1,1)
        ax1.legend(loc="upper left", fontsize=12)

        # y trajectory
        ax2 = fig.add_subplot(grid[1, :2])

        ax2.plot(self.t, self.sin_phi_actual, color="red", lw=2,
                 alpha=0.6, label="actual y trajectory" if not noise else \
                                  "actual y trajectory, w/o noise")
 
        if noise:
            ax2.plot(self.t, self.mean_sin_phi_actual_noise,
                     color="red", lw=2, alpha=0.6, ls=":",
                     label="mean actual y trajectory, w/ noise")

        ax2.plot(self.t, self.sin_phi_desired, color="red", lw=2,
                 ls="-.", label="expected y trajectory")

        ax2.set_ylabel(u"<sin(${\phi}$)>", fontsize=14)
        ax2.set_ylim(-1,1)
        ax2.set_xlabel("Time [ps]", fontsize=14)
        ax2.legend(loc="upper left", fontsize=12)

        # phase plot
        ax3 = fig.add_subplot(grid[:, 2:])

        ax3.plot(self.cos_phi_actual, self.sin_phi_actual,
                 color="black", lw=2, alpha=0.6, 
                 label="actual phase plot" if not noise else \
                       "actual phase plot, w/o noise")

        if noise:
            ax3.plot(self.mean_cos_phi_actual_noise,
                     self.mean_sin_phi_actual_noise,
                     color="black", lw=2, alpha=0.6, ls=":",
                     label="mean actual phase plot, w/ noise")

        ax3.plot(self.cos_phi_desired, self.sin_phi_desired,
                 color="black", lw=2, ls="-.",
                 label="expected phase plot")

        ax3.set_ylabel(u"<sin(${\phi}$)>", fontsize=14)
        ax3.set_xlabel(u"<cos(${\phi}$)>", fontsize=14)
        ax3.set_xlim(-1,1)
        ax3.set_ylim(-1,1)
        ax3.legend(loc="upper right", fontsize=12)

        fig.suptitle("Trajectory of the rigid rotor", fontsize=20)

        # save or display figure
        if out:
            plt.switch_backend('agg')
            plt.savefig(out)
        else:
            plt.show()

    def fields(self, out=None):
        """Plot contol fields amplitude over time for both x and y 
        dimensions.

        Parameters
        ----------
        out : str
            The path to save the figure if not None. If None, show 
            the figure in the console. Default to None.

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
            plt.switch_backend('agg')
            plt.savefig(out)
        else:
            plt.show()

    def noise_variance(self, out=None):
        """Plot variance of trajectories over time under noise.

        Parameters
        ----------
        out : str
            The path to save the figure if not None. If None, show
            the figure in the console. Default to None.

        """
        plt.figure(figsize=(10, 10))

        plt.plot(self.t, self.noise_stat_var[:, 0], color="blue",
                 lw=2, label=u"<cos(${\phi}$)>")
        plt.plot(self.t, self.noise_stat_var[:, 1], color="red",
                 lw=2, label=u"<sin(${\phi}$)>")

        plt.xlabel("Time [ps]", fontsize=14)
        plt.ylabel("Variance", fontsize=14)
        plt.title("Variance of trajectory under noise", 
                  fontsize=20)
        plt.legend(loc="upper right", fontsize=12)

        # save or display figure
        if out:
            plt.switch_backend('agg')
            plt.savefig(out)
        else:
            plt.show()

