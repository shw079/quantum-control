import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

class Visualization:
    """Class for visualizing data of a DataStore object.

    """

    def __init__(self, dat):
        self.state = dat.state
        self.m = dat.Const.m
        self.t = dat.t
        self.field = dat.field
        self.sin_phi_actual = dat.path_actual[:, 1]
        self.cos_phi_actual = dat.path_actual[:, 0]
        self.sin_phi_desired = dat.path_desired[:, 1]
        self.cos_phi_desired = dat.path_desired[:, 0]

    def density(self, n_grid=100, out=None):
        """Plot probability density over time for
        different rotational angles

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
        # prob_proj of shape (len(phi), len(t))
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
        """Plot trajectories from field <cos(phi)> vs t, 
        and <sin(phi) vs t, <sin(phi)> vs <cos(phi)>

        """
        # set up grids
        fig = plt.figure(figsize=(16.5, 8))
        grid = plt.GridSpec(2, 4, wspace=0.5)

        # trajectories over time
        # x track
        ax1 = fig.add_subplot(grid[0, :2])

        ax1.plot(self.t, self.cos_phi_actual, color="blue", lw=2,
                 alpha=0.6, label="actual x track")
        ax1.plot(self.t, self.cos_phi_actual, color="blue", lw=2,
                 ls="--", label="expected x track")

        ax1.set_ylabel(u"<cos(${\phi}$)>", fontsize=14)
        ax1.set_ylim(-1,1)
        ax1.legend(loc="upper left", fontsize=12)

        # y track
        ax2 = fig.add_subplot(grid[1, :2])

        ax2.plot(self.t, self.sin_phi_actual, color="red", lw=2,
                 alpha=0.6, label="actual y track")
        ax2.plot(self.t, self.sin_phi_actual, color="red", lw=2,
                 ls="--", label="expected y track")

        ax2.set_ylabel(u"<sin(${\phi}$)>", fontsize=14)
        ax2.set_ylim(-1,1)
        ax2.set_xlabel("Time [ps]", fontsize=14)
        ax2.legend(loc="upper left", fontsize=12)

        # phase plot
        ax3 = fig.add_subplot(grid[:, 2:])

        ax3.plot(self.cos_phi_actual, self.sin_phi_actual,
                 color="black", lw=2, label="phase plot")

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

    def field(self):
        """Plot the field over time

        """
        plt.figure(figsize=(8, 8))

        # plot the real part only
        plt.plot(self.t, np.real(self.field[:, 0]), color="blue", 
                 lw=2, label="x field")
        plt.plot(self.t, np.real(self.field[:, 1]), color="red", 
                 lw=2, label="y field")

        plt.xlabel("Time [ps]", fontsize=14)
        plt.ylabel("Amplitude [V/A]", fontsize=14)
        plt.title("Control field over time", fontsize=20)
        plt.legend(loc="upper right", fontsize=12)

        # save or display figure
        if out:
            plt.savefig(out)
        else:
            plt.show()

