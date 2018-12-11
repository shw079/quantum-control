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
        pass

    def density(self):
        """Plot probability density over time for 
        different rotational angles

        """
        pass
