#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class Animation(object):
    """ Simulation renderer using matplotlib. """

    def __init__(self, simu, axis=[0, 1, 0, 1], dim=2):
        """ Initialize an animation view.

        Parameters:
        -----------
        simu: object
            Simulation object with coords and next methods
        axis: list
            Axis bounds [xmin, xmax, ymin, ymax, zmin, zmax].
        dim: int
            Dimension of the plot (2 or 3).
        """

        self.simu = simu
        self.dim = dim

        self.fig = plt.figure(figsize=(10, 10))
        
        if self.dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_facecolor('black')
            self.ax.set_xlim(axis[0], axis[1])
            self.ax.set_ylim(axis[2], axis[3])
            self.ax.set_zlim(axis[4], axis[5])

            coords = simu.coords()
            self.scatter = self.ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='white', s=.5)
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor('black')
            self.ax.axis(axis[:4])

            coords = simu.coords()
            self.scatter = self.ax.scatter(coords[:, 0], coords[:, 1], c='white', s=.5)

    def _update_coords(self, i):
        """ Update scatter coordinates. """
        self.simu.next()
        coords = self.simu.coords()

        if self.dim == 3:
            self.scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
        else:
            self.scatter.set_offsets(coords[:, :2])

        # We need to return an iterable since FuncAnimation expects a returned
        # object of this nature
        return self.scatter,

    def main_loop(self):
        """ Animation main loop. """
        # We need to keep the animation object around otherwise it is garbage
        # collected. So we use the dummy `_`
        _ = animation.FuncAnimation(self.fig, self._update_coords, blit=True)
        plt.show()
