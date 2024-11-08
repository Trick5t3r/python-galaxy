#!/usr/bin/env python

import numpy as np


class Animation(object):
    """ Simulation renderer in output file. """

    def __init__(self, simu, axis=[0, 1, 0, 1]):
        """ Initialize an animation view.

        Parameters:
        -----------
        simu: object
            Simulation object with coords and next methods
        axis: list
            Axis bounds [ xmin, xmax, ymin, ymax ].
        """

        self.simu = simu

        self.number_iterations = 50
        self.data = np.empty(
            shape=(self.number_iterations, *self.simu.particles.shape))
        self.data[0] = self.simu.particles

    def _update_coords(self, i):
        """ Update scatter coordinates. """
        self.simu.next()

        # We need to return an iterable since FuncAnimation expects a returned
        # object of this nature
        return self.simu.particles

    def main_loop(self):
        """ main loop. """

        for i in range(self.number_iterations):
            print(f"{i}/{self.number_iterations}", end="\r")
            self.data[i] = self._update_coords(i)

        print("save history...")
        # save all history into a file
        np.save("output.npy", self.data)
