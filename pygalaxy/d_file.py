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
        self.history = np.empty(
            shape=(self.number_iterations, *self.simu.particles.shape))
        self.history[0] = self.simu.particles

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
            self.history[i] = self._update_coords(i)

        print("save history...")
        # save all history into a file
        np.save("particles.npy", self.history)
        np.save("mass.npy", self.simu.mass)

        # Optionally validate the format
        self._validate_data()

    def _validate_data(self):
        """ Validate the structure of the particles data. """
        if self.history.shape[-1] < 2:
            raise ValueError("Particles data must include at least position and velocity.")

        if len(self.history.shape) != 4:
            raise ValueError("Particles data should follow the format [iteration, nb_particles, nb_dimension, x].")

        print("Validation complete. Data structure is correct.")
