import numpy as np
from .d_rk4 import RK4

class ADB6:
    def __init__(self, dt, nbodies, dim, method):
        """
        Initialise l'intégrateur ADB6 pour un espace à d dimensions.

        Args:
            dt: Pas de temps.
            nbodies: Nombre de particules.
            dim: Nombre de dimensions de l'espace.
            method: Méthode pour calculer les forces.
        """
        self.dt = dt
        self.method = method
        self.dim = dim
        self.c = [4277.0 / 1440.0,
                 -7923.0 / 1440.0,
                  9982.0 / 1440.0,
                 -7298.0 / 1440.0,
                  2877.0 / 1440.0,
                  -475.0 / 1440.0]
        self.f = np.zeros((6, nbodies, dim, 2))  # Positions et vitesses dans chaque dimension

    def init(self, mass, particles):
        """
        Initialise les valeurs nécessaires pour l'intégrateur.

        Args:
            mass: Masses des particules.
            particles: Tableau des particules ([nb_particules, nb_dimension, x]).
        """
        rk4 = RK4(self.dt, particles.shape[0], self.dim, self.method)

        for i in range(5):
            rk4.update(mass, particles)
            self.f[i, :, :, :] = rk4.k1

        self.method(mass, particles, self.f[5])

    def update(self, mass, particles):
        """
        Met à jour les particules en utilisant l'intégrateur ADB6.

        Args:
            mass: Masses des particules.
            particles: Tableau des particules ([nb_particules, nb_dimension, x]).
        """
        particles[:, :, :] += self.dt * (
            self.c[0] * self.f[5, :, :, :] +
            self.c[1] * self.f[4, :, :, :] +
            self.c[2] * self.f[3, :, :, :] +
            self.c[3] * self.f[2, :, :, :] +
            self.c[4] * self.f[1, :, :, :] +
            self.c[5] * self.f[0, :, :, :]
        )
        self.f = np.roll(self.f, -1, axis=0)
        self.method(mass, particles, self.f[5], dim=self.dim)
