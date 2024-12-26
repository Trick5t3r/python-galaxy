import numpy as np

class RK4:
    def __init__(self, dt, nbodies, dim, method):
        """
        Initialise le solveur RK4 pour un espace à d dimensions.

        Args:
            dt: Pas de temps.
            nbodies: Nombre de particules.
            dim: Nombre de dimensions de l'espace.
            method: Méthode pour calculer les forces.
        """
        self.dt = dt
        self.method = method
        self.dim = dim
        self.k1 = np.zeros((nbodies, dim, 2))  # Positions et vitesses dans chaque dimension
        self.k2 = np.zeros((nbodies, dim, 2))
        self.k3 = np.zeros((nbodies, dim, 2))
        self.k4 = np.zeros((nbodies, dim, 2))
        self.tmp = np.zeros((nbodies, dim, 2))

    def init(self, mass, particles):
        """
        Initialisation optionnelle pour le solveur RK4.

        Args:
            mass: Masses des particules.
            particles: Tableau des particules ([nb_particules, nb_dimension, x]).
        """
        pass

    def update(self, mass, particles):
        """
        Met à jour les particules en utilisant la méthode RK4.

        Args:
            mass: Masses des particules.
            particles: Tableau des particules ([nb_particules, nb_dimension, x]).
        """
        # k1
        self.method(mass, particles, self.k1)
        self.tmp[:, :, :] = particles[:, :, :2] + self.dt * 0.5 * self.k1

        # k2
        self.method(mass, self.tmp, self.k2)
        self.tmp[:, :, :] = particles[:, :, :2] + self.dt * 0.5 * self.k2

        # k3
        self.method(mass, self.tmp, self.k3)
        self.tmp[:, :, :] = particles[:, :, :2] + self.dt * self.k3

        # k4
        self.method(mass, self.tmp, self.k4)

        # Mise à jour des particules
        particles[:, :, :2] += self.dt / 6.0 * (
            self.k1 + 2.0 * (self.k2 + self.k3) + self.k4
        )
