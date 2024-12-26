import numpy as np

class Euler:
    def __init__(self, dt, nbodies, dim, method):
        """
        Initialise la méthode d'Euler pour un espace à d dimensions.

        Args:
            dt: Pas de temps.
            nbodies: Nombre de particules.
            dim: Nombre de dimensions de l'espace.
            method: Méthode pour calculer les forces.
        """
        self.dt = dt
        self.dim = dim
        self.method = method
        self.k1 = np.zeros((nbodies, dim, 2))

    def init(self, mass, particles):
        pass

    def update(self, mass, particles):
        """
        Met à jour les particules en utilisant la méthode d'Euler.

        Args:
            mass: Masses des particules.
            particles: Tableau des particules ([nb_particules, nb_dimension, x]).
        """
        self.method(mass, particles, self.k1, dim=self.dim)
        particles[:, :, :2] += self.dt * self.k1

class Euler_symplectic:
    def __init__(self, dt, nbodies, dim, method):
        """
        Initialise la méthode symplectique d'Euler pour un espace à d dimensions.

        Args:
            dt: Pas de temps.
            nbodies: Nombre de particules.
            dim: Nombre de dimensions de l'espace.
            method: Méthode pour calculer les forces.
        """
        self.dt = dt
        self.dim = dim
        self.method = method
        self.k1 = np.zeros((nbodies, dim, 2))

    def init(self, mass, particles):
        pass

    def update(self, mass, particles):
        """
        Met à jour les particules en utilisant la méthode symplectique d'Euler.

        Args:
            mass: Masses des particules.
            particles: Tableau des particules ([nb_particules, nb_dimension, x]).
        """
        self.method(mass, particles, self.k1, dim=self.dim)
        particles[:, :, 0] += self.dt * self.k1[:, :, 0]
        self.method(mass, particles, self.k1, dim=self.dim)
        particles[:, :, 1] += self.dt * self.k1[:, :, 1]

class Euler_symplectic_tree:
    def __init__(self, dt, nbodies, dim, method):
        """
        Initialise la méthode symplectique d'Euler avec arbre Barnes-Hut pour un espace à d dimensions.

        Args:
            dt: Pas de temps.
            nbodies: Nombre de particules.
            dim: Nombre de dimensions de l'espace.
            method: Méthode pour calculer les forces et l'énergie, comme `compute_energy_and_tree_structure`.
        """
        self.dt = dt
        self.dim = dim
        self.method = method
        self.k1 = np.zeros((nbodies, dim, 2))
        self.tree = None

    def init(self, mass, particles):
        pass

    def update(self, mass, particles):
        """
        Met à jour les particules en utilisant la méthode symplectique d'Euler avec arbre Barnes-Hut.

        Args:
            mass: Masses des particules.
            particles: Tableau des particules ([nb_particules, nb_dimension, x]).
        """
        # Calcul des forces et extraction de la structure de l'arbre
        self.tree = self.method(mass, particles, self.k1, dim=self.dim)

        # Mise à jour des positions (utilisant les vitesses)
        particles[:, :, 0] += self.dt * self.k1[:, :, 0]

        # Recalcul des forces avec les nouvelles positions
        self.method(mass, particles, self.k1, dim=self.dim)

        # Mise à jour des vitesses (utilisant les accélérations)
        particles[:, :, 1] += self.dt * self.k1[:, :, 1]
