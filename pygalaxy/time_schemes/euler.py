import numpy as np

class Euler:
    def __init__(self, dt, nbodies, method):
        self.dt = dt
        self.method = method
        self.k1 = np.zeros((nbodies, 4))

    def init(self, mass, particles):
        pass

    def update(self, mass, particles):
        self.method(mass, particles, self.k1)
        particles[:, :] += self.dt*self.k1

class Euler_symplectic:
    def __init__(self, dt, nbodies, method):
        self.dt = dt
        self.method = method
        self.k1 = np.zeros((nbodies, 4))

    def init(self, mass, particles):
        pass

    def update(self, mass, particles):
        self.method(mass, particles, self.k1)
        particles[:, :2] += self.dt*self.k1[:, :2]
        self.method(mass, particles, self.k1)
        particles[:, 2:] += self.dt*self.k1[:, 2:]

class Euler_symplectic_tree:
    def __init__(self, dt, nbodies, method):
        """
        Initialisation de la méthode symplectique avec arbre Barnes-Hut.

        Args:
            dt: Pas de temps (float).
            nbodies: Nombre de particules (int).
            method: Méthode pour calculer les forces et l'énergie, ici `compute_energy_and_tree_structure`.
        """
        self.dt = dt
        self.method = method  # Méthode utilisée pour calculer les forces et l'énergie
        self.k1 = np.zeros((nbodies, 4))  # Tableau pour stocker les forces et vitesses intermédiaires
        self.tree = None  # Structure de l'arbre Barnes-Hut

    def init(self, mass, particles):
        """
        Initialisation des données. Ici, aucune action spécifique n'est requise.
        """
        pass

    def update(self, mass, particles):
        """
        Met à jour les positions et vitesses des particules en utilisant un schéma d'Euler symplectique.

        Args:
            mass: Tableau des masses des particules.
            particles: Tableau des positions et vitesses des particules.
        """
        # Calcul des forces et extraction de la structure de l'arbre
        self.tree = self.method(mass, particles, self.k1)

        # Mise à jour des positions (utilisant les vitesses)
        particles[:, :2] += self.dt * self.k1[:, :2]

        # Recalcul des forces avec les nouvelles positions
        self.method(mass, particles, self.k1)

        # Mise à jour des vitesses (utilisant les accélérations)
        particles[:, 2:] += self.dt * self.k1[:, 2:]
