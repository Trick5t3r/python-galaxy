from six import print_

import numpy as np
from .dTree import TreeArray
import time

from . import d_numba_functions
import numba

@numba.njit(parallel=True)
def compute_force(nbodies, child, center_of_mass, mass, cell_radius, particles, energy, dim):
    for i in numba.prange(particles.shape[0]):
        acc = d_numba_functions.computeForce(nbodies, child, center_of_mass, mass, cell_radius, particles[i], dim)
        energy[i, :, 1] = acc

def compute_energy(mass, particles, energy, dim):
    bmin = np.min(particles[:, :, 0], axis=0)
    bmax = np.max(particles[:, :, 0], axis=0)
    root = TreeArray(bmin, bmax, particles.shape[0], dim)

    root.buildTree(particles)
    root.computeMassDistribution(particles, mass)
    compute_force(root.nbodies, root.child, root.center_of_mass, root.mass, root.cell_radius, particles, energy, dim)
    energy[:, :, 0] = particles[:, :, 1]

def compute_energy_and_tree_structure(mass, particles, energy, dim):
    """
    Calcul de l'énergie avec Barnes-Hut et extraction de la structure de l'arbre.

    Args:
        mass: Tableau des masses des particules.
        particles: Tableau des particules (positions et vitesses).
        energy: Tableau pour stocker les forces/énergies.
        dim: Nombre de dimensions de l'espace.

    Returns:
        tree_structure: Dictionnaire décrivant la structure de l'arbre :
            - "cell_centers": Coordonnées des centres des cellules.
            - "cell_radii": Dimensions des cellules.
            - "masses": Masses des cellules et des particules.
            - "children": Indices des enfants de chaque cellule.
            - "center_of_mass": Centres de masse des cellules.
    """
    bmin = np.min(particles[:, :, 0], axis=0)
    bmax = np.max(particles[:, :, 0], axis=0)
    root = TreeArray(bmin, bmax, particles.shape[0], dim)

    root.buildTree(particles)
    root.computeMassDistribution(particles, mass)
    compute_force(root.nbodies, root.child, root.center_of_mass, root.mass, root.cell_radius, particles, energy, dim)
    energy[:, :, 0] = particles[:, :, 1]

    tree_structure = {
        "cell_centers": root.cell_center,  # Centres des cellules
        "cell_radii": root.cell_radius,    # Rayons des cellules
        "masses": root.mass,  # Masses
        "children": root.child,      # Enfants de chaque cellule
        "center_of_mass": root.center_of_mass,  # Centres de masse
    }

    return tree_structure
