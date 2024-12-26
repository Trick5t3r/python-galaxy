from six import print_

import numpy as np
from .quadTree import quadArray
import time

from . import numba_functions
import numba

@numba.njit(parallel=True)
def compute_force( nbodies, child, center_of_mass, mass, cell_radius, particles, energy):
    for i in numba.prange(particles.shape[0]):
        acc = numba_functions.computeForce( nbodies, child, center_of_mass, mass, cell_radius, particles[i] )
        energy[i, 2] = acc[0]
        energy[i, 3] = acc[1]

def compute_energy(mass, particles, energy):
    #print('compute energy:')
    t_tot = time.time()

    bmin = np.min(particles[: ,:2], axis=0)
    bmax = np.max(particles[: ,:2], axis=0)
    root = quadArray(bmin, bmax, particles.shape[0])

    #print_('\tbuild tree:    ', end='', flush=True)
    #t1 = time.time()
    root.buildTree(particles)
    #t2 = time.time()
    #print_('{:9.4f}ms'.format(1000*(t2-t1)))

    #print_('\tcompute mass:  ', end='', flush=True)
    #t1 = time.time()
    root.computeMassDistribution(particles, mass)
    #t2 = time.time()
    #print_('{:9.4f}ms'.format(1000*(t2-t1)))

    #print_('\tcompute force: ', end='', flush=True)
    #t1 = time.time()    
    compute_force( root.nbodies, root.child, root.center_of_mass, root.mass, root.cell_radius, particles, energy )
    energy[:, :2] = particles[:, 2:]
    #t2 = time.time()
    #print_('{:9.4f}ms'.format(1000*(t2-t1)))

    #print_('\ttotal:       {:11.4f}ms'.format(1000*(time.time()-t_tot)))

def compute_energy_and_tree_structure(mass, particles, energy):
    """
    Calcul de l'énergie avec Barnes-Hut et extraction de la structure de l'arbre.
    
    Args:
        mass: Tableau des masses des particules.
        particles: Tableau des particules (positions et vitesses).
        energy: Tableau pour stocker les forces/énergies.

    Returns:
        tree_structure: Dictionnaire décrivant la structure de l'arbre :
            - "cell_centers": Coordonnées des centres des cellules.
            - "cell_radii": Dimensions des cellules.
            - "masses": Masses des cellules et des particules.
            - "children": Indices des enfants de chaque cellule.
            - "center_of_mass": Centres de masse des cellules.
    """
    import time

    # Étape 1 : Déterminer les limites de la boîte englobante
    #t_tot = time.time()
    bmin = np.min(particles[:, :2], axis=0)
    bmax = np.max(particles[:, :2], axis=0)

    # Étape 2 : Initialisation de la structure quadArray
    root = quadArray(bmin, bmax, particles.shape[0])

    # Étape 3 : Construction de l'arbre
    root.buildTree(particles)

    # Étape 4 : Calcul de la distribution de masse
    root.computeMassDistribution(particles, mass)

    # Étape 5 : Calcul des forces
    compute_force(
        root.nbodies, root.child, root.center_of_mass, root.mass, root.cell_radius, particles, energy
    )
    energy[:, :2] = particles[:, 2:]

    # Étape 6 : Extraction de la structure de l'arbre
    tree_structure = {
        "cell_centers": root.cell_center[: root.ncell + 1],  # Centres des cellules
        "cell_radii": root.cell_radius[: root.ncell + 1],    # Rayons des cellules
        "masses": root.mass[: root.nbodies + root.ncell + 1],  # Masses
        "children": root.child[: 4 * (root.ncell + 1)],      # Enfants de chaque cellule
        "center_of_mass": root.center_of_mass[: root.nbodies + root.ncell + 1],  # Centres de masse
    }

    # Temps total
    #print(f"Total time: {1000*(time.time() - t_tot):.4f} ms")

    return tree_structure
