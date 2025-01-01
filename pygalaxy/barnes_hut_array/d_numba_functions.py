import numpy as np
import numba
from ..d_forces import force
from ..physics import gamma_si, theta

@numba.njit
def buildTree(center0, box_size0, child, cell_center, cell_radius, particles, dim):
    ncell = 0
    nbodies = particles.shape[0]

    for ip in range(nbodies):
        center = center0.copy()
        box_size = box_size0.copy()
        position = particles[ip, :, 0]
        cell = 0

        childPath = 0
        for d in range(dim):
            if position[d] > center[d]:
                childPath += 1 << d

        childIndex = nbodies + (2**dim) * cell + childPath

        while (child[childIndex] > nbodies):    
            cell = child[childIndex] - nbodies
            center[:] = cell_center[cell]
            childPath = 0

            for d in range(dim):
                if position[d] > center[d]:
                    childPath += 1 << d

            childIndex = nbodies + (2**dim) * cell + childPath



        # no particle on this cell, just add it
        if (child[childIndex] == -1):
            child[childIndex] = ip
            child[ip] = cell
        # this cell already has a particle
        # subdivide and set the two particles
        elif (child[childIndex] < nbodies):
            npart = child[childIndex]

            oldchildPath = newchildPath = childPath
            while (oldchildPath == newchildPath):
                ncell += 1
                child[childIndex] = nbodies + ncell 
                center[:] = cell_center[cell]
                box_size[:] = 0.5 * cell_radius[cell]

                # Ajuster le centre pour `oldchildPath`
                for d in range(dim):
                    if (oldchildPath >> d) & 1:  # Vérifie le bit `d` de `oldchildPath`
                        center[d] += box_size[d]
                    else:
                        center[d] -= box_size[d]

                # Recalculer `oldchildPath` pour `npart`
                oldchildPath = 0
                for d in range(dim):
                    if particles[npart, d, 0] > center[d]:
                        oldchildPath += 1 << d

                # Recalculer `newchildPath` pour `ip`
                newchildPath = 0
                for d in range(dim):
                    if particles[ip, d, 0] > center[d]:
                        newchildPath += 1 << d

                # Créer une nouvelle cellule
                cell = ncell
                cell_center[ncell] = center
                cell_radius[ncell] = box_size

                # Calculer le nouvel index pour `oldchildPath`
                childIndex = nbodies + (2**dim) * ncell + oldchildPath

            # Assigner les particules aux cellules enfants
            child[childIndex] = npart
            child[npart] = ncell

            childIndex = nbodies + (2**dim) * ncell + newchildPath
            child[childIndex] = ip
            child[ip] = ncell

    return ncell



@numba.njit
def computeForce(nbodies, child_array, center_of_mass, mass, cell_radius, p, dim):
    """
    Calcule la force pour une particule donnée en utilisant l'arbre.

    Arguments :
    - nbodies : Nombre de particules.
    - child_array : Tableau contenant les indices des enfants des cellules.
    - center_of_mass : Tableau des centres de masse.
    - mass : Tableau des masses.
    - cell_radius : Tableau des rayons des cellules.
    - p : Position de la particule pour laquelle la force est calculée.
    - dim : Dimension du problème.

    Retourne :
    - acc : Accélération résultante.
    """
    depth = 0
    max_depth = 2 * nbodies  # Taille maximale pour les structures locales
    localPos = np.zeros(max_depth, dtype=np.int32)
    localNode = np.zeros(max_depth, dtype=np.int32)
    localNode[0] = nbodies

    pos = p[:, 0]
    acc = np.zeros(dim)

    while depth >= 0:
        while localPos[depth] < 2**dim:
            # Obtenir l'enfant actuel
            child_idx = localNode[depth] + localPos[depth]
            child = child_array[child_idx]
            localPos[depth] += 1

            if child >= 0:
                if child < nbodies:
                    # Feuille : calculer la force avec la particule
                    acc += force(pos, center_of_mass[child], mass[child])
                else:
                    # Cellule intermédiaire
                    dx = center_of_mass[child] - pos
                    dist = np.sqrt(np.sum(dx**2))
                    if dist > 0 and np.max(cell_radius[child - nbodies]) / dist < theta:
                        # La cellule est suffisamment éloignée, ajouter la force
                        acc += force(pos, center_of_mass[child], mass[child])
                    else:
                        # Descendre dans l'arbre
                        depth += 1
                        localNode[depth] = nbodies + (2**dim) * (child - nbodies)
                        localPos[depth] = 0
        # Remonter dans l'arbre
        depth -= 1

    return acc



@numba.njit
def computeMassDistribution(nbodies, ncell, child, mass, center_of_mass, dim):
    max_child_index = len(child)

    for i in range(ncell, -1, -1):
        this_mass = 0.0
        this_center_of_mass = np.zeros(dim)

        child_start = nbodies + (2**dim) * i
        child_end = child_start + (2**dim)

        # Validation des indices
        if child_start >= max_child_index or child_end > max_child_index:
            continue

        for j in range(child_start, child_end):
            element_id = child[j]
            if element_id >= 0:
                this_mass += mass[element_id]
                this_center_of_mass += center_of_mass[element_id] * mass[element_id]

        if this_mass > 0:
            center_of_mass[nbodies + i] = this_center_of_mass / this_mass
            mass[nbodies + i] = this_mass
        else:
            center_of_mass[nbodies + i] = np.zeros(dim)
            mass[nbodies + i] = 0.0
