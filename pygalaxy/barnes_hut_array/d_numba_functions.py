import numpy as np
import numba
from ..d_forces import force
from ..physics import gamma_si, theta

@numba.njit
def buildTree(center0, box_size0, child, cell_center, cell_radius, particles, dim):
    """
    Construit un quadtree généralisé pour une dimension arbitraire `dim`.

    Arguments :
    - center0 : Centre initial du domaine
    - box_size0 : Taille initiale de la boîte
    - child : Tableau des enfants des cellules
    - cell_center : Tableau des centres des cellules
    - cell_radius : Tableau des rayons des cellules
    - particles : Tableau des positions des particules
    - dim : Dimension du problème

    Retourne :
    - ncell : Nombre total de cellules créées
    """
    ncell = 0
    nbodies = particles.shape[0]

    for ip in range(nbodies):
        # Initialisation pour la particule actuelle
        center = center0.copy()
        box_size = box_size0.copy()
        position = particles[ip, :, 0]
        cell = 0

        while True:
            # Calcul de l'index enfant dans la cellule courante
            childPath = 0
            for d in range(dim):
                if position[d] > center[d]:
                    childPath += 1 << d  # Décalage binaire

            childIndex = nbodies + (2**dim) * cell + childPath

            if child[childIndex] == -1:
                # Pas de particule ou cellule ici, place la particule
                child[childIndex] = ip
                child[ip] = cell
                break
            elif child[childIndex] < nbodies:
                # Une particule est déjà ici, créer une nouvelle cellule
                npart = child[childIndex]
                while True:
                    ncell += 1

                    # Créer un nouveau noeud
                    child[childIndex] = nbodies + ncell
                    new_center = center.copy()
                    new_box_size = 0.5 * box_size

                    # Calculer le nouveau centre pour la cellule
                    for d in range(dim):
                        if (childPath >> d) & 1:
                            new_center[d] += new_box_size[d]
                        else:
                            new_center[d] -= new_box_size[d]

                    cell_center[ncell] = new_center
                    cell_radius[ncell] = new_box_size

                    # Réattribuer l'ancienne particule
                    old_childPath = 0
                    for d in range(dim):
                        if particles[npart, d, 0] > new_center[d]:
                            old_childPath += 1 << d

                    new_childIndex = nbodies + (2**dim) * ncell + old_childPath
                    if child[new_childIndex] == -1:
                        child[new_childIndex] = npart
                        child[npart] = ncell
                        break

                    # Passer au niveau suivant
                    npart = child[new_childIndex]

                # Recalculer l'index pour la nouvelle particule
                center = new_center
                box_size = new_box_size

            else:
                # La cellule est une cellule intermédiaire, descendre d'un niveau
                cell = child[childIndex] - nbodies
                for d in range(dim):
                    if (childPath >> d) & 1:
                        center[d] += 0.5 * box_size[d]
                    else:
                        center[d] -= 0.5 * box_size[d]
                box_size *= 0.5

    return ncell



@numba.njit
def computeForce(nbodies, child_array, center_of_mass, mass, cell_radius, p, dim):
    depth = 0
    localPos = np.zeros(2 * nbodies, dtype=np.int32)
    localNode = np.zeros(2 * nbodies, dtype=np.int32)
    localNode[0] = nbodies

    pos = p[:, 0]
    acc = np.zeros(dim)

    while depth >= 0:
        while localPos[depth] < 2**dim:
            child = child_array[localNode[depth] + localPos[depth]]
            localPos[depth] += 1
            if child >= 0:
                if child < nbodies:
                    acc += force(pos, center_of_mass[child], mass[child])
                else:
                    dx = center_of_mass[child] - pos
                    dist = np.sqrt(np.sum(dx**2))
                    if dist > 0 and np.max(cell_radius[child - nbodies]) / dist < theta:
                        acc += force(pos, center_of_mass[child], mass[child])
                    else:
                        depth += 1
                        localNode[depth] = nbodies + (2**dim) * (child - nbodies)
                        localPos[depth] = 0
        depth -= 1
    return acc


@numba.njit
def computeMassDistribution(nbodies, ncell, child, mass, center_of_mass, dim):
    for i in range(ncell, -1, -1):
        this_mass = 0.0
        this_center_of_mass = np.zeros(dim)
        for j in range(nbodies + (2**dim) * i, nbodies + (2**dim) * i + (2**dim)):
            element_id = child[j]
            if element_id >= 0:
                this_mass += mass[element_id]
                this_center_of_mass += center_of_mass[element_id, :] * mass[element_id]

        if this_mass > 0:
            center_of_mass[nbodies + i] = this_center_of_mass / this_mass
            mass[nbodies + i] = this_mass
        else:
            center_of_mass[nbodies + i] = np.zeros(dim)
            mass[nbodies + i] = 0.0
