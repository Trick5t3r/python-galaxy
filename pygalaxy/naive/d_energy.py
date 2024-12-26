from ..forces import force
import numpy as np
import numba

@numba.njit
def compute_energy(mass, particles, energy, dim):
    """
    Calcul des énergies et forces dans un espace à d dimensions.

    Args:
        mass: Tableau des masses des particules.
        particles: Tableau des particules (positions et vitesses).
        energy: Tableau pour stocker les forces/énergies.
        dim: Nombre de dimensions de l'espace.

    """
    energy[:] = 0.
    N = energy.shape[0]
    for i in range(N):
        for j in range(N):
            if i != j:
                F = force(particles[i, :dim], particles[j, :dim], mass[j])
                energy[i, dim:] += F  # Mise à jour des forces sur les dimensions
    energy[:, :dim] = particles[:, dim:2*dim]  # Mise à jour des vitesses
