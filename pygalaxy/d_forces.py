from .physics import gamma_si, eps, gamma_1
import numpy as np
import numba

@numba.njit
def force(p1, p2, m2):
    """
    Calcule la force gravitationnelle entre deux particules en d dimensions.

    Args:
        p1: Position de la première particule ([dim]).
        p2: Position de la deuxième particule ([dim]).
        m2: Masse de la deuxième particule.
        dim: Nombre de dimensions de l'espace.

    Returns:
        F_vec: Vecteur de la force exercée sur la première particule ([dim]).
    """
    dx = p2 - p1  # Différences de positions sur toutes les dimensions
    dist2 = np.sum(dx**2) + eps  # Distance au carré avec adoucissement
    dist = np.sqrt(dist2)  # Distance

    F = 0.
    if dist > 0:
        F = (gamma_si * m2) / (dist * dist2)  # Force gravitationnelle divisée par dist

    return F * dx  # Retourne le vecteur de force
