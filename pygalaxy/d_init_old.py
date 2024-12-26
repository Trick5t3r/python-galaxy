
import numpy as np
from .physics import gamma_si

def init_solar_system(dim=2):
    """ Initialize the solar system with particles in d dimensions.

    Args:
        dim: Number of dimensions (default is 3).

    Returns:
        mass: Array of masses.
        bodies: Array of particles [nb_particles, dim, x], where x includes position and velocity.
    """
    assert( (dim>=2)&(dim<=3))
    # Initial positions and velocities
    if dim == 2:
        positions = np.array([[        0, 0],       # Sun
                            [    -46e9, 0],       # Mercury
                            [ -10748e7, 0],       # Venus
                            [-147095e6, 0],       # Earth
                            [ -20662e7, 0]])      # Mars

        velocities = np.array([[      0,      0],   # Sun
                            [      0, -58980],    # Mercury
                            [      0, -35260],    # Venus
                            [      0, -30300],    # Earth
                            [      0, -26500]])   # Mars
    elif dim == 3:
        positions = np.array([[        0, 0, 0],       # Sun
                            [    -46e9, 0, 0],       # Mercury
                            [ -10748e7, 0, 0],       # Venus
                            [-147095e6, 0, 0],       # Earth
                            [ -20662e7, 0, 0]])      # Mars

        velocities = np.array([[      0,      0, 0],   # Sun
                            [      0, -58980, 0],    # Mercury
                            [      0, -35260, 0],    # Venus
                            [      0, -30300, 0],    # Earth
                            [      0, -26500, 0]])   # Mars
    else:
        return ValueError

    # Combine positions and velocities
    bodies = np.empty((positions.shape[0], dim, 2))  # [nb_particles, dim, 2]
    bodies[:, :, 0] = positions
    bodies[:, :, 1] = velocities

    # Masses of the bodies
    mass = np.array([  1.989e30,  # Sun
                      0.33011e24, # Mercury
                      4.8675e24,  # Venus
                       5.972e24,  # Earth
                      6.4171e23]) # Mars

    return mass, bodies

def getOrbitalVelocity(position_b, mb, position_s, dim=2):
    """
    Calculate the orbital velocity required for circular orbit.

    Args:
        position_b: Coordinates of the central body (e.g., Sun) as a numpy array.
        mb: Mass of the central body.
        position_s: Coordinates of the orbiting body as a numpy array.
        dim: Number of dimensions.

    Returns:
        velocity: Velocity vector for the orbiting body in d dimensions.
    """
    # Calculate the distance vector and magnitude
    r = position_b[:dim] - position_s[:dim]  # Vector difference
    dist = np.sqrt(np.sum(r**2))  # Euclidean distance

    # Calculate the orbital velocity magnitude
    v = np.sqrt(gamma_si * mb / dist)

    # Create the velocity vector perpendicular to r
    velocity = np.zeros(dim)
    if dim >= 2:
        velocity[0] = -r[1] / dist * v
        velocity[1] = r[0] / dist * v
    if dim > 2:
        # Higher dimensions: additional components remain zero
        velocity[2:] = 0

    return velocity


def init_collisions(blackHole, dim=2):
    """ Initialize a collision simulation with stars around black holes.

    Args:
        blackHole: List of dictionaries describing black hole parameters.
        dim: Number of dimensions.

    Returns:
        mass: Array of masses.
        particles: Array of particles [nb_particles, dim, x].
    """
    npart = sum(b['stars'] for b in blackHole) + len(blackHole)
    particles = np.zeros((npart, dim, 2))  # [nb_particles, dim, 2]
    mass = np.zeros(npart)

    ind = 0
    for b in blackHole:
        # Set black hole data
        particles[ind, :, 0] = b['coord']
        mass[ind] = b['mass']

        if ind == 0:
            particles[ind, :, 1] = 0  # Stationary central black hole
        else:
            velocity = getOrbitalVelocity(
                blackHole[0]['coord'],
                blackHole[0]['mass'], b['coord'], dim
            )
            particles[ind, :, :] = b['svel'] * velocity

        ind += 1

        # Add stars around the black hole
        nstars = b['stars']
        rad = b['radstars']
        r = 0.3 + 0.8 * (rad * np.random.rand(nstars))  # Distance radiale
        angles = 2 * np.pi * np.random.rand(nstars, dim - 1)  # Angles pour les dimensions supplémentaires
        tmp_mass = 0.03 + 20 * np.random.rand(nstars)  # Masse aléatoire des étoiles

        # Convert radial distances and angles to Cartesian coordinates
        positions = np.zeros((nstars, dim))  # Stockage pour les positions
        positions[:, 0] = r  # Initialement, met tout le rayon dans la première coordonnée

        for d in range(1, dim):
            positions[:, d] = positions[:, d - 1] * np.sin(angles[:, d - 1])
            positions[:, d - 1] *= np.cos(angles[:, d - 1])

        # Décaler les positions pour centrer autour du trou noir
        positions += b['coord']


        velocity = getOrbitalVelocity(b['coord'], b['mass'], positions, dim)

        particles[ind:ind+nstars, :, 0] = positions
        particles[ind:ind+nstars, :, 1] = velocity
        mass[ind:ind+nstars] = tmp_mass
        ind += nstars

    return mass, particles
