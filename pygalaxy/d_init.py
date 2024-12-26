
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
    planets = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 
           'saturn', 'uranus', 'neptune']

    return mass, bodies, planets[:mass.shape[0]]

def getOrbitalVelocity(position_b, mb, positions_s, dim=3):
    """
    Calculate the orbital velocity required for circular orbit for multiple positions.

    Args:
        position_b: Coordinates of the central body (e.g., Sun).
        mb: Mass of the central body.
        positions_s: Coordinates of the orbiting bodies as a numpy array.
        dim: Number of dimensions.

    Returns:
        velocities: Velocity vectors for the orbiting bodies.
    """
    position_b = np.array(position_b, dtype=np.float64)  # Convert to float
    positions_s = np.array(positions_s, dtype=np.float64)  # Convert to float

    r = positions_s - position_b[:dim]  # Vector difference
    dist = np.linalg.norm(r, axis=1)  # Euclidean distance for each star
    dist[dist == 0] = np.finfo(np.float64).eps  # Avoid division by zero

    v = np.sqrt(gamma_si * mb / dist)  # Orbital velocity magnitude
    velocities = np.zeros_like(r, dtype=np.float64)

    # Generate velocity vectors perpendicular to r
    if dim == 2:
        velocities[:, 0] = -r[:, 1] / dist * v
        velocities[:, 1] = r[:, 0] / dist * v
    elif dim == 3:
        for i, vec in enumerate(r):
            ortho = np.cross(vec, np.array([0, 0, 1], dtype=np.float64) if vec[2] == 0 else np.array([1, 0, 0], dtype=np.float64))
            ortho /= np.linalg.norm(ortho)
            velocities[i] = np.cross(ortho, vec) / np.linalg.norm(vec) * v[i]

    return velocities

def init_collisions(blackHole, dim=3):
    """
    Initialize a collision simulation with stars around black holes.

    Args:
        blackHole: List of black hole configurations.
        dim: Number of dimensions.

    Returns:
        mass: Array of masses.
        particles: Array of particle positions and velocities.
    """
    npart = sum(b['stars'] for b in blackHole) + len(blackHole)
    particles = np.zeros((npart, dim, 2), dtype=np.float64)  # Ensure float64
    mass = np.zeros(npart, dtype=np.float64)

    ind = 0
    for b in blackHole:
        b['coord'] = np.array(b['coord'], dtype=np.float64)  # Convert to NumPy array
        particles[ind, :, 0] = b['coord']
        mass[ind] = b['mass']

        velocity = getOrbitalVelocity(
            blackHole[0]['coord'], blackHole[0]['mass'], [b['coord']], dim
        )[0]
        particles[ind, :, 1] = b['svel'] * velocity

        ind += 1

        # Add stars around the black hole
        nstars = b['stars']
        rad = b['radstars']
        r = 0.3 + 0.8 * (rad * np.random.rand(nstars))  # Radial distance
        angles = 2 * np.pi * np.random.rand(nstars, dim - 1)
        positions = np.zeros((nstars, dim), dtype=np.float64)
        positions[:, 0] = r
        for d in range(1, dim):
            positions[:, d] = positions[:, d - 1] * np.sin(angles[:, d - 1])
            positions[:, d - 1] *= np.cos(angles[:, d - 1])
        positions += b['coord']

        velocity = getOrbitalVelocity(b['coord'], b['mass'], positions, dim)
        particles[ind:ind+nstars, :, 0] = positions
        particles[ind:ind+nstars, :, 1] = velocity
        mass[ind:ind+nstars] = 0.03 + 20 * np.random.rand(nstars)
        ind += nstars

    return mass, particles
