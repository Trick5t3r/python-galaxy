import numpy as np
from ..forces import force
from .d_numba_functions import buildTree, computeMassDistribution, computeForce

class TreeArray:
    def __init__(self, bmin, bmax, size, dim):
        self.nbodies = size
        self.dim = dim

        self.child = -np.ones((2**dim) * (2*size+1), dtype=np.int32)
        self.bmin = np.asarray(bmin)
        self.bmax = np.asarray(bmax)
        self.center = 0.5 * (self.bmin + self.bmax)
        self.box_size = self.bmax - self.bmin

        self.ncell = 0
        self.cell_center = np.zeros((2*size+1, dim))
        self.cell_radius = np.zeros((2*size+1, dim))
        self.cell_center[0] = self.center
        self.cell_radius[0] = self.box_size

    def buildTree(self, particles):
        self.ncell = buildTree(
            self.center, self.box_size, self.child, self.cell_center, self.cell_radius, particles, self.dim
        )

    def computeMassDistribution(self, particles, mass):
        self.mass = np.zeros(self.nbodies + self.ncell + 1)
        self.mass[:self.nbodies] = mass
        self.center_of_mass = np.zeros((self.nbodies + self.ncell + 1, self.dim))
        self.center_of_mass[:self.nbodies] = particles[:, :, 0]

        computeMassDistribution(
            self.nbodies, self.ncell, self.child, self.mass, self.center_of_mass, self.dim
        )

    def computeForce(self, p):
        return computeForce(
            self.nbodies, self.child, self.center_of_mass, self.mass, self.cell_radius, p, self.dim
        )

    def __str__(self):
        indent = ' ' * 2
        s = 'Tree :\n'
        for i in range(self.ncell + 1):
            s += indent + f'cell {i}\n'
            cell_elements = self.child[self.nbodies + (2**self.dim) * i : self.nbodies + (2**self.dim) * i + (2**self.dim)]
            s += 2 * indent + f'box: {self.cell_center[i] - self.cell_radius[i]} {self.cell_center[i] + self.cell_radius[i]} \n'
            s += 2 * indent + f'particles: {cell_elements[np.logical_and(0 <= cell_elements, cell_elements < self.nbodies)]}\n'
            s += 2 * indent + f'cells: {cell_elements[cell_elements >= self.nbodies] - self.nbodies}\n'
        
        return s

