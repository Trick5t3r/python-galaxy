
#!/usr/bin/env python

from OpenGL.GL import *
from OpenGL.GLUT import *
import OpenGL.arrays.vbo as glvbo
from OpenGL.GL import shaders

import sys
import math
import numpy as np
import time
from copy import deepcopy


class Animation(object):
    """ Simulation renderer using OpenGL.

    Press left button to move.
    Press right button to zoom.
    """

    def __init__(self, simu,
                 axis=[0., 1., 0., 1.], size=[640, 480], dim=2, title=b"Animation",
                 use_colors=False, update_colors=False,
                 use_adaptative_opacity=False, start_paused=False):
        """ Initialize an animation view.

        Parameters:
        -----------
        simu: object
            Simulation object with coords and next methods
        axis: list
            Axis bounds [xmin, xmax, ymin, ymax, (zmin, zmax if dim == 3)].
        size: list
            Initial window size [width, height].
        dim: int
            Number of dimensions (2 or 3).
        """

        self.simu = simu
        self.dim = dim
        faxis = [float(v) for v in axis]
        self.axis = Animation._Axis(
            [faxis[0], faxis[2]] if dim == 2 else [faxis[0], faxis[2], faxis[4]],
            max((faxis[1] - faxis[0]) / size[0],
                (faxis[3] - faxis[2]) / size[1])
        )
        self.size = size
        self.mouse_action = None

        # Initialize the OpenGL Utility Toolkit
        glutInit(sys.argv)

        # Initial display mode (RGBA colors and double buffered window)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)

        # Initial window
        glutInitWindowSize(size[0], size[1])
        glutInitWindowPosition(0, 0)
        glutCreateWindow(title)

        # Callbacks
        glutDisplayFunc(self._draw)
        glutIdleFunc(self.draw_next_frame)
        glutReshapeFunc(self._resize)
        glutMouseFunc(self._mouse)
        glutMotionFunc(self._motion)
        glutKeyboardFunc(self._keyboard)

        # Create a Vertex Buffer Object for the vertices
        coords = simu.coords()
        self._star_vbo = glvbo.VBO(coords)
        self._star_count = coords.shape[0]

        # Display options
        self.use_colors_update = update_colors
        self.use_colors = use_colors
        self.adaptative_opacity_factor = self.axis.scale
        self.is_paused = start_paused

    ###########################################################################
    # Internal methods

    def _draw(self):
        """ Called when the window must be redrawn. """

        # Clear the buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT if self.dim == 3 else GL_COLOR_BUFFER_BIT)

        # Update perspective transformation
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.dim == 2:
            glOrtho(self.axis.origin[0],
                    self.axis.origin[0] + self.axis.scale * self.size[0],
                    self.axis.origin[1],
                    self.axis.origin[1] + self.axis.scale * self.size[1],
                    -1, 1)
        elif self.dim == 3:
            gluPerspective(45.0, self.size[0] / self.size[1], 0.1, 100.0)
            glTranslatef(-self.axis.origin[0], -self.axis.origin[1], -self.axis.origin[2])

        # Bind the vertex VBO
        self._star_vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(self.dim, GL_DOUBLE, 0, None)

        # Draw stars
        glDrawArrays(GL_POINTS, 0, self._star_count)

        # Swap display buffers
        glutSwapBuffers()

    def _resize(self, width, height):
        """ Called when the window is resized. """
        self.size = [max(width, 1), max(height, 1)]
        glViewport(0, 0, self.size[0], self.size[1])

    def _update_coords(self):
        """ Update vertex coordinates. """
        coords = self.simu.coords()
        self._star_vbo.set_array(coords)
        self._star_count = coords.shape[0]

    def draw_next_frame(self):
        """ Update simulation data and display it. """
        if not self.is_paused:
            self.simu.next()
            self._update_coords()
        glutPostRedisplay()

    ###########################################################################
    # Internal classes

    class _Axis(object):
        """ View axis. """
        def __init__(self, origin, scale):
            self.origin = origin
            self.scale = scale


###############################################################################
# Demo
if __name__ == '__main__':
    """ Demo """

    class SpinningCloud:
        def __init__(self, size, dim=2, theta=math.pi / 18000):
            self.dim = dim
            self._coords = np.random.randn(size, dim)
            self._rot = self._create_rotation_matrix(theta)

        def _create_rotation_matrix(self, theta):
            if self.dim == 2:
                return np.array([[math.cos(theta), -math.sin(theta)],
                                 [math.sin(theta), math.cos(theta)]])
            elif self.dim == 3:
                return np.array([[math.cos(theta), -math.sin(theta), 0],
                                 [math.sin(theta), math.cos(theta), 0],
                                 [0, 0, 1]])

        def next(self):
            self._coords = np.dot(self._coords, self._rot)

        def coords(self):
            return self._coords

    simu = SpinningCloud(1000, dim=3)
    anim = Animation(simu, axis=[-2, 2, -2, 2, -2, 2], size=[640, 480], dim=3)
    anim.main_loop()
