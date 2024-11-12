#!/usr/bin/env python

"""
This program uses the Barnes-Hut algorithm to solve the N-body problem on the
solar system.


Usage:
    solar.py [options]

Options:
    -R, --render=<render_option>    The typology of render engine to be used. By
                                    default it uses `matplotlib`. The other
                                    option is to use the more fancy `opengl`, or
                                    `file` to store output into a file.
                                    [default: matplotlib]

    --step=<step>                   Simulation step between each render
                                    [default: 5]
"""


import importlib
import numpy as np
from docopt import docopt
# autopep8: off
import sys
sys.path.append('../')
import pygalaxy
from pygalaxy.barnes_hut_array import compute_energy
# autopep8: on


class SolarSystem:
    def __init__(self, dt=pygalaxy.physics.day_in_sec, display_step=1):
        self.mass, self.particles = pygalaxy.init_solar_system()
        # self.time_method = pygalaxy.RK4(dt, self.particles.shape[0],
        # compute_energy)
        # self.time_method = pygalaxy.Euler_symplectic(dt,
        # self.particles.shape[0], compute_energy)
        self.time_method = pygalaxy.Optimized_815(dt, self.particles.shape[0],
                                                  compute_energy)
        self.display_step = display_step

    def next(self):
        for i in range(self.display_step):
            self.time_method.update(self.mass, self.particles)

    def coords(self):
        return self.particles[:, :2]


if __name__ == '__main__':
    args = docopt(__doc__)

    display_step = int(args['--step'])
    render_engine = args['--render']

    # Importing the right class for rendering from the right module
    anim_module = importlib.import_module('pygalaxy.'+render_engine)
    Animation = getattr(anim_module, 'Animation')

    sim = SolarSystem(10*pygalaxy.physics.day_in_sec,
                      display_step=display_step)

    bmin = np.min(sim.coords(), axis=0)
    bmax = np.max(sim.coords(), axis=0)
    xmin = -1.25*np.max(np.abs([bmin, bmax]))

    anim = Animation(sim, [xmin, -xmin, xmin, -xmin])
    anim.main_loop()
