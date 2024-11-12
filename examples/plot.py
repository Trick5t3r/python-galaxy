#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# load data from file render (saved by NumPy)
particles = np.load("particles.npy")
mass = np.load("mass.npy")

time = np.linspace(0, 10, particles.shape[0])

# particles is a 3D array
# particles[i, j, k] means:
# - ith iteration
# - jth body
# - kth composant of data on particles (x, y, vx, vy)
#
# mass is a 1D array:
# mass[i] means: mass of ith body
#
# time isn't stored in simulation, need to create one with the correct shape (particles.shape[0])


# first plot: history of a particles

fig = plt.figure(figsize=(6, 4))
fig.suptitle("history of position")
fig.subplots_adjust(top=1.25, bottom=-0.25, left=-0.5, right=1.5)
gs = fig.add_gridspec(1, 1)
axlist = []

ax = fig.add_subplot(gs[:, 0], projection='3d')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("time")

for i in range(1, 11):
    x = particles[:, i, 0]
    y = particles[:, i, 1]
    vx = particles[:, i, 2]
    vy = particles[:, i, 3]
    m = mass[i]
    ax.plot(x, y, time, lw=int(m))

plt.show()

# second plot: history of velocity (magnitude)

plt.title("history of velocity")
for i in range(1, 11):
    vx = particles[:, i, 2]
    vy = particles[:, i, 3]
    velocity = np.sqrt(vx**2 + vy**2)
    plt.plot(time, velocity)

plt.xlabel("time")
plt.ylabel("velocity")

plt.show()

# third plot: history of velocity in a 3d plot

fig = plt.figure(figsize=(6, 4))
fig.suptitle("history of velocity")
fig.subplots_adjust(top=1.25, bottom=-0.25, left=-0.5, right=1.5)
gs = fig.add_gridspec(1, 1)
axlist = []

ax = fig.add_subplot(gs[:, 0], projection='3d')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("time")

vmax = max([max(np.sqrt(particles[:, i, 2]**2 + particles[:, i, 3]**2))
           for i in range(1, 11)])

for i in range(1, 11):
    x = particles[:, i, 0]
    y = particles[:, i, 1]
    vx = particles[:, i, 2]
    vy = particles[:, i, 3]

    ax.quiver(x, y, time, 5.*vx/vmax, 5.*vy/vmax, 0.*time, color=f"C{i}")

plt.show()
