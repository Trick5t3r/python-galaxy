# pyGalaxy

In this project, we study the nbody problem and the Barnes-Hut algorithm.

The Barnes-Hut algorithm uses a tree data structure to store the bodies and perform the computation of the acceleration in nlog(n). Unfortunately, in order to use Numba on this problem, it is not suitable to have this type of data structure. So, we write the tree using an array where the first components are the bodies and the folowing entries are the cells which are represented by 4 integers (ie the quad tree).

For more information about the Barnes-Hut algorithm

https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation?oldid=469278664

## Installation of dependencies

To install dependencies, we strongly encourage to use a [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or a [`conda env`](https://conda.io/docs/user-guide/tasks/manage-environments.html). 

For example, you can create your conda environment and install the required modules :

```
conda create -n galaxy 
conda activate galaxy
conda install numpy scipy matplotlib docopt numba jupyter plotly pyopengl
```

## Examples

There are two examples in the `examples` directory of each version:

- solar system 
- two galaxies with 3000 bodies

To try the examples just run the examples doing : 

```
cd examples
python galaxy.py
```

You can print an help test doing:

`python galaxy.py -h`. 

If you have installed `opengl`, you can specify the renderer with

`python galaxy.py -R opengl`

For solar system, a notebook is also available.


# Contributors
Check the [CONTRIBUTORS.md](CONTRIBUTORS.md) file.
