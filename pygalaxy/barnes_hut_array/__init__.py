dim = 3

if dim == 2:
    from .energy import compute_energy, compute_energy_and_tree_structure
    from .quadTree import quadArray
else:
    from .d_energy import compute_energy, compute_energy_and_tree_structure
    from .dTree import TreeArray