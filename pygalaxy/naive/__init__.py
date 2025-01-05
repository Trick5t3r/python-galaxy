dim = 3
if dim == 2:
    from .energy import compute_energy
else:
    from .d_energy import compute_energy