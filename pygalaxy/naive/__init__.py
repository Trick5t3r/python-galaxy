old_version = True
if old_version:
    from .energy import compute_energy
else:
    from .d_energy import compute_energy