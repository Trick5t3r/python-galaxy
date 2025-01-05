old_version = True
if old_version == 2:
    from . import physics
    from .init import init_solar_system, init_collisions
    from .time_schemes.euler import Euler, Euler_symplectic, Euler_symplectic_tree
    from .time_schemes.rk4 import RK4
    from .time_schemes.adb6 import ADB6
    from .time_schemes.stormer import Stormer_verlet, Optimized_815
else:
    from . import physics
    from .d_init import init_solar_system, init_collisions
    from .time_schemes.d_euler import Euler, Euler_symplectic, Euler_symplectic_tree
    from .time_schemes.d_rk4 import RK4
    from .time_schemes.d_adb6 import ADB6
    from .time_schemes.d_stormer import Stormer_verlet, Optimized_815
