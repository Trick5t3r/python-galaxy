mass_sun = 1.988435e30
gamma_si = 6.67408e-11

pc_in_m = 3.08567758129e16
gamma_1 = gamma_si/(pc_in_m*pc_in_m*pc_in_m)*mass_sun*(365.25*86400)*(365.25*86400)

eps = 1e-2 # prevent division by zero
day_in_sec=86400

theta = 0.5