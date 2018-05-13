from common import *


dt = 1.0e-9
t_max = 1.0e-7
my_ensemble = initial_state.copy()
trap_potential.reset_phase()
evolve_ensemble(dt, t_max, my_ensemble, mode_analysis.B,
                forces + [in_plane_cooling] + axial_cooling)
print(my_ensemble.x)
print(my_ensemble.v)

#TODO:
# - measure temperature
# - create a plot
