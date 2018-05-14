from common import *
import ion_trapping


dt = 1.0e-9
t_max = 1.0e-7
my_ensemble = initial_state.copy()
trap_potential.reset_phase()
evolve_ensemble(dt, t_max, my_ensemble, mode_analysis.B,
                forces + [in_plane_cooling] + axial_cooling)


def compute_energy(ensemble, B_z, omega, theta, kx, ky, kz):
    energy = ion_trapping.kinetic_energy(
            ensemble.x, ensemble.v,
            ensemble.ensemble_properties['mass'], omega, theta)
    energy += ion_trapping.trap_energy(
            ensemble.x,
            kx, ky, kz, theta,
            ensemble.ensemble_properties['charge'],
            ensemble.ensemble_properties['mass'],
            omega, B_z)
    energy += ion_trapping.coulomb_energy(
            ensemble.x,
            ensemble.ensemble_properties['charge'])
    return energy


print(compute_energy(my_ensemble,
    mode_analysis.B, mode_analysis.wr,
    trap_potential.phi,
    trap_potential.kx,
    trap_potential.ky,
    trap_potential.kz))
            
            
#TODO:
# - measure temperature
# - create a plot
