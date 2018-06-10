from common import *
import ion_trapping


def potential_energy(ensemble, B_z, omega, theta, kx, ky, kz):
    energy = 0
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


def kinetic_energy(ensemble, omega):
    energy = ion_trapping.kinetic_energy(
            ensemble.x, ensemble.v,
            ensemble.ensemble_properties['mass'], omega)
    return energy


def kinetic_energy_in_plane(ensemble, omega):
    energy = ion_trapping.kinetic_energy_in_plane(
            ensemble.x, ensemble.v,
            ensemble.ensemble_properties['mass'], omega)
    return energy


def kinetic_energy_out_of_plane(ensemble):
    energy = ion_trapping.kinetic_energy_out_of_plane(
            ensemble.x, ensemble.v,
            ensemble.ensemble_properties['mass'])
    return energy


def total_energy(ensemble, B_z, omega, theta, kx, ky, kz):
    energy = 0
    energy += potential_energy(ensemble, B_z, omega, theta, kx, ky, kz)
    energy += kinetic_energy(ensemble, omega)
    return energy


dt = 1.0e-9
t_max = 5.0e-7
num_steps = 50
my_ensemble = initial_state.copy()
trap_potential.reset_phase()


kin_in_plane = [kinetic_energy_in_plane(my_ensemble, mode_analysis.wrot)]
kin_out_of_plane = [kinetic_energy_out_of_plane(my_ensemble)]
t = 0.0
print(' step      t        T_in_plane/mK   T_out_of_plane/mK   T_tot/mK')
format_string = '%5d  %.2e      %.5e       %.5e       %.5e'
for i in range(num_steps):
    t_in_plane = kin_in_plane[-1] / (num_ions * (2.0 / 2.0) * kB * 1.0e-3)
    t_out_of_plane = kin_out_of_plane[-1] / (num_ions * (1.0 / 2.0) * kB * 1.0e-3)
    t_tot = (kin_in_plane[-1] + kin_out_of_plane[-1]) / (num_ions * (3.0 / 2.0) * kB * 1.0e-3)
    print(format_string%
            (i, t, t_in_plane, t_out_of_plane, t_tot))
    evolve_ensemble(dt, t_max, my_ensemble, mode_analysis.B,
                    forces + [in_plane_cooling] + axial_cooling)
    t += t_max
    kin_in_plane.append(kinetic_energy_in_plane(my_ensemble, mode_analysis.wrot))
    kin_out_of_plane.append(kinetic_energy_out_of_plane(my_ensemble))


