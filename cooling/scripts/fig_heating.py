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


kin = [kinetic_energy(my_ensemble, mode_analysis.wrot)]
pot = [potential_energy(my_ensemble,
        mode_analysis.B, mode_analysis.wrot,
        trap_potential.phi,
        trap_potential.kx,
        trap_potential.ky,
        trap_potential.kz)]
t = 0.0
print(' step      t        Ekin         Epot         Etot      Temperature/mK')
format_string = '%5d  %.2e  %.5e  %.5e  %.5e    %.5e'
for i in range(num_steps):
    print(format_string%
            (i, t, kin[-1], pot[-1], kin[-1] + pot[-1],
                kin[-1] / num_ions / 1.5 / kB))
    evolve_ensemble(dt, t_max, my_ensemble, mode_analysis.B,
                    forces + [in_plane_cooling] + axial_cooling)
    t += t_max
    kin.append(kinetic_energy(my_ensemble, mode_analysis.wrot))
    pot.append(potential_energy(my_ensemble,
            mode_analysis.B, mode_analysis.wrot,
            trap_potential.phi,
            trap_potential.kx,
            trap_potential.ky,
            trap_potential.kz))

print(format_string %
        (num_steps, t, kin[-1], pot[-1], kin[-1] + pot[-1],
            kin[-1] / num_ions / 1.5 / kB))
