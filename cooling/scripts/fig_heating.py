from common import *
import ion_trapping
import sys


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
t_max = 1.0e-6
num_steps = 100


for (i, delta) in enumerate(np.linspace(-1.0 * gamma, 1.0 * gamma, 20)):
    my_ensemble = initial_state.copy()
    trap_potential.reset_phase()

    in_plane_cooling = coldatoms.RadiationPressure(gamma, hbar * np.array([k, 0.0, 0.0]),
                                                   GaussianBeam(in_plane_S0, np.array([0.0, sigma, 0.0]), np.array([k, 0.0, 0.0]), sigma),
                                                   DopplerDetuning(delta, np.array([k, 0.0, 0.0])))
    f = open('heating_run_0_' + str(i) + '.dat', 'w')

    kin_in_plane = [kinetic_energy_in_plane(my_ensemble, mode_analysis.wrot)]
    kin_out_of_plane = [kinetic_energy_out_of_plane(my_ensemble)]
    t = 0.0
    f.write('Delta = %.3e gamma\n' % (delta / gamma))
    f.write(' step      t        T_in_plane/mK   T_out_of_plane/mK   T_tot/mK\n')
    format_string = '%5d  %.2e      %.5e       %.5e       %.5e\n'
    for i in range(num_steps):
        t_in_plane = kin_in_plane[-1] / (num_ions * (2.0 / 2.0) * kB * 1.0e-3)
        t_out_of_plane = kin_out_of_plane[-1] / (num_ions * (1.0 / 2.0) * kB * 1.0e-3)
        t_tot = (kin_in_plane[-1] + kin_out_of_plane[-1]) / (num_ions * (3.0 / 2.0) * kB * 1.0e-3)
        f.write(format_string %
                (i, t, t_in_plane, t_out_of_plane, t_tot))
        f.flush()
        evolve_ensemble(dt, t_max, my_ensemble, mode_analysis.B,
                        forces + [in_plane_cooling] + axial_cooling)
        t += t_max
        kin_in_plane.append(kinetic_energy_in_plane(my_ensemble, mode_analysis.wrot))
        kin_out_of_plane.append(kinetic_energy_out_of_plane(my_ensemble))

    f.close()


