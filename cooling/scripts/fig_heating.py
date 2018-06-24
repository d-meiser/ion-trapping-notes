from common import *
import ion_trapping
import sys
import os


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
num_steps = 10000

axial_S0 = 0.0
in_plane_S0 = 0.5
offset = 0.0 * sigma
T_initial = 30.0e-3
#axial_cooling = [
#    coldatoms.RadiationPressure(gamma, hbar * np.array([0.0, 0.0, k]),
#                                UniformBeam(S0=axial_S0),
#                                DopplerDetuning(-0.5 * gamma, np.array([0.0, 0.0, k]))),
#    coldatoms.RadiationPressure(gamma, hbar * np.array([0.0, 0.0, -k]),
#                                UniformBeam(S0=axial_S0),
#                                DopplerDetuning(-0.5 * gamma, np.array([0.0, 0.0, -k]))),
#                            ]

Rc = np.max(initial_state.x[:,0])
print('Rc == ' + str(Rc))

for (i, delta) in enumerate(np.linspace(-5.0 * gamma, 0.0, 10)):
    my_ensemble = initial_state.copy()
    delta_v = np.sqrt(2.0 * kB * T_initial / my_ensemble.ensemble_properties['mass'])
    # TODO: We really need to add the velocities in the rotating frame. I'm
    # hacking it here. This velocity distribution is actually rather non-uniform.
    my_ensemble.v[:,:2] += np.random.normal(0, 0.3*delta_v, my_ensemble.v[:,:2].shape)
    trap_potential.reset_phase()

    in_plane_cooling = coldatoms.RadiationPressure(gamma, hbar * np.array([k, 0.0, 0.0]),
                                                   GaussianBeam(in_plane_S0, np.array([0.0, offset, 0.0]), np.array([k, 0.0, 0.0]), sigma),
                                                   DopplerDetuning(delta - 2.0 * np.pi * frot * offset * k, np.array([k, 0.0, 0.0])))
    f = open(os.path.join('data', 'heating_run_7_' + str(i) + '.dat'), 'w')

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
        # No axial cooling:
        evolve_ensemble(dt, t_max, my_ensemble, mode_analysis.B,
                        forces + [in_plane_cooling])
        t += t_max
        kin_in_plane.append(kinetic_energy_in_plane(my_ensemble, mode_analysis.wrot))
        kin_out_of_plane.append(kinetic_energy_out_of_plane(my_ensemble))

    f.close()


