import numpy as np
import matplotlib.pyplot as plt

import mode_analysis_code
import coldatoms


# Use specific seed to make numpy random number generation reproducible
np.random.seed(2000)


# Configuration
num_ions = 127
frot = 180.0e3
v_wall = 1.0


# Forces
coulomb_force = coldatoms.CoulombForce()


class TrapPotential(object):

    def __init__(self, kz, delta, omega, phi_0):
        self.kz = kz
        self.kx = -(0.5 + delta) * kz
        self.ky = -(0.5 - delta) * kz
        self.phi_0 = phi_0
        self.phi = phi_0
        self.omega = omega
        self.trap_potential = coldatoms.HarmonicTrapPotential(self.kx, self.ky, self.kz)

    def reset_phase(self):
        self.phi = self.phi_0

    def force(self, dt, ensemble, f):
        self.phi += self.omega * 0.5 * dt
        self.trap_potential.phi = self.phi
        self.trap_potential.force(dt, ensemble, f)
        self.phi += self.omega * 0.5 * dt


mode_analysis = mode_analysis_code.ModeAnalysis(N=num_ions,
                                                Vtrap=(0.0, -1750.0, -1970.0),
                                                Vwall=v_wall,
                                                frot=1.0e-3 * frot)
trap_potential = TrapPotential(
    2.0 * mode_analysis.Coeff[2], mode_analysis.Cw, mode_analysis.wrot, np.pi / 2.0)


forces = [coulomb_force, trap_potential]


def evolve_ensemble(dt, t_max, ensemble, Bz, forces):
    num_steps = int(t_max / dt)
    coldatoms.bend_kick(dt, Bz, ensemble, forces, num_steps=num_steps)
    coldatoms.bend_kick(t_max - dt * num_steps, Bz, ensemble, forces)


delta_x = 1.0e-7
delta_v = 1.0e-4

initial_state = coldatoms.json_to_ensemble(open('initial_state_180kHz.txt').read())
initial_state.x += np.random.normal(loc=0.0, scale=delta_x, size=initial_state.x.shape)
initial_state.v += np.random.normal(loc=0.0, scale=delta_v, size=initial_state.v.shape)


t_max = 1.0e-5
dt_ref = 5.0e-10
ref_solution = initial_state.copy()
trap_potential.reset_phase()
evolve_ensemble(dt_ref, t_max, ref_solution, mode_analysis.B, forces)
f = open('reference_solution_180kHz.txt', 'w')
f.write(coldatoms.ensemble_to_json(ref_solution))
f.close()


ref_solution = coldatoms.json_to_ensemble(open('reference_solution_180kHz.txt', 'r').read())


def mean_error(x, y):
    n = x.shape[0]
    assert(n == y.shape[0])
    assert(x.shape[1] == y.shape[1])
    error = 0.0
    for i in range(n):
        error += np.linalg.norm(x[i] - y[i])**2
    error /= n
    error = np.sqrt(error)
    return error


def compute_error(dt, ref_sol):
    trap_potential.reset_phase()
    my_ensemble = initial_state.copy()
    evolve_ensemble(dt, t_max, my_ensemble, mode_analysis.B, forces)
    return mean_error(ref_sol.x, my_ensemble.x)


dt_min = 1.0e-9
dt_max = 1.0e-7
num_pts = 100
ddt = (dt_max / dt_min) ** (1.0 / (num_pts - 1.0))
dts = [dt_min * ddt**e for e in range(num_pts)]


errors = [compute_error(dt, ref_solution) for dt in dts]


plt.clf()
plt.loglog(dts, errors)
plt.savefig('fig_errors.pdf')

