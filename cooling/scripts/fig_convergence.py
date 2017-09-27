import numpy as np
import matplotlib.pyplot as plt
import time

import mode_analysis_code
import coldatoms

# Enable LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {
    'text.latex.preamble':
    [r'\usepackage{siunitx}', r'\usepackage{amsmath}'],
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
    'axes.labelsize': 'medium'
}
plt.rcParams.update(params)


line_width = 246.0 / 72.0
default_width = 0.95 * line_width 
golden_ratio = 1.61803398875
default_height = default_width / golden_ratio


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


initial_state = coldatoms.json_to_ensemble(open('initial_state_180kHz.txt').read())


t_max = 1.0e-5
dt_ref = 2.0e-10

ref_solution = initial_state.copy()
trap_potential.reset_phase()

t0 = time.clock()
evolve_ensemble(dt_ref, t_max, ref_solution, mode_analysis.B, forces)
t1 = time.clock()
t_elapsed = t1 - t0
print('total elapsed time: ', t_elapsed)
print('time per iter:      ', t_elapsed / (t_max / dt_ref))

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


def compute_error(dt, initial_state, ref_sol):
    trap_potential.reset_phase()
    my_ensemble = initial_state.copy()
    evolve_ensemble(dt, t_max, my_ensemble, mode_analysis.B, forces)
    return mean_error(ref_sol.x, my_ensemble.x)


dt_min = 1.0e-9
dt_max = 1.0e-5
num_pts = 100
ddt = (dt_max / dt_min) ** (1.0 / (num_pts - 1.0))
dts = np.array([dt_min * ddt**e for e in range(num_pts)])


errors = [compute_error(dt, initial_state, ref_solution) for dt in dts]


plt.clf()
plt.loglog(dts, errors)

quadratic_dts = dts[15:42]
quadratic = quadratic_dts**2 * (8.0e-9 / dts[0]**2)
plt.loglog(quadratic_dts, quadratic, '--')

plt.xlabel(r'$\Delta t /\si{s}$')
plt.ylabel(r'$\Delta x /\si{m}$')
plt.xlim([0.8e-9, 1.2e-5])
plt.ylim([1.0e-8, 1.1e1])
plt.gcf().set_size_inches([default_width, default_height])
plt.subplots_adjust(left=0.18, right=0.97, top=0.96, bottom=0.21)
plt.savefig('fig_convergence.pdf')

