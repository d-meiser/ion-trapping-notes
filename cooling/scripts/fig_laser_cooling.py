import numpy as np
import matplotlib.pyplot as plt
import time
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


class UniformBeam(object):
    """A laser beam with a uniform intensity profile."""

    def __init__(self, S0):
        """Construct a Gaussian laser beam from position, direction, and width.

        S0 -- Peak intensity (in units of the saturation intensity).
        k -- Propagation direction of the beam (need not be normalized).
        """
        self.S0 = S0

    def intensities(self, x):
        return self.S0 * np.ones_like(x[:, 0])


class DopplerDetuning(object):
    def __init__(self, Delta0, k):
        self.Delta0 = Delta0
        self.k = np.copy(k)

    def detunings(self, x, v):
        return self.Delta0 - np.inner(self.k, v)


num_ptcls = 10
ensemble = coldatoms.Ensemble(num_ptcls=num_ptcls)
amu = 1.66057e-27
m_Be = 9.012182 * amu
ensemble.ensemble_properties['mass'] = m_Be
wavelength = 313.0e-9
k = 2.0 * np.pi / wavelength
gamma = 2.0 * np.pi * 18.0e6
hbar = 1.0e-34
sigma = 1.0e0

wave_vectors = [
    np.array([k, 0.0, 0.0]),
    np.array([-k, 0.0, 0.0]),
    np.array([0.0, k, 0.0]),
    np.array([0.0, -k, 0.0]),
    np.array([0.0, 0.0, k]),
    np.array([0.0, 0.0, -k]),
]

three_d_mot = [
    coldatoms.RadiationPressure(gamma, hbar * k,
                      UniformBeam(S0=0.1),
                      DopplerDetuning(-0.5 * gamma, k)) for k in wave_vectors]


# Reset positions and velocities to initial conditions.
v0 = 10.0
ensemble.x *= 0.0
ensemble.v *= 0.0
ensemble.v[:, 0] = v0

# The time step size.
dt = 1.0e-6
num_steps = 10000

v = []
t = []
# Now do the time integration and record velocities and times.
for i in range(num_steps):
    v.append(np.copy(ensemble.v))
    t.append(i * dt)
    coldatoms.drift_kick(dt=dt, ensemble=ensemble,forces=three_d_mot)
v = np.array(v)
t = np.array(t)


plt.clf()
num_vis_steps = 500
for i in range(5):
    plt.plot(t[:num_vis_steps] / 1.0e-6, v[:num_vis_steps, i, 0])

plt.xlabel(r'$t /\si{\us}$')
plt.ylabel(r'$v_x/(\si{\meter/\second})$')
plt.gcf().set_size_inches([default_width, default_height])
plt.subplots_adjust(left=0.18, right=0.97, top=0.96, bottom=0.21)
plt.savefig('fig_laser_cooling.pdf')

num_equilibrium_steps = 5000
v_rms = 0.0
for i in range(num_ptcls):
    v_rms += np.linalg.norm(v[num_equilibrium_steps:, i, 0])**2
    v_rms += np.linalg.norm(v[num_equilibrium_steps:, i, 1])**2
    v_rms += np.linalg.norm(v[num_equilibrium_steps:, i, 2])**2
v_rms = np.sqrt(v_rms / (num_steps - num_equilibrium_steps) / num_ptcls)
print(v_rms)

v_rms_theory = np.sqrt(hbar * gamma/ ensemble.ensemble_properties['mass'])
print(v_rms_theory)
