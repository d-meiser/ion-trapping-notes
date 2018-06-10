import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import time

import mode_analysis_code
import coldatoms


kB = 1.38064852e-23


# Preamble for figures
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


# Trap and crystal configuration
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



# Now specify the cooling laser configuration
amu = 1.66057e-27
m_Be = 9.012182 * amu
initial_state.ensemble_properties['mass'] = m_Be
wavelength = 313.0e-9
k = 2.0 * np.pi / wavelength
gamma = 2.0 * np.pi * 18.0e6
hbar = 1.0e-34
sigma = 3.0e-5
axial_S0 = 0.1
in_plane_S0 = 0.1


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


class GaussianBeam(object):
    """A laser beam with a Gaussian intensity profile."""

    def __init__(self, S0, x0, k, sigma):
        """Construct a Gaussian laser beam from position, direction, and width.

        S0 -- Peak intensity (in units of the saturation intensity).
        x0 -- A location on the center of the beam.
        k -- Propagation direction of the beam (need not be normalized).
        sigma -- 1/e width of the beam."""
        self.S0 = S0
        self.x0 = np.copy(x0)
        self.k_hat = k / np.linalg.norm(k)
        self.sigma = sigma

    def intensities(self, x):
        xp = x - self.x0
        xperp = xp - np.outer(xp.dot(self.k_hat[:, np.newaxis]), self.k_hat)
        return self.S0 * np.exp(-np.linalg.norm(xperp, axis=1)**2 / self.sigma**2)


class DopplerDetuning(object):
    def __init__(self, Delta0, k):
        self.Delta0 = Delta0
        self.k = np.copy(k)

    def detunings(self, x, v):
        return self.Delta0 - np.inner(self.k, v)

axial_cooling = [
    coldatoms.RadiationPressure(gamma, hbar * np.array([0.0, 0.0, k]),
                                UniformBeam(S0=axial_S0),
                                DopplerDetuning(-0.5 * gamma, np.array([0.0, 0.0, k]))),
    coldatoms.RadiationPressure(gamma, hbar * np.array([0.0, 0.0, -k]),
                                UniformBeam(S0=axial_S0),
                                DopplerDetuning(-0.5 * gamma, np.array([0.0, 0.0, -k]))),
                            ]

in_plane_cooling = coldatoms.RadiationPressure(gamma, hbar * np.array([k, 0.0, 0.0]),
                                               GaussianBeam(in_plane_S0, np.array([0.0, sigma, 0.0]), np.array([k, 0.0, 0.0]), sigma),
                                               DopplerDetuning(-0.5 * gamma - sigma * 2.0 * np.pi * frot * k, np.array([k, 0.0, 0.0])))

