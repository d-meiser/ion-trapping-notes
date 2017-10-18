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



# Now specify the cooling laser configuration
amu = 1.66057e-27
m_Be = 9.012182 * amu
initial_state.ensemble_properties['mass'] = m_Be
wavelength = 313.0e-9
k = 2.0 * np.pi / wavelength
gamma = 2.0 * np.pi * 18.0e6
hbar = 1.0e-34
sigma = 1.0e0
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
        return self.S0 * np.exp(-np.linalg.norm(xperp, axis=1)**2/self.sigma)


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
                                               DopplerDetuning(-0.5 * gamma + sigma * mode_analysis.wr *k, np.array([k, 0.0, 0.0])))

dt = 1.0e-9
t_max = 1.0e-4
my_ensemble = initial_state.copy()
trap_potential.reset_phase()
evolve_ensemble(dt, t_max, my_ensemble, mode_analysis.B,
                forces + [in_plane_cooling] + axial_cooling)
plt.clf()
plt.plot(my_ensemble.x[:,0], my_ensemble.x[:,1],'o')
plt.savefig('scr_top_view.pdf')

plt.clf()
plt.plot(1.0e6 * my_ensemble.x[:,0], 1.0e6 * my_ensemble.x[:,2], 'o', ms=2)
plt.xlim([-120, 120])
plt.ylim([-0.3, 0.3])
plt.ylabel(r'$y/\si{\micro\meter}$')
plt.xlabel(r'$x/\si{\micro\meter}$')
plt.gcf().set_size_inches([default_width,default_height])
plt.subplots_adjust(left=0.19, right=0.99, top=0.99, bottom=0.21)
plt.savefig('scr_side_view.pdf')


# Now record trajectories for spectra
dt = 1.0e-9
sampling_period = 5.0e-7
num_samples = 10000
finite_temperature_ensemble = my_ensemble.copy()
trap_potential.reset_phase()
trajectories = [finite_temperature_ensemble.x.copy()]

for i in range(num_samples):
    evolve_ensemble(dt, sampling_period,
                    finite_temperature_ensemble, mode_analysis.B, forces)
    trajectories.append(finite_temperature_ensemble.x.copy())
trajectories = np.array(trajectories)

nu_nyquist = 0.5 / sampling_period
nu_axis = np.linspace(-nu_nyquist, nu_nyquist, trajectories.shape[0])
psd =  np.sum(np.abs(np.fft.fft(trajectories[:,:,2],axis=0))**2, axis=1)

fig = plt.figure()
spl = fig.add_subplot(111)
spl.fill_between(nu_axis / 1.0e6, 1.0e-20, psd)
spl.set_yscale("log")
plt.semilogy(nu_axis / 1.0e6, psd, linewidth=0.75)
plt.xlabel(r'$\nu / \rm{MHz}$')
plt.ylabel(r'PSD($z$)')
plt.xlim([0.0, 0.8 * nu_nyquist/1.0e6])
plt.ylim([1.0e-13, 1.0e-7])
plt.gcf().set_size_inches([default_width, default_height])
plt.subplots_adjust(left=0.2, right=0.97, top=0.98, bottom=0.2)
plt.savefig('fig_axial_spectrum.pdf')
