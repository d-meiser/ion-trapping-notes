__doc__ = """In this simulation we cool a beam of ions in a moving frame. This
is to debug cooling of the rotating crystal in a Penning trap with the
perpendicular cooling beam."""


import coldatoms
import numpy as np


# The rotation frequency of the ions in the trap is 180kHz
omega_r = 180.0e3 * 2.0 * np.pi
# We plan to displace the cooling laser about 30 mu m off the center of the
# crystal. At that distance from the center the mean velocity of the ions is
# given by
r = 30.0e-6
v = omega_r * r
# The velocity spread
delta_v = 5.0


# Here is the ensemble
amu = 1.66057e-27
m_Be = 9.012182 * amu
num_ptcls = 1000
initial_state = coldatoms.Ensemble(num_ptcls)
initial_state.ensemble_properties['mass'] = m_Be
# The particles move in the negative x direction
initial_state.v[:, 0] = np.random.normal(loc=-v, scale=delta_v, size=num_ptcls)


# We use the following laser parameters
wavelength = 313.0e-9
k = 2.0 * np.pi / wavelength
gamma = 2.0 * np.pi * 18.0e6
hbar = 1.0e-34
# 30 micron beam radius
sigma = 3.0e-5
in_plane_S0 = 0.5


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


def in_plane_cooling(delta):
    return coldatoms.RadiationPressure(gamma, hbar * np.array([k, 0.0, 0.0]),
        GaussianBeam(in_plane_S0, np.array([0.0, 0.0, 0.0]), np.array([k, 0.0, 0.0]), sigma),
        DopplerDetuning(delta, np.array([k, 0.0, 0.0])))


def velocity_distribution_after_cooling(initial_state, cooling, dt, num_steps):
    ensemble = initial_state.copy()
    for i in range(num_steps):
        coldatoms.drift_kick(dt, ensemble, forces=[cooling])
    return ensemble.v


detunings = -0.5 * gamma - k * v + 5.0 * np.linspace(-gamma, gamma, 30)
num_steps = 1000
dt = 5.0e-8
velocity_spreads = [np.std(
    velocity_distribution_after_cooling(
        initial_state, in_plane_cooling(d), dt, num_steps)[:, 0]) for d in detunings]
