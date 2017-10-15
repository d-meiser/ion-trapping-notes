import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import time

import mode_analysis_code
import coldatoms
import ion_trapping

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


mode_analysis = mode_analysis_code.ModeAnalysis(N=127,
                                                Vtrap=(0.0, -1750.0, -1970.0),
                                                Vwall=1.0,
                                                frot=185.0)
mode_analysis.run()


def create_ensemble(uE, omega_z, mass, charge):
    num_ions = int(uE.size / 2)
    x = uE[:num_ions]
    y = uE[num_ions:]
    r = np.sqrt(x**2 + y**2)
    r_hat = np.transpose(np.array([x / r, y / r]))
    phi_hat = np.transpose(np.array([-y / r, x / r]))
    v = np.zeros([num_ions, 2], dtype=np.float64)
    for i in range(num_ions):
        v[i, 0] = omega_z * r[i] * phi_hat[i, 0]
        v[i, 1] = omega_z * r[i] * phi_hat[i, 1]

    ensemble = coldatoms.Ensemble(num_ions)
    for i in range(num_ions):
        ensemble.x[i, 0] = x[i]
        ensemble.x[i, 1] = y[i]
        ensemble.x[i, 2] = 0.0
        ensemble.v[i, 0] = v[i, 0]
        ensemble.v[i, 1] = v[i, 1]
        ensemble.v[i, 2] = 0.0

    ensemble.ensemble_properties['mass'] = mass
    ensemble.ensemble_properties['charge'] = charge

    return ensemble



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

trap_potential = TrapPotential(2.0 * mode_analysis.Coeff[2], mode_analysis.Cw, mode_analysis.wrot, np.pi / 2.0)


# Dampings

class AngularDamping(object):

    def __init__(self, omega, kappa_theta):
        self.omega = omega
        self.kappa_theta = kappa_theta

    def dampen(self, dt, ensemble):
        ion_trapping.angular_damping(self.omega, self.kappa_theta * dt, ensemble.x, ensemble.v)


class AxialDamping(object):
    """Doppler cooling along z without recoil."""

    def __init__(self, kappa):
        """kappa is the damping rate."""
        self.kappa = kappa

    def dampen(self, dt, ensemble):
        ion_trapping.axial_damping(self.kappa * dt, ensemble.v)


def evolve_ensemble_with_damping(dt, t_max, ensemble, Bz, forces, dampings):
    num_steps = int(np.floor(t_max / dt))
    for i in range(num_steps):
        coldatoms.bend_kick(dt, Bz, ensemble, forces, num_steps=1)
        for d in dampings:
            d.dampen(dt, ensemble)
    fractional_dt = t_max - (num_steps * dt)
    if (fractional_dt / dt > 1.0e-6):
        coldatoms.bend_kick(fractional_dt, Bz, ensemble, forces, num_steps=1)
        for d in dampings:
            d.dampen(fractional_dt, ensemble)


def radius(x):
    return np.sqrt(x[:, 0]**2 + x[:, 1]**2)

def speed(v):
    return np.sqrt(v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2)

def radial_velocity(x, v):
    num_ptcls = v.shape[0]
    velocity = np.zeros(num_ptcls)
    for i in range(num_ptcls):
        r_hat = np.copy(x[i, :2])
        r_hat /= np.linalg.norm(r_hat)
        velocity[i] = r_hat.dot(v[i, :2])
    return velocity

def angular_velocity(x, v):
    num_ptcls = v.shape[0]
    velocity = np.zeros(num_ptcls)
    for i in range(num_ptcls):
        r_hat = np.copy(x[i, :2])
        r_hat /= np.linalg.norm(r_hat)
        v_r = r_hat * (r_hat.dot(v[i, :2]))
        velocity[i] = np.linalg.norm(v[i, :2] - v_r)
    return velocity


def make_sweep(omega_0, t_hold, t_sweep, delta_omega, t_hold_2):
    """Creates an omega(t).

    The frequency is held at omega_0 for t_hold. Then it increases linearly
    with a sweet rate delta_omega/t_sweep for t_sweep. For t_hold + t_sweep < t < t_hold + t_sweep +t_hold_2
    the frequency is constant again."""

    ts = np.array([0,
                   t_hold,
                   t_hold + t_sweep,
                   t_hold + t_sweep + t_hold_2,
                   t_hold + t_sweep + t_hold_2 + t_sweep
                  ])
    omegas = np.array([omega_0, omega_0, omega_0 + delta_omega, omega_0 + delta_omega, omega_0])
    omega_of_t = lambda t: np.interp(t, ts, omegas, left=omega_0, right=omegas[-1])
    return omega_of_t


# First just make a plot of the frequency sweep omega(t)

t_max = 4.0e-3
dt = 1.0e-9
# We start out at 185kHz
omega_0 = 2.0 * np.pi * 185.0e3
# And stay there for a couple of revolutions
t_hold = 1.0e-5
# Then we sweep by 20kHz in 20 revolutions
t_sweep = 1.0e-3
delta_omega = 2.0 * np.pi * 20e3
# Stay at that level for a millisecond, then sweep back
t_hold_2 = 1.0e-3
my_omega_of_t = make_sweep(omega_0, t_hold, t_sweep, delta_omega, t_hold_2)
t = np.linspace(-1.0e-5, t_max, 200)
omega = my_omega_of_t(t)
plt.figure()
plt.plot(t / 1.0e-3, omega / (1.0e3 * 2.0 * np.pi));
plt.xlabel(r'$t/\si{\milli \second}$')
plt.ylabel(r'$\omega / (2\pi \si{\kilo\hertz})$')
plt.savefig('fig_frequency_sweep.pdf')


# Now we simulate the frequency sweep

kappa_theta = 5.0e6
kappa_z = 1.0e6

my_trap_potential = TrapPotential(2.0 * mode_analysis.Coeff[2],
                                  mode_analysis.Cw,
                                  my_omega_of_t(0.0),
                                  np.pi / 2.0)
my_axial_damping = AxialDamping(kappa=kappa_z)
my_angular_damping = AngularDamping(omega=my_omega_of_t(0.0), kappa_theta=kappa_theta)


# We start out with the steady state distribution with some Gaussian noise in
# the z velocities to seed the multi-plane instability
vz_perturbation = 1.0e-3
my_ensemble = create_ensemble(mode_analysis.uE,
                              my_omega_of_t(0.0),
                              mode_analysis.m_Be,
                              mode_analysis.q)
my_ensemble.v[:, 2] += np.random.normal(loc=0.0, scale=vz_perturbation, size=my_ensemble.num_ptcls)

# Take a snap shot every micro second
snap_shot_times = np.linspace(0, t_max, 4000)
snap_shot_times = np.round(snap_shot_times / dt) * dt

snapshots = []
thicknesses = []
times = []
omegas = []


t_start = time.clock()
t = -dt
t_max = 4.0e-3
while t < t_max:
    # Get the rotation frequency at the mid-point of the time integration interval
    omega = my_omega_of_t(t + 0.5*dt)

    # And use it for the rotating wall potential and the angular damping
    my_trap_potential.omega = omega
    my_angular_damping.omega = omega

    # Now take a time step
    evolve_ensemble_with_damping(
        dt,
        dt,
        my_ensemble,
        mode_analysis.B,
        [coulomb_force, trap_potential],
        [my_axial_damping, my_angular_damping])
    t += dt

    # Record snapshot
    if np.any(np.abs(snap_shot_times - t) < 0.5 * dt):
        snapshots.append((t, omega, my_ensemble.copy()))

    # Record thickness of cloud
    thicknesses.append(np.std(my_ensemble.x[:,2]))
    times.append(t)
    omegas.append(omega)

t_end = time.clock()

thicknesses = np.array(thicknesses)
times = np.array(times)
omegas = np.array(omegas)

print('total time (seconds): ', t_end - t_start)
print('t per iter (micro seconds): ', 1.0e6 * (t_end - t_start) / times.size)

plt.clf()
fig, ax1 = plt.subplots()

period = 100
ax1.plot(omegas[::period] / (2.0 * np.pi * 1.0e3), thicknesses[::period] / 1.0e-6)
plt.xlabel(r'$\omega/(2\pi\si{\kilo\hertz})$')
plt.ylabel(r'$\Delta z /\si{\micro\meter}$')
plt.ylim([-0.1, 4])
ax1.annotate("", xy=(194, 0.8), xytext=(194, 0.1),arrowprops=dict(arrowstyle="->"))
ax1.annotate("", xy=(190, 0.7), xytext=(191.7, 1.2),arrowprops=dict(arrowstyle="->"))
plt.gcf().set_size_inches([default_width, default_height])

l, b, w, h = [0.25, 0.68, 0.25, 0.25]
ax2 = fig.add_axes([l, b, w, h])
ax2.plot(times[::period] * 1.0e3, omegas[::period] / (2.0 * np.pi * 1.0e3),
         linewidth=0.75)
plt.xlabel(r'$t / \si{\milli \second}$', fontsize=6, labelpad=2)
plt.ylabel(r'$\omega / (2\pi\si{\kilo\hertz})$', fontsize=6, labelpad=2)
plt.xlim([-0.2, 4.2])
plt.ylim([184, 210])
ax2.tick_params(axis='both', which='major', labelsize=6, pad=1, length=2)
ax2.tick_params(axis='both', which='minor', labelsize=6, pad=1, length=2)

plt.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.21)
plt.savefig('fig_single_plane_instability.pdf')
