from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import matplotlib.lines
import numpy as np
import pandas as pd
import mode_analysis_code
import coldatoms
import os
import scipy.signal
import scipy.optimize

plt.style.use('ggplot')

# Build a function to generate an `ensemble` from the coldatoms library, eventually
# using the equilibrium conditions of the crystal

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

# Instantiate a means to implement the Coulomb force between ions (Complete. NOT harmonic approx.)

coulomb_force = coldatoms.CoulombForce()

# Instantiate the trapping forces on the crystal

class TrapPotential(object):

    def __init__(self, kz, delta, omega, phi_0):
        self.kz = kz
        self.kx = -(0.5 + delta) * kz
        self.ky = -(0.5 - delta) * kz
        self.phi_0 = phi_0
        self.phi = phi_0
        self.omega = omega

    def reset_phase(self):
        self.phi = self.phi_0
            
    def force(self, dt, ensemble, f):
        self.phi += self.omega * 0.5 * dt
        
        q = ensemble.ensemble_properties['charge']
        if q is None:
            q = ensemble.particle_properties['charge']
            if q is None:
                raise RuntimeError('Must provide ensemble or per particle charge')

        cphi = np.cos(self.phi)
        sphi = np.sin(self.phi)
        kx = self.kx
        ky = self.ky
        
        x = ensemble.x[:, 0]
        y = ensemble.x[:, 1]
        z = ensemble.x[:, 2]
        
        f[:, 0] += dt * q * (
            (-kx * cphi * cphi - ky * sphi * sphi) * x + cphi * sphi * (ky - kx) * y)
        f[:, 1] += dt * q * (
            cphi * sphi * (ky - kx) * x + (-kx * sphi * sphi - ky * cphi * cphi) * y)
        f[:, 2] += -dt * q *self.kz * z

        self.phi += self.omega * 0.5 * dt

# From coldatoms lib, create a function to evolve the crystal

def evolve_ensemble(dt, t_max, ensemble, Bz, forces):
    num_steps = int(t_max / dt)
    coldatoms.bend_kick(dt, Bz, ensemble, forces, num_steps=num_steps)
    coldatoms.bend_kick(t_max - num_steps * dt, Bz, ensemble, forces)

##############################################################
############### INSTANTIATE CRYSTAL & TRAP ###################

mode_analysis = mode_analysis_code.ModeAnalysis(N=127,
                                                Vtrap=(0.0, -1750.0, -2000.0),
                                                Vwall=5.0,
                                                frot=180.0)

mode_analysis.run()

trap_potential = TrapPotential(2.0 * mode_analysis.Coeff[2], mode_analysis.Cw, mode_analysis.wrot, np.pi / 2.0)

##############################################################
####################### EVOLUTION  ###########################

x = mode_analysis.u[:mode_analysis.Nion]
y = mode_analysis.u[mode_analysis.Nion:]

dx = x.reshape((x.size, 1)) - x
dy = y.reshape((y.size, 1)) - y
rsep = np.sqrt(dx ** 2 + dy ** 2)

with np.errstate(divide='ignore'):
    rsep3 = np.where(rsep != 0., rsep ** (-3), 0)

K = np.diag((-1 + 0.5 * np.sum(rsep3, axis=0)))
K -= 0.5 * rsep3
K = np.mat(K)

Mmat = np.diag(mode_analysis.md)
Mmat = np.mat(Mmat)

# Function to perform evolution

def generate_seededPSD(storage_directory, pMode, sMode, totalEnergy, percentPrimaryE,
                    t_max = 10.0e-3, dt = 1.0e-9, num_dump = 50, cooling = False):
    """
    Params:
    * storage_directory: directory to which to write results
    * pMode: Mode to provide majority of energy
    * sMode: Mode to seed with minimal energy
    * totalEnergy: total energy of crystal
    * percentPrimaryE: (Percentage of energy in the primary mode
    * max_time: duration of evolution
    * t_step: time step for each evolution
    
    Returns:
    Nothing.  Instead, saves numpy arrays 'x_freq' and 'PSD_data', obtained from performing FFT^2 on the z position
        time series.

    MAKE SURE YOU HAVE CHOSEN AN EMPTY DIRECTORY (OR DONT CARE ABOUT OVERWRITING)
    """
    
    num_steps = int(np.ceil(t_max / dt))
    
    modal_positions = [] #instantiate array to hold z-positions as crystal evolves
    
    # Create new ensemble from mode_analysis.
    modal_ensemble = create_ensemble(mode_analysis.uE,
                                    mode_analysis.wrot,
                                    mode_analysis.m_Be,
                                    mode_analysis.q)
    
    x = mode_analysis.u[:mode_analysis.Nion]
    y = mode_analysis.u[mode_analysis.Nion:]

    dx = x.reshape((x.size, 1)) - x
    dy = y.reshape((y.size, 1)) - y
    rsep = np.sqrt(dx ** 2 + dy ** 2)

    with np.errstate(divide='ignore'):
        rsep3 = np.where(rsep != 0., rsep ** (-3), 0)

    K = np.diag((-1 + 0.5 * np.sum(rsep3, axis=0)))
    K -= 0.5 * rsep3
    K = np.mat(K)

    Mmat = np.diag(mode_analysis.md)
    Mmat = np.mat(Mmat)
    
    
    # Based on desired mode number, establish ICs with z-components based on eigenmode
    eigVect1a = mode_analysis.axialEvects[:, -2*pMode]
    eigVect1b = mode_analysis.axialEvects[:, -2*pMode + 1]
    
    eigVect2a = mode_analysis.axialEvects[:, -2*sMode]
    eigVect2b = mode_analysis.axialEvects[:, -2*sMode + 1]
    
    
    pCombo = eigVect1a + eigVect1b # Choose this combo to get a 'pure' position state
    sCombo = eigVect2a + eigVect2b # Choose this combo to get a 'pure' position state
    
    pri_pos = pCombo[:mode_analysis.Nion]
    pri_vel = pCombo[mode_analysis.Nion:]
    seed_pos = sCombo[:mode_analysis.Nion]
    seed_vel = sCombo[mode_analysis.Nion:]
    
    pri_x = np.mat(pri_pos.reshape((mode_analysis.Nion,1)))
    pri_v = np.mat(pri_vel.reshape((mode_analysis.Nion,1)))
    seed_x = np.mat(seed_pos.reshape((mode_analysis.Nion,1)))
    seed_v = np.mat(seed_vel.reshape((mode_analysis.Nion,1)))
    
    pri_energy = 0.5*pri_v.H*Mmat*pri_v - 0.5*pri_x.H*K*pri_x
    seed_energy = 0.5*seed_v.H*Mmat*seed_v - 0.5*seed_x.H*K*seed_x
    
    pri_E = totalEnergy * percentPrimaryE
    seed_E = totalEnergy * (1 - percentPrimaryE)
    
    pri_pos = np.sqrt(pri_E/pri_energy) * pri_pos
    pri_vel = np.sqrt(pri_E/pri_energy) * pri_vel
    seed_pos = np.sqrt(seed_E/seed_energy) * seed_pos
    seed_vel = np.sqrt(seed_E/seed_energy) * seed_vel
    
    modal_ensemble.x[:,2] = (pri_pos + seed_pos) * mode_analysis.l0
    modal_ensemble.v[:,2] = (pri_vel + seed_vel) * mode_analysis.v0 # Should be within computer error of zero
    
    # Establish positions and evolve, as per Dominic's Example
    if cooling:
        modal_positions.append(np.copy(modal_ensemble.x))
        for i in range(num_steps // num_dump):
            coldatoms.bend_kick(dt, mode_analysis.B, modal_ensemble,
                               [coulomb_force, trap_potential]+cooling_beams,
                               num_steps = num_dump)
            modal_positions.append(np.copy(modal_ensemble.x))
        modal_positions = np.array(modal_positions)
        
    else:
        modal_positions.append(np.copy(modal_ensemble.x))
        for i in range(num_steps // num_dump):
            coldatoms.bend_kick(dt, mode_analysis.B, modal_ensemble,
                               [coulomb_force, trap_potential],
                               num_steps = num_dump)
            modal_positions.append(np.copy(modal_ensemble.x))
        modal_positions = np.array(modal_positions)
        
    
    # Convert time series to frequency data
    delta_t = num_dump * dt
    nu_nyquist = 0.5 / delta_t
    nu_axis = np.linspace(0.0, 2.0 * nu_nyquist, modal_positions.shape[0])
        
    freq_data = np.sum(np.abs(np.fft.fft(modal_positions[:,:,2], axis=0))**2, axis=1)
    
    np.save(storage_directory + 'freqs', nu_axis)
    np.save(storage_directory + 'PSD_data', freq_data)
