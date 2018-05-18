import ion_trapping
import numpy as np


def angular_damping_reference(omega, kappa_dt, x, v):
    xx = x[:, 0]
    xy = x[:, 1]
    xz = x[:, 2]
    vx = v[:, 0]
    vy = v[:, 1]
    vz = v[:, 2]

    expMinusKappaDt = np.exp(-kappa_dt)

    num_ptcls = x.shape[0]

    for i in range(num_ptcls):
        r = np.sqrt(xx[i] * xx[i] + xy[i] * xy[i])
        speed = np.array([vx[i], vy[i]])
        v_hat = np.array([-xy[i], xx[i]]) / r
        v_par = v_hat.dot(speed) * v_hat
        v_perp = speed - v_par
        v_target = omega * r * v_hat

        v_updated = v_perp + v_target + (v_par - v_target) * expMinusKappaDt
        v[i, 0] = v_updated[0]
        v[i, 1] = v_updated[1]


def test_angular_damping():
    num_ptcls = 20
    x = np.random.random([num_ptcls, 3])
    v = np.random.random([num_ptcls, 3])
    omega = 1.0e5
    kappa_dt = 1.0e-1
    v_ref = np.copy(v)
    angular_damping_reference(omega, kappa_dt, x, v_ref)
    ion_trapping.angular_damping(omega, kappa_dt, x, v)

    epsilon = 1.0e-9
    normalization = np.linalg.norm(v) + np.linalg.norm(v_ref)
    assert(np.linalg.norm(v - v_ref) / normalization < epsilon)


def axial_damping_reference(kappa_dt, v):
    assert(v.shape[1] == 3)
    vz = v[:, 2]
    damping_factor = np.exp(-kappa_dt)
    v[:, 2] = damping_factor * vz


def test_axial_damping():
    num_ptcls = 20
    v = np.random.random([num_ptcls, 3])
    kappa_dt = 1.0e-1
    v_ref = np.copy(v)
    axial_damping_reference(kappa_dt, v_ref)
    ion_trapping.axial_damping(kappa_dt, v)

    epsilon = 1.0e-9
    normalization = np.linalg.norm(v) + np.linalg.norm(v_ref)
    assert(np.linalg.norm(v - v_ref) / normalization < epsilon)


def coulomb_energy_reference(x, q):
    num_ptcls = x.shape[0]
    energy = 0.0
    k_e = 1.0 / (4.0 * np.pi * 8.854187817e-12)
    for i in range(num_ptcls):
        for j in range(i):
            r = np.linalg.norm(x[i] - x[j])
            energy += k_e * q * q / r
    return energy


def coulomb_energy_per_particle_charge_reference(x, q):
    num_ptcls = x.shape[0]
    energy = 0.0
    k_e = 1.0 / (4.0 * np.pi * 8.854187817e-12)
    for i in range(num_ptcls):
        for j in range(i):
            r = np.linalg.norm(x[i] - x[j])
            energy += k_e * q[i] * q[j] / r
    return energy


def rel_error(a, b):
    return np.abs(a - b) / (np.abs(a) + np.abs(b))


def test_coulomb_energy():
    num_ptcls = 5
    x = np.random.random([num_ptcls, 3]) - 0.5
    q = 3.5

    energy = ion_trapping.coulomb_energy(x, q)
    energy_ref = coulomb_energy_reference(x, q)
    assert(rel_error(energy, energy_ref) < 1.0e-6)
    

def test_coulomb_energy_per_particle_charge():
    num_ptcls = 5
    x = np.random.random([num_ptcls, 3]) - 0.5
    q = np.random.random(num_ptcls) - 0.5

    energy = ion_trapping.coulomb_energy_per_particle_charge(x, q)
    energy_ref = coulomb_energy_per_particle_charge_reference(x, q)
    assert(rel_error(energy, energy_ref) < 1.0e-6)
    

def transform_to_rotating_frame(x, v, omega):
    transformed_v = np.zeros_like(v)
    transformed_v[:, 0] = v[:, 0] + omega * x[:, 1]
    transformed_v[:, 1] = v[:, 1] - omega * x[:, 0]
    transformed_v[:, 2] = v[:, 2]
    return transformed_v


def test_temperature_of_particles_at_rest_is_zero():
    num_ptcls = 5
    x = np.random.random([num_ptcls, 3]) - 0.5
    v = np.zeros_like(x)
    omega = 134.0
    v = transform_to_rotating_frame(x, v, omega)

    m = 23.0
    kin_energy = ion_trapping.kinetic_energy(x, v, m, omega)
    assert(abs(kin_energy) < 1.0e-6)


