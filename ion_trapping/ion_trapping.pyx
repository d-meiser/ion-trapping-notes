import cython
import numpy as np
cimport numpy as np
cimport ion_trapping_lib
import coldatoms


@cython.boundscheck(False)
@cython.wraparound(False)
def angular_damping(omega, kappa_dt,
                    np.ndarray[double, ndim=2, mode="c"] x not None,
                    np.ndarray[double, ndim=2, mode="c"] v not None):
    assert(x.shape[1] == 3)
    assert(v.shape[1] == 3)
    assert(x.shape[0] == v.shape[0])

    cdef num_ptcls
    num_ptcls = x.shape[0]

    ion_trapping_lib.angular_damping(
        num_ptcls,
        omega,
        kappa_dt,
        &x[0, 0],
        &v[0, 0]
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def axial_damping(kappa_dt,
                  np.ndarray[double, ndim=2, mode="c"] v not None):
    assert(v.shape[1] == 3)

    cdef num_ptcls
    num_ptcls = v.shape[0]

    ion_trapping_lib.axial_damping(
        num_ptcls,
        kappa_dt,
        &v[0, 0]
    )


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


@cython.boundscheck(False)
@cython.wraparound(False)
def coulomb_energy(np.ndarray[double, ndim=2, mode="c"] x not None,
        charge):
    assert(x.shape[1] == 3)
    assert(np.isscalar(charge))
    num_ptcls = x.shape[0]
    return ion_trapping_lib.coulomb_energy(num_ptcls, &x[0, 0], charge)


@cython.boundscheck(False)
@cython.wraparound(False)
def coulomb_energy_per_particle_charge(
        np.ndarray[double, ndim=2, mode="c"] x not None,
        np.ndarray[double, ndim=1, mode="c"] charges):
    assert(x.shape[1] == 3)
    assert(charges.shape[0] == x.shape[0])
    num_ptcls = x.shape[0]
    return ion_trapping_lib.coulomb_energy_per_particle_charge(
            num_ptcls, &x[0, 0], &charges[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def trap_energy(np.ndarray[double, ndim=2, mode="c"] x not None,
        kx, ky, kz, theta,
        charge, mass, omega, B_z):
    assert(x.shape[1] == 3)
    num_ptcls = x.shape[0]
    return ion_trapping_lib.trap_energy(num_ptcls, &x[0, 0],
            kx, ky, kz, theta, charge, mass, omega, B_z)


@cython.boundscheck(False)
@cython.wraparound(False)
def kinetic_energy(
        np.ndarray[double, ndim=2, mode="c"] x not None,
        np.ndarray[double, ndim=2, mode="c"] v not None,
        mass, omega):
    assert(x.shape[1] == 3)
    assert(v.shape[1] == 3)
    assert(x.shape[0] == v.shape[0])
    num_ptcls = x.shape[0]
    return ion_trapping_lib.kinetic_energy(num_ptcls,
            &x[0, 0], &v[0, 0],
            mass, omega)


@cython.boundscheck(False)
@cython.wraparound(False)
def kinetic_energy_in_plane(
        np.ndarray[double, ndim=2, mode="c"] x not None,
        np.ndarray[double, ndim=2, mode="c"] v not None,
        mass, omega):
    assert(x.shape[1] == 3)
    assert(v.shape[1] == 3)
    assert(x.shape[0] == v.shape[0])
    num_ptcls = x.shape[0]
    return ion_trapping_lib.kinetic_energy_in_plane(num_ptcls,
            &x[0, 0], &v[0, 0],
            mass, omega)


@cython.boundscheck(False)
@cython.wraparound(False)
def kinetic_energy_out_of_plane(
        np.ndarray[double, ndim=2, mode="c"] x not None,
        np.ndarray[double, ndim=2, mode="c"] v not None,
        mass):
    assert(x.shape[1] == 3)
    assert(v.shape[1] == 3)
    assert(x.shape[0] == v.shape[0])
    num_ptcls = x.shape[0]
    return ion_trapping_lib.kinetic_energy_out_of_plane(num_ptcls,
            &x[0, 0], &v[0, 0], mass)
