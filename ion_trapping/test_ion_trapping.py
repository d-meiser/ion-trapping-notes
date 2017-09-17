import ion_trapping
import numpy as np


def test_can_call_foo():
    assert(ion_trapping.foo() == 42)


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
