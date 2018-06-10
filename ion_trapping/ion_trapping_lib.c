#include <ion_trapping_lib.h>
#include <stdio.h>
#include <math.h>


static double sqr(double x) {
	return x * x;
}


struct Vec3 {
	double x;
	double y;
	double z;
};

void angular_damping(
	int num_ptcls,
	double omega,
	double kappa_dt,
	const double *x,
	double *v
	)
{
	int i;
	const struct Vec3 *r = (const struct Vec3 *)x;
	struct Vec3 *speed = (struct Vec3 *)v;
	const double expMinusKappaDt = exp(-kappa_dt);
	double radius;
	double v_hat[2];
	double v_par[2];
	double v_perp[2];
	double v_target[2];
	double v_updated[2];
	double v_hat_dot_speed;

	for (i = 0; i < num_ptcls; ++i) {
		radius = sqrt(sqr(r[i].x) + sqr(r[i].y));

		v_hat[0] = -r[i].y / radius;
		v_hat[1] = r[i].x / radius;

		v_hat_dot_speed = v_hat[0] * speed[i].x + v_hat[1] * speed[i].y;
		v_par[0] = v_hat_dot_speed * v_hat[0];
		v_par[1] = v_hat_dot_speed * v_hat[1];

		v_perp[0] = speed[i].x - v_par[0];
		v_perp[1] = speed[i].y - v_par[1];

		v_target[0] = omega * radius * v_hat[0];
		v_target[1] = omega * radius * v_hat[1];

		v_updated[0] = v_perp[0] + v_target[0] +
			(v_par[0] - v_target[0]) * expMinusKappaDt;
		v_updated[1] = v_perp[1] + v_target[1] +
			(v_par[1] - v_target[1]) * expMinusKappaDt;

		speed[i].x = v_updated[0];
		speed[i].y = v_updated[1];
	}
}

void axial_damping(
	int num_ptcls,
	double kappa_dt,
	double *v
	)
{
	int i;
	const double expMinusKappaDt = exp(-kappa_dt);
	struct Vec3 *speed = (struct Vec3 *)v;

	for (i = 0; i < num_ptcls; ++i) {
		speed[i].z = expMinusKappaDt * speed[i].z;
	}
}

// electrostatic constant 1 / (4 pi epsilon_0)
static const double k_e = 8.9875517873681764e9;

double coulomb_energy(
	int num_ptcls,
	const double *x,
	double charge
	)
{
	double energy = 0.0;
	int i, j, m;
	const double *xi;
	const double *xj;
	double r;
	double k_e_q_squared = charge * charge * k_e;

	for (i = 0; i < num_ptcls; ++i) {
		xi = x + i * 3;
		for (j = 0; j < i; ++j) {
			xj = x + j * 3;
			r = 0.0;
			for (m = 0; m < 3; ++m) {
				r += (xi[m] - xj[m]) * (xi[m] - xj[m]);
			}
			r = sqrt(r);
			energy += k_e_q_squared / r;
		}
	}
	return energy;
}

double coulomb_energy_per_particle_charge(
	int num_ptcls,
	const double *x,
	const double *charge
	)
{
	double energy = 0.0;
	int i, j, m;
	const double *xi;
	const double *xj;
	double qi, qj;
	double r;

	for (i = 0; i < num_ptcls; ++i) {
		xi = x + i * 3;
		qi = charge[i];
		for (j = 0; j < i; ++j) {
			xj = x + j * 3;
			qj = charge[j];
			r = 0.0;
			for (m = 0; m < 3; ++m) {
				r += (xi[m] - xj[m]) * (xi[m] - xj[m]);
			}
			r = sqrt(r);
			energy += k_e * qi * qj / r;
		}
	}
	return energy;
}

double trap_energy(
	int num_ptcls,
	const double *x,
	double kx,
	double ky,
	double kz,
	double theta,
	double charge,
	double mass,
	double omega,
	double B_z
	)
{
	double energy = 0.0;
	double xr, yr, zr;
	int i;

	for (i = 0; i < num_ptcls; ++i) {
		xr = cos(theta) * x[i * 3 + 0] - sin(theta) * x[i * 3 + 1];
		yr = sin(theta) * x[i * 3 + 0] + cos(theta) * x[i * 3 + 1];
		zr = x[i * 3 + 2];

		energy += 0.5 * charge * (
			kz * zr * zr +
			kx * xr * xr +
			ky * yr * yr +
			(B_z * omega  - 2.0 * mass * omega * omega) *
			(xr * xr + yr * yr));

	}
	return energy;
}

double kinetic_energy(
	int num_ptcls,
	const double *x,
	const double *v,
	double mass,
	double omega
	)
{
	double energy = 0.0;
	int i;
	double vr[3];

	for (i = 0; i < num_ptcls; ++i) {
		vr[0] = v[3 * i + 0];
		vr[1] = v[3 * i + 1];
		vr[0] -= (-omega) * x[3 * i + 1];
		vr[1] += (-omega) * x[3 * i + 0];
		vr[2] = v[3 * i + 2];
		energy += 0.5 * mass * (
			vr[0] * vr[0] + vr[1] * vr[1] + vr[2] * vr[2]);
	}
	return energy;
}

double kinetic_energy_in_plane(
	int num_ptcls,
	const double *x,
	const double *v,
	double mass,
	double omega)
{
	double energy = 0.0;
	int i;
	double vr[2];

	for (i = 0; i < num_ptcls; ++i) {
		vr[0] = v[3 * i + 0];
		vr[1] = v[3 * i + 1];
		vr[0] -= (-omega) * x[3 * i + 1];
		vr[1] += (-omega) * x[3 * i + 0];
		energy += 0.5 * mass * (vr[0] * vr[0] + vr[1] * vr[1]);
	}
	return energy;
}

double kinetic_energy_out_of_plane(
	int num_ptcls,
	const double *x,
	const double *v,
	double mass)
{
	double energy = 0.0;
	int i;

	for (i = 0; i < num_ptcls; ++i) {
		energy += 0.5 * mass * v[3 * i + 0] * v[3 * i + 0];
	}
	return energy;
}

