#include <ion_trapping_lib.h>
#include <stdio.h>
#include <math.h>


static double sqr(double x) {
	return x * x;
}


int foo()
{
	return 42;
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
