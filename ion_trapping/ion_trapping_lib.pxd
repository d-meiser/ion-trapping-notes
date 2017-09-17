cdef extern from "ion_trapping_lib.h":
    int foo();
    void angular_damping(
	    int num_ptcls,
	    double omega,
	    double kappa_dt,
	    const double *x,
	    double *v
	);
