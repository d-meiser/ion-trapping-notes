from __future__ import division, with_statement
from scipy.constants import pi
import scipy.constants as cons
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import scipy.linalg

__author__ = 'sbt'

# -*- coding: utf-8 -*-
"""
Contains the ModeAnalysis class, which can simulate the positions of ions in a crystal
of desired size. The class contains methods for the generation of a crystal,
relaxation to a minimum potential energy state, and determination of axial and (eventually) planar modes of motion
by methods derived by Wang, Keith, and Freericks in 2013.

Translated from MATLAB code written by Adam Keith by Justin Bohnet.
Standardized and slightly revised by Steven Torrisi.

Be careful. Sometimes when using the exact same parameters this 
code will make different crystals with the same potential energy. That is,
crystal are degenerate when reflecting over the axes.
"""


class ModeAnalysis:
    """
    Simulates a 2-dimensional ion crystal, determining an equilibrium plane configuration given
    Penning trap parameters, and then calculates the eigenvectors and eigenmodes.

    For reference the following ion number correspond the closed shells:
    1  2  3  4  5   6   7   8   9  10  11  12  13  14
    7 19 37 61 91 127 169 217 271 331 397 469 547 631...



    """
    #Establish fundamental physical constants as class variables
    q = 1.602176565E-19
    amu = 1.66057e-27
    m_Be = 9.012182 * amu
    k_e = 8.9875517873681764E9

    def __init__(self, N=19, Vtrap=(0.0, -1750.0, -2000.0), Ctrap=1.0,
                 ionmass=None, B=4.4588, frot=180., Vwall=5., wall_order=2,
                 quiet=True, precision_solving=True):
        """
        :param N:       integer, number of ions
        :param shells:  integer, number of shells to instantiate the plasma with
        :param Vtrap: array of 3 elements, defines the [end, middle, center] voltages on the trap electrodes.
        :param Ctrap: float, constant coefficient on trap potentials
        :param B: float, defines strength of axial magnetic field.
        :param frot: float, frequency of rotation
        :param Vwall: float, strength of wall potential in volts
        :param wall_order: integer, defines the order of the rotating wall potential
        :param mult: float, mutliplicative factor for simplifying numerical calculations
        :param quiet: will print some things if False
        :param precision_solving: Determines if perturbations will be made to the crystal to find
                                    a low energy state with a number of attempts based on the
                                    number of ions.
                                    Disable for speed, but recommended.
        """

        self.quiet = quiet
        self.precision_solving = precision_solving
        # Initialize basic variables such as physical constants
        self.Nion = N
        # self.shells = shells
        # self.Nion = 1 + 6 * np.sum(range(1, shells + 1))

        # if no input masses, assume all ions are beryllium
        if ionmass is None:
            self.m = self.m_Be * np.ones(self.Nion)
            # mass order is irrelevant and don't assume it will be fixed
            # FUTURE: heavier (than Be) ions will be added to outer shells

        # for array of ion positions first half is x, last is y
        self.u0 = np.empty(2 * self.Nion)  # initial lattice
        self.u = np.empty(2 * self.Nion)  # equilibrium positions

        # trap definitions
        self.B = B
        self.wcyc = self.q * B / self.m_Be  # Beryllium cyclotron frequency

        # axial trap coefficients; see Teale's final paper
        self.C = Ctrap * np.array([[0.0756, 0.5157, 0.4087],
                                   [-0.0001, -0.005, 0.005],
                                   [1.9197e3, 3.7467e3, -5.6663e3],
                                   [0.6738e7, -5.3148e7, 4.641e7]])

        # wall order
        if wall_order == 2:
            self.Cw2 = self.q * Vwall * 1612
            self.Cw3 = 0
        if wall_order == 3:
            self.Cw2 = 0
            self.Cw3 = self.q * Vwall * 3e4

        self.relec = 0.01  # rotating wall electrode distance in meters
        self.Vtrap = np.array(Vtrap)  # [Vend, Vmid, Vcenter] for trap electrodes
        self.Coeff = np.dot(self.C, self.Vtrap)  # Determine the 0th, first, second, and fourth order
        #  potentials at trap center
        #self.wz = 4.9951e6  # old trapping frequency
        self.wz = np.sqrt(2 * self.q * self.Coeff[2] / self.m_Be)  # Compute axial frequency
        self.wrot = 2 * pi * frot * 1e3  # Rotation frequency in units of angular fre   quency

        # Not used vvv
        #self.wmag = 0.5 * (self.wcyc - np.sqrt(self.wcyc ** 2 - 2 * self.wz ** 2))
        self.wmag=0 # a hack for now

        self.V0 = (0.5 * self.m_Be * self.wz ** 2) / self.q  # Find quadratic voltage at trap center
        #self.Cw = 0.045 * Vwall / 1000  # old trap
        self.Cw = Vwall * 1612 / self.V0  # dimensionless coefficient in front
        # of rotating wall terms in potential

        self.dimensionless()  # Make system dimensionless

        self.axialEvals = []  # Axial eigenvalues
        self.axialEvects = []  # Axial eigenvectors
        self.planarEvals = []  # Planar eigenvalues
        self.planarEvects = []  # Planar Eigenvectors

        self.axialEvalsE = []  # Axial eigenvalues in experimental units
        self.planarEvalsE = []  # Planar eigenvalues in experimental units

        self.p0 = 0    # dimensionless potential energy of equilibrium crystal
        self.r = []
        self.rsep = []
        self.dx = []
        self.dy = []

        self.hasrun = False

    def dimensionless(self):
        """Calculate characteristic quantities and convert to a dimensionless
        system
        """
        # characteristic length
        self.l0 = ((self.k_e * self.q ** 2) / (.5 * self.m_Be * self.wz ** 2)) ** (1 / 3)
        self.t0 = 1 / self.wz  # characteristic time
        self.v0 = self.l0 / self.t0  # characteristic velocity
        self.E0 = 0.5*self.m_Be*(self.wz**2)*self.l0**2 # characteristic energy
        self.wr = self.wrot / self.wz  # dimensionless rotation
        self.wc = self.wcyc / self.wz  # dimensionless cyclotron
        self.md = self.m / self.m_Be  # dimensionless mass

    def expUnits(self):
        """Convert dimensionless outputs to experimental units"""
        self.u0E = self.l0 * self.u0  # Seed lattice
        self.uE = self.l0 * self.u  # Equilibrium positions
        self.axialEvalsE = self.wz * self.axialEvals
        self.planarEvalsE = self.wz * self.planarEvals
        # eigenvectors are dimensionless anyway

    def run(self):
        """
        Generates a crystal from the generate_crystal method (by the find_scalled_lattice_guess method,
        adjusts it into an eqilibirium position by find_eq_pos method)
        and then computes the eigenvalues and eigenvectors of the axial modes by calc_axial_modes.

        Sorts the eigenvalues and eigenvectors and stores them in self.Evals, self.Evects.
        Stores the radial separations as well.
        """
        if self.wmag > self.wrot:
            print("Warning: Rotation frequency", self.wrot/(2*pi),
                  " is below magnetron frequency of", float(self.wrot/(2*pi)))
            return 0

        self.generate_crystal()

        self.axialEvals, self.axialEvects = self.calc_axial_modes(self.u)
        self.planarEvals, self.planarEvects = self.calc_planar_modes(self.u)
        self.expUnits()  # make variables of outputs in experimental units
        self.hasrun = True

    def generate_crystal(self):
        """
        The run method already is a "start-to-finish" implementation of crystal generation and
        eigenmode determination, so this simply contains the comopnents which generate a crystal.
        :return: Returns a crystal's position vector while also saving it to the class.
        """

        # This check hasn't been working properly, and so wmag has been set to
        # 0 for the time being (July 2015, SBT)
        if self.wmag > self.wrot:
            print("Warning: Rotation frequency", self.wrot/(2*pi),
                  " is below magnetron frequency of", float(self.wrot/(2*pi)))
            return 0

        #Generate a lattice in dimensionless units
        self.u0 = self.find_scaled_lattice_guess(mins=1, res=50)
        # self.u0 = self.generate_2D_hex_lattice(2)

        # if masses are not all beryllium, force heavier ions to be boundary
        # ions, and lighter ions to be near center
        # ADD self.addDefects()

        #Solve for the equilibrium position
        self.u = self.find_eq_pos(self.u0)



        # Will attempt to nudge the crystal to a slightly lower energy state via some
        # random perturbation.
        # Only changes the positions if the perturbed potential energy was reduced.

        #Will perturb less for bigger crystals, as it takes longer depending on how many ions
        #there are.
        if self.precision_solving is True:
            if self.quiet is False:
                print("Perturbing crystal...")


            if self.Nion <= 62:
                for attempt in np.linspace(.05, .5, 50):
                    self.u = self.perturb_position(self.u, attempt)
            if 62 < self.Nion <= 126:
                for attempt in np.linspace(.05, .5, 25):
                    self.u = self.perturb_position(self.u, attempt)
            if 127 <= self.Nion <= 200:
                for attempt in np.linspace(.05, .5, 10):
                    self.u = self.perturb_position(self.u, attempt)
            if 201 <= self.Nion:
                for attempt in np.linspace(.05, .3, 5):
                    self.u = self.perturb_position(self.u, attempt)

            if self.quiet is False:
                pass
                #print("Perturbing complete")

        self.r, self.dx, self.dy, self.rsep = self.find_radial_separation(self.u)
        self.p0 = self.pot_energy(self.u)
        return self.u

    def generate_lattice(self):
        """Generate lattice for an arbitrary number of ions (self.Nion)

        :return: a flattened xy position vector defining the 2d hexagonal
                 lattice
        """
        # number of closed shells
        S = int((np.sqrt(9 - 12 * (1 - self.Nion)) - 3) / 6)
        u0 = self.generate_2D_hex_lattice(S)
        N0 = int(u0.size / 2)
        x0 = u0[0:N0]
        y0 = u0[N0:]
        Nadd = self.Nion - N0  # Number of ions left to add
        self.Nion = N0

        pair = self.add_hex_shell(S + 1)  # generate next complete shell
        xadd = pair[0::2]
        yadd = pair[1::2]

        for i in range(Nadd):
            # reset number of ions to do this calculation
            self.Nion += 1

            # make masses all one (add defects later)
            self.md = np.ones(self.Nion)

            V = []  # list to store potential energies from calculation

            # for each ion left to add, calculate potential energy if that
            # ion is added
            for j in range(len(xadd)):
                V.append(self.pot_energy(np.hstack((x0, xadd[j], y0,
                                                    yadd[j]))))
            ind = np.argmin(V)  # ion added with lowest increase in potential

            # permanently add to existing crystal
            x0 = np.append(x0, xadd[ind])
            y0 = np.append(y0, yadd[ind])

            # remove ion from list to add
            xadd = np.delete(xadd, ind)
            yadd = np.delete(yadd, ind)

        # Restore mass array
        self.md = self.m / self.m_Be  # dimensionless mass
        return np.hstack((x0, y0))

    def pot_energy(self, pos_array):
        """
        Computes the potential energy of the ion crystal, 
        taking into consideration:
            Coulomb repulsion
            qv x B forces
            Trapping potential
            and some other things (#todo to be fully analyzed; june 10 2015)

        :param pos_array: The position vector of the crystal to be analyzed.
        :return: The scalar potential energy of the crystal configuration.
        """
        # Frequency of rotation, mass and the number of ions in the array

        # the x positions are the first N elements of the position array
        x = pos_array[0:self.Nion]
        # The y positions are the last N elements of the position array
        y = pos_array[self.Nion:]

        # dx flattens the array into a row vector
        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        # rsep is the distances between
        rsep = np.sqrt(dx ** 2 + dy ** 2)

        with np.errstate(divide='ignore'):
            Vc = np.where(rsep != 0., 1 / rsep, 0)

        """
        #Deprecated version below which takes into account anharmonic effects, to be used later

        V = 0.5 * (-m * wr ** 2 - q * self.Coeff[2] + q * B * wr) * np.sum((x ** 2 + y ** 2)) \
            - q * self.Coeff[3] * np.sum((x ** 2 + y ** 2) ** 2) \
            + np.sum(self.Cw2 * (x ** 2 - y ** 2)) \
            + np.sum(self.Cw3 * (x ** 3 - 3 * x * y ** 2)) \
            + 0.5 * k_e * q ** 2 * np.sum(Vc)
        """
        V = -np.sum((self.md * self.wr ** 2 + 0.5 * self.md - self.wr * self.wc) * (x ** 2 + y ** 2)) \
            + np.sum(self.md * self.Cw * (x ** 2 - y ** 2)) + 0.5 * np.sum(Vc)

        return V

    def force_penning(self, pos_array):
        """
        Computes the net forces acting on each ion in the crystal;
        used as the jacobian by find_eq_pos to minimize the potential energy
        of a crystal configuration.

        :param pos_array: crystal to find forces of.
        :return: a vector of size 2N describing the x forces and y forces.
        """

        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        rsep = np.sqrt(dx ** 2 + dy ** 2)

        # Calculate coulomb force on each ion
        with np.errstate(divide='ignore'):
            Fc = np.where(rsep != 0., rsep ** (-2), 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            fx = np.where(rsep != 0., np.float64((dx / rsep) * Fc), 0)
            fy = np.where(rsep != 0., np.float64((dy / rsep) * Fc), 0)

        # total force on each ion

        """ Deprecated version below which uses anharmonic trap potentials
        Ftrapx = (-m * wr ** 2 - q * self.Coeff[2] + q * B * wr + 2 * self.Cw2) * x \
            - 4 * q * self.Coeff[3] * (x ** 3 + x * y ** 2) + 3 * self.Cw3 * (x ** 2 - y ** 2)
        Ftrapy = (-m * wr ** 2 - q * self.Coeff[2] + q * B * wr - 2 * self.Cw2) * y \
            - 4 * q * self.Coeff[3] * (y ** 3 + y * x ** 2) - 6 * self.Cw3 * x * y

        # Ftrap =  (m*w**2 + q*self.V0 - 2*q*self.Vw - q*self.B* w) * pos_array
        """
        Ftrapx = -2 * self.md * (self.wr ** 2 - self.wr * self.wc + 0.5 -
                                 self.Cw) * x
        Ftrapy = -2 * self.md * (self.wr ** 2 - self.wr * self.wc + 0.5 +
                                 self.Cw) * y

        Fx = -np.sum(fx, axis=1) + Ftrapx
        Fy = -np.sum(fy, axis=1) + Ftrapy

        return np.array([Fx, Fy]).flatten()

    def hessian_penning(self, pos_array):
        """Calculate Hessian of potential"""

        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        rsep = np.sqrt(dx ** 2 + dy ** 2)
        with np.errstate(divide='ignore'):
            rsep5 = np.where(rsep != 0., rsep ** (-5), 0)
        dxsq = dx ** 2
        dysq = dy ** 2

        # X derivatives, Y derivatives for alpha != beta
        Hxx = np.mat((rsep ** 2 - 3 * dxsq) * rsep5)
        Hyy = np.mat((rsep ** 2 - 3 * dysq) * rsep5)

        # Above, for alpha == beta
        Hxx += np.mat(np.diag(-2 * self.md * (self.wr ** 2 - self.wr * self.wc + .5 -
                                              self.Cw) -
                              np.sum((rsep ** 2 - 3 * dxsq) * rsep5, axis=0)))
        Hyy += np.mat(np.diag(-2 * self.md * (self.wr ** 2 - self.wr * self.wc + .5 +
                                              self.Cw) -
                              np.sum((rsep ** 2 - 3 * dxsq) * rsep5, axis=0)))

        # Mixed derivatives
        Hxy = np.mat(-3 * dx * dy * rsep5)
        Hxy += np.mat(np.diag(3 * np.sum(dx * dy * rsep5, axis=0)))

        H = np.bmat([[Hxx, Hxy], [Hxy, Hyy]])
        H = np.asarray(H)
        return H

    def find_scaled_lattice_guess(self, mins, res):
        """
        Will generate a 2d hexagonal lattice based on the shells intialiization parameter.
        Guesses initial minimum separation of mins and then increases spacing until a local minimum of
        potential energy is found.

        This doesn't seem to do anything. Needs a fixin' - AK

        :param mins: the minimum separation to begin with.
        :param res: the resizing parameter added onto the minimum spacing.
        :return: the lattice with roughly minimized potential energy (via spacing alone).
        """

        # Make a 2d lattice; u represents the position
        uthen = self.generate_lattice()
        uthen = uthen * mins
        # Figure out the lattice's initial potential energy
        pthen = self.pot_energy(uthen)

        # Iterate through the range of minimum spacing in steps of res/resolution
        for scale in np.linspace(mins, 10, res):
            # Quickly make a 2d hex lattice; perhaps with some stochastic procedure?
            uguess = uthen * scale
            # Figure out the potential energy of that newly generated lattice
            # print(uguess)
            pnow = self.pot_energy(uguess)

            # And if the program got a lattice that was less favorably distributed, conclude
            # that we had a pretty good guess and return the lattice.
            if pnow >= pthen:
                # print("find_scaled_lattice: Minimum found")
                # print "initial scale guess: " + str(scale)
                # self.scale = scale
                # print(scale)
                return uthen
            # If not, then we got a better guess, so store the energy score and current arrangement
            # and try again for as long as we have mins and resolution to iterate through.
            uthen = uguess
            pthen = pnow
        # If you're this far it means we've given up
        # self.scale = scale
        # print "find_scaled_lattice: no minimum found, returning last guess"
        return uthen

    def find_eq_pos(self, u0, method="bfgs"):
        """
        Runs optimization code to tweak the position vector defining the crystal to a minimum potential energy
        configuration.

        :param u0: The position vector which defines the crystal.
        :return: The equilibrium position vector.
        """
        newton_tolerance = 1e-34
        bfgs_tolerance = 1e-34
        if method is "newton":

            out = optimize.minimize(self.pot_energy, u0, method='Newton-CG', jac=self.force_penning,
                                    hess=self.hessian_penning,
                                    options={'xtol': newton_tolerance, 'disp': not self.quiet})
        else:
            out = optimize.minimize(self.pot_energy, u0, method='BFGS', jac=self.force_penning,
                                    options={'gtol': bfgs_tolerance, 'disp': False})  # not self.quiet})
        return out.x

    def calc_axial_modes(self, pos_array):
        """
        Calculate the modes of axial vibration for a crystal defined
        by pos_array. 
        
        THIS MAY NEED TO BE EDITED FOR NONHOMOGENOUS MASSES

        :param pos_array: Position vector which defines the crystal
                          to be analyzed.
        :return: Array of eigenvalues, Array of eigenvectors
        """

        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        rsep = np.sqrt(dx ** 2 + dy ** 2)

        with np.errstate(divide='ignore'):
            rsep3 = np.where(rsep != 0., rsep ** (-3), 0)

        K = np.diag((-1 + 0.5 * np.sum(rsep3, axis=0)))
        K -= 0.5 * rsep3
        # Make first order system by making space twice as large
        Zn = np.zeros((self.Nion, self.Nion))
        eyeN = np.identity(self.Nion)
        Mmat = np.diag(self.md)
        Minv = np.linalg.inv(Mmat)
        firstOrder = np.bmat([[Zn, eyeN], [np.dot(Minv,K), Zn]])
        Eval, Evect = np.linalg.eig(firstOrder)
        
        # Convert 2N imaginary eigenvalues to N real eigenfrequencies
        ind = np.argsort(np.absolute(np.imag(Eval)))
        Eval = np.imag(Eval[ind])
        Eval = Eval[Eval >= 0]      # toss the negative eigenvalues
        Evect = Evect[:, ind]     # sort eigenvectors accordingly

        # Normalize by energy of mode
        for i in range(2*self.Nion):
            pos_part = Evect[:self.Nion, i]
            vel_part = Evect[self.Nion:, i]
            norm = vel_part.H*Mmat*vel_part - pos_part.H*K*pos_part

            with np.errstate(divide='ignore',invalid='ignore'):
                Evect[:, i] = np.where(np.sqrt(norm) != 0., Evect[:, i]/np.sqrt(norm), 0)
            #Evect[:, i] = Evect[:, i]/np.sqrt(norm)

        Evect = np.asarray(Evect)
        return Eval, Evect

    def calc_planar_modes(self, pos_array):
        """Calculate Planar Mode Eigenvalues and Eigenvectors

        THIS MAY NEED TO BE EDITED FOR NONHOMOGENOUS MASSES  
        
        :param pos_array: Position vector which defines the crystal
                          to be analyzed.
        :return: Array of eigenvalues, Array of eigenvectors
        """

        V = -self.hessian_penning(pos_array)  # -Hessian
        Zn = np.zeros((self.Nion, self.Nion))
        Z2n = np.zeros((2 * self.Nion, 2 * self.Nion))
        offdiag = (2 * self.wr - self.wc) * np.identity(self.Nion)
        A = np.bmat([[Zn, offdiag], [-offdiag, Zn]])

        Mmat = np.diag(np.concatenate((self.md,self.md)))
        Minv = np.linalg.inv(Mmat)
        
        firstOrder = np.bmat([[Z2n, np.identity(2 * self.Nion)], [np.dot(Minv,V/2), A]])
        #mp.dps = 25
        #firstOrder = mp.matrix(firstOrder)
        #Eval, Evect = mp.eig(firstOrder)
        Eval, Evect = np.linalg.eig(firstOrder)
        # currently giving too many zero modes (increase numerical precision?)

        # make eigenvalues real.
        ind = np.argsort(np.absolute(np.imag(Eval)))
        Eval = np.imag(Eval[ind])
        Eval = Eval[Eval >= 0]      # toss the negative eigenvalues
        Evect = Evect[:, ind]    # sort eigenvectors accordingly

        # Normalize by energy of mode
        for i in range(4*self.Nion):
            pos_part = Evect[:2*self.Nion, i]
            vel_part = Evect[2*self.Nion:, i]
            norm = vel_part.H*Mmat*vel_part - pos_part.H*(V/2)*pos_part

            with np.errstate(divide='ignore'):
                Evect[:, i] = np.where(np.sqrt(norm) != 0., Evect[:, i]/np.sqrt(norm), 0)
            #Evect[:, i] = Evect[:, i]/np.sqrt(norm)

        # if there are extra zeros, chop them
        Eval = Eval[(Eval.size - 2 * self.Nion):]
        return Eval, Evect

    def show_crystal(self, pos_vect):
        """
        Makes a pretty plot of the crystal with a given position vector.

        :param pos_vect: The crystal position vector to be seen.
        """
        plt.plot(pos_vect[0:self.Nion], pos_vect[self.Nion:], '.')
        plt.xlabel('x position [um]')
        plt.ylabel('y position [um]')
        plt.axes().set_aspect('equal')

        plt.show()

    def show_crystal_modes(self, pos_vect, Evects, modes):
        """
        For a given crystal, plots the crystal with colors based on the eigenvectors.

        :param pos_vect: the position vector of the current crystal
        :param Evects: the eigenvectors to be shown
        :param modes: the number of modes you wish to see
        """
        plt.figure(1)
        print(np.shape(pos_vect[0:self.Nion]))
        print(np.shape(Evects[:, 0]))
        for i in range(modes):
            plt.subplot(modes, 1, i + 1, aspect='equal')
            plt.scatter(1e6 * pos_vect[0:self.Nion], 1e6 * pos_vect[self.Nion:], 
                        c=Evects[:self.Nion, -i-1], vmin=-.01, vmax=0.01,
                        cmap='viridis')
            plt.xlabel('x position [um]')
            plt.ylabel('y position [um]')
            #plt.axis([-200, 200, -200, 200])
        plt.tight_layout()

    def show_low_freq_mode(self):
        """
        Gets the lowest frequency modes and eigenvectors,
        then plots them, printing the lowest frequency mode.

        """
        num_modes = np.size(self.Evals)
        low_mode_freq = self.Evals[-1]
        low_mode_vect = self.Evects[-1]

        plt.scatter(1e6 * self.u[0:self.Nion], 1e6 * self.u[self.Nion:],
                    c=low_mode_vect, vmin=-.25, vmax=0.25, cmap='RdGy')
        plt.axes().set_aspect('equal')
        plt.xlabel('x position [um]', fontsize=12)
        plt.ylabel('y position [um]', fontsize=12)
        plt.axis([-300, 300, -300, 300])
        print(num_modes)
        print("Lowest frequency mode at {0:0.1f} kHz".format(float(np.real(low_mode_freq))))
        return 0

    def perturb_position(self, pos_vect, strength=.1):
        """
        Slightly displaces each ion by a random proportion (determined by 'strength' parameter)
        and then solves for a new equilibrium position.

        If the new configuration has a lower global potential energy, it is returned.
        If the new configuration has a higher potential energy, it is discarded and
            the previous configuration is returned.
        :param u: The input coordinate vector of each x,y position of each ion.
        :return: Either the previous position vector, or a new position vector.
        """
        # print("U before:", self.pot_energy(u))
        unudge = self.find_eq_pos([coord * abs(np.random.normal(1, strength)) for coord in pos_vect])
        if self.pot_energy(unudge) < self.pot_energy(pos_vect):
            # print("Nudge successful")
            # print("U After:", self.pot_energy(unudge))
            return unudge
        else:
            # print("Nudge failed!")
            return pos_vect

    def show_axial_Evals(self, experimentalunits=False, flatlines=False):
        """
        Plots the axial eigenvalues vs mode number.
        :param experimentalunits:
        :return:
        """
        if self.axialEvals is []:
            print("Warning-- no axial eigenvalues found. Cannot show axial eigenvalues")
            return False

        if flatlines is True:
            fig = plt.figure(figsize=(8, 5))
            fig = plt.axes(frameon=True)
            # fig= plt.axes.get_yaxis().set_visible(False)
            fig.set_yticklabels([])

            if experimentalunits is False:
                fig = plt.xlabel("Eigenfrequency (Units of $\omega_z$)")
                fig = plt.xlim(min(self.axialEvals) * .99, max(self.axialEvals * 1.01))

                for x in self.axialEvals:
                    fig = plt.plot([x, x], [0, 1], color='black', )
            if experimentalunits is True:
                fig = plt.xlabel("Eigenfrequency (2 \pi* Hz)")
                fig = plt.xlim(min(self.axialEvalsE) * .99, max(self.axialEvalsE) * 1.01)
                for x in self.axialEvalsE:
                    fig = plt.plot([x, x], [0, 1], color='black')

            fig = plt.ylim([0, 1])
            # fig= plt.axes.yaxis.set_visible(False)
            fig = plt.title("Axial Eigenvalues for %d Ions, $f_{rot}=$%.1f kHz, and $V_{wall}$= %.1f V " %
                            (self.Nion, self.wrot / (2 * pi * 1e3), self.Cw * self.V0 / 1612))
            plt.show()
            return True

        fig = plt.figure()
        xvals = np.array(range(self.Nion))
        xvals += 1
        if experimentalunits is False:
            fig = plt.plot(xvals, sorted(self.axialEvals), "o")
            fig = plt.ylim((.97 * min(self.axialEvals), 1.01 * max(self.axialEvals)))
            fig = plt.ylabel("Eigenfrequency (Units of $\omega_z$)")
            fig = plt.plot([-1, self.Nion + 1], [1, 1], color="black", linestyle="--")

        else:
            fig = plt.plot(xvals, sorted(self.axialEvalsE), "o")
            fig = plt.ylim(.95 * min(self.axialEvalsE), 1.05 * max(self.axialEvalsE))
            fig = plt.ylabel("Eigenfrequency (Hz)")
            fig = plt.plot([-1, self.Nion + 1], [max(self.axialEvalsE), max(self.axialEvalsE)], color="black",
                           linestyle="--")

        fig = plt.xlabel("Mode Number")

        fig = plt.title("Axial Eigenvalues for %d Ions, $f_{rot}=$%.1f kHz, and $V_{wall}$= %.1f V " %
                        (self.Nion, self.wrot / (2 * pi * 1e3), self.Cw * self.V0 / 1612))
        fig = plt.xlim((0, self.Nion + 1))
        fig = plt.grid(True)
        fig = plt.show()
        return True

    def get_x_and_y(self, pos_vect):
        """
        Hand it a position vector and it will return the x and y vectors
        :param pos_vect:
        :return: [x,y] arrays
        """
        return [pos_vect[:self.Nion], pos_vect[self.Nion:]]

    def is_plane_stable(self):
        """
        Checks to see if any of the axial eigenvalues in the current configuration of the crystal
        are equal to zero. If so, this indicates that the one-plane configuration is unstable
        and a 1-2 plane transistion is possible.

        :return: Boolean: True if no 1-2 plane transistion mode exists, false if it does
        (Answers: "is the plane stable?")
        
        """
        if self.hasrun is False:
            self.run()

        for x in self.axialEvals:
            if x == 0.0:
                return False

        return True

    def rotate_crystal(self, pos_vect, theta, Nion=None):
        """
        Given a position vector defining a crystal, rotates it by the angle theta
        counter-clockwise.

        :param pos_vect: Array of length 2*Nion defining the crystal to be rotated.
        :param theta: Theta defining the angle to rotate the crystal around.
        :param Nion: Number of ions in the crystal (can be optionally defined,
            but will default to the number of ions in the class)
        :return: The returned position vector of the new crystal
        """
        if Nion is None:
            Nion = self.Nion

        x = pos_vect[:Nion]
        y = pos_vect[Nion:]

        xmod = x * np.cos(theta) - y * np.sin(theta)
        ymod = x * np.sin(theta) + y * np.cos(theta)
        newcrys = np.concatenate((xmod, ymod))
        return newcrys

    @staticmethod
    def nan_to_zero(my_array):
        """
        Converts all elements of an array which are np.inf or nan to 0.

        :param my_array: array to be  filtered of infs and nans.
        :return: the array.
        """
        my_array[np.isinf(my_array) | np.isnan(my_array)] = 0
        return my_array

    @staticmethod
    def save_positions(u):
        """
        Takes a position vector and saves it as a text file.
        :param u: position vector to store.
        :return: nothing
        """
        np.savetxt("py_u.csv", u, delimiter=",")

    @staticmethod
    def crystal_spacing_fit(r, offset, curvature):
        """
        """
        return np.sqrt(2 / (np.sqrt(3) * offset * np.sqrt(1 - (r * curvature) ** 2)))

    @staticmethod
    def find_radial_separation(pos_array):
        """
        When given the position array of a crystal,
        returns 4 arrays:
        N radii, N^2 x separations, N^2 y separations, and N^2 radial separations.

        :param pos_array: position array of a crystal.

        :return: radius, x separations, y separations, radial separations
        """
        N = int(pos_array.size / 2)
        x = pos_array[0:N]
        y = pos_array[N:]
        r = np.sqrt(x ** 2 + y ** 2)

        sort_ind = np.argsort(r)

        r = r[sort_ind]
        x = x[sort_ind]
        y = y[sort_ind]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        rsep = np.sqrt(dx ** 2 + dy ** 2)

        return r, dx, dy, rsep

    @staticmethod
    def generate_2D_hex_lattice(shells=1, scale=1):
        """Generate closed shell hexagonal lattice with shells and scale spacing.

        :param scale: scales lattice
        :return: a flattened xy position vector defining the 2d hexagonal lattice.
        """
        posvect = np.array([0.0, 0.0])  # center ion at [0,0]

        for s in range(1, shells + 1):
            posvect = np.append(posvect, ModeAnalysis.add_hex_shell(s))
        posvect *= scale
        return np.hstack((posvect[0::2], posvect[1::2]))

    @staticmethod
    # A slave function used to append shells onto a position vector
    def add_hex_shell(s):
        """
        A method used by generate_2d_hex_lattice to add the s-th hex shell to the 2d lattice.
        Generates the sth shell.
        :param s: the sth shell to be added to the lattice.

        :return: the position vector defining the ions in sth shell.
        """
        a = list(range(s, -s - 1, -1))
        a.extend(-s * np.ones(s - 1))
        a.extend(list(range(-s, s + 1)))
        a.extend(s * np.ones(s - 1))

        b = list(range(0, s + 1))
        b.extend(s * np.ones(s - 1))
        b.extend(list(range(s, -s - 1, -1)))
        b.extend(-s * np.ones(s - 1))
        b.extend(list(range(-s, 0)))

        x = np.sqrt(3) / 2.0 * np.array(b)
        y = 0.5 * np.array(b) + np.array(a)
        pair = np.column_stack((x, y)).flatten()
        return pair

########################################################################################

if __name__ == "__main__":
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestCalculationConsistency)
    # unittest.TextTestRunner(verbosity=1).run(suite)

    # NOTE: class now takes number of ions instead of shells
    # For reference the following ion number correspond the closed shells:
    # 1  2  3  4  5   6   7   8   9  10  11  12  13  14
    # 7 19 37 61 91 127 169 217 271 331 397 469 547 631...

    #a = ModeAnalysis(N=19, Vwall=35, frot=45)  # oldtrap
    a = ModeAnalysis(N=37, Vwall=1, frot=180)
    a.run()
    #D,E = a.calc_planar_modes(a.u)

