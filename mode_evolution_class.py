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

class modeEvolution:
    """
    provides an object capable of simulating several excited modes
    Pulls from the ModeAnalysis class
    """

    q = 1.602176565E-19
    amu = 1.66057e-27
    m_Be = 9.012182 * amu
    k_e = 8.9875517873681764E9 # electrostatic constant k_e = 1 / (4.0 pi epsilon_0)

    def __init__(self, N=127, Vtrap=(0., -1750., -2000.), Vwall=5., frot=180.,
                 B=4.4588):
        """
        Assumes many of the following parameters, each set as default in Freericks/
        Meiser codes.  Can be adjusted in later methods.
        """

        self.Nion = N
        self.Vtrap = Vtrap
        self.Vwall = Vwall
        self.frot = frot
        self.mode_analysis = mode_analysis_code.ModeAnalysis(N=self.Nion,
                                                             Vtrap=self.Vtrap,
                                                             Vwall=self.Vwall,
                                                             frot=self.frot)

    def run(self):
        """

        """

        self.mode_analysis.run()
        
