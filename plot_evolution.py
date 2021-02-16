import matplotlib.pyplot as plt
import matplotlib.lines
import numpy as np
import scipy.signal
import scipy.optimize
import mode_analysis_code
import argparse

plt.style.use('ggplot')
mode_analysis = mode_analysis_code.ModeAnalysis()
nu_z = np.sqrt(2*mode_analysis.q*mode_analysis.Coeff[2]/mode_analysis.m_Be)/(2*np.pi)


def lorentzian(x, A, FWHM, x0):#(x, height, FWHM, x0):
    p = (x0 - x)/(FWHM/2)
    L = A/(1+p**2)
#    A = np.pi * FWHM * height / 2
#    L = (A / np.pi) * (0.5 * FWHM / ((x - x0)**2 + (0.5 * FWHM)**2))
    return L

def double_lorentzian(x, A, FWHM_A, x0_A, B, FWHM_B, x0_B, C):
    pA = (x0_A - x)/(FWHM_A/2)
    pB = (x0_B - x)/(FWHM_B/2)
    L = C * (A/(1 + pA**2) + B/(1 + pB**2))
    return L

def report_values(value, error):
    a = int(np.floor(np.log10(np.abs(error))))
    report_Err = round(error, -a)
    report_Val = round(value, -a)
    return report_Val, report_Err


def plot_evolution(data_dir, get_Params=True, bParams=None, b_sigma=None):# pParams=None, sParams=None, bParams=None):
    """
    PARAMETERS
    data_dir: the directory path that contains the 'freqs.npy' and 'PSD_data.npy' files.  Assumes the final folder is of the format '/modes**/' where, the first number is the primary mode and the secondary is the seeded mode
    get_Params: If True (default), the function has the user input bounds for use in fitting the various peaks to Lorentzians.  If False, the user must provide the lorentzian parameters for the primary and seeded peaks in pParams and sParams

    RETURNS
    Nothing.  Produces a plot of the Power Spectral Density

    TODO
    Put something in here to choose peaks and limits for each peak for curve fitting
    """

    ax_mode_dir = '/'.join(data_dir.split('/')[:-5]) + '/'
    axialEvalsE = np.load(ax_mode_dir + 'axialEvalsE.npy') / (2* np.pi)

    x = np.load(data_dir + 'freqs.npy')
    y = np.load(data_dir + 'PSD_data.npy')

    mode_list = data_dir.split('/')
    while True:
        if mode_list[-1] == '':
            mode_list.remove('')
        else:
            mode_str = mode_list[-1]
            break

    pMode = int(mode_str[-2])
    sMode = int(mode_str[-1])

    exps = np.log10(y)
    trap_point = 10**(.75*exps.max() + .25*exps.min())

    fig, ax = plt.subplots(figsize = (30,20))
    ax.semilogy(x / 1.0e6, y, c = 'blue', linewidth = 2.0)

    for i in np.arange(1,11):
        ax.axvline(x = axialEvalsE[-i]/1.0e6, c = 'sienna', linewidth=0.5)

    ax.axvline(x = axialEvalsE[-pMode] / 1.e6, c = 'red', label = 'Primary Mode')
    ax.axvline(x = axialEvalsE[-sMode] / 1.e6, c = 'green', label = 'Seeded Mode')
    ax.axvline(x = nu_z / 1.e6, c = 'purple', label = 'Axial Trap Frequency')

    ax.set_xlim(1.56, 1.62)
    ax.set_xlabel(r'$\nu / \rm{MHz}$')
    ax.set_ylabel(r'PSD($z$)')

    if get_Params:
        ax.legend(loc='best')
        ax.set_title('Select Primary left then right peak bounds, then Seeded left then right')
        pLeft, pRight = fig.ginput(2)
        sLeft, sRight = fig.ginput(2)

        
        return [pLeft[0], pRight[0]], [sLeft[0], sRight[0]]

    else:
#        ax.semilogy(x / 1.e6, lorentzian(x/1.e6, *pParams), label = 'Primary Peak:\n\tFWHM={} MHz\n\tx0={} MHz'.format(pParams[1], pParams[2]))

#        ax.semilogy(x / 1.e6, lorentzian(x/1.e6, *sParams), label = 'Seeded Peak:\n\tFWHM={} MHz\n\tx0={} MHz'.format(sParams[1], sParams[2]))

#        ax.semilogy(x / 1.e6, lorentzian(x/1.e6, *pParams) + lorentzian(x/1.e6, *sParams), label = 'Seeded + Primary Peaks')

        #
        pFWHMerr = b_sigma[1]*1.e6
        pNuerr = b_sigma[2]
        sFWHMerr = b_sigma[4]*1.e6
        sNuerr = b_sigma[5]
        

        pFWHM = bParams[1]*1.e6
        sFWHM = bParams[4]*1.e6

#        print(bParams)
#        print(b_sigma)

        rep_pFWHM, rep_perrFWHM = report_values(pFWHM, pFWHMerr)
        rep_pNu, rep_perrNu = report_values(bParams[2], pNuerr)
        rep_sFWHM, rep_serrFWHM = report_values(sFWHM, sFWHMerr)
        rep_sNu, rep_serrNu = report_values(bParams[5], sNuerr)

#        print("pFWHM: {} $\\pm$ {}".format(rep_pFWHM, rep_perrFWHM))
#        print("pNu: {} $\\pm$ {}".format(rep_pNu, rep_perrNu))
#        print("sFWHM: {} $\\pm$ {}".format(rep_sFWHM, rep_serrFWHM))
#        print("sNu: {} $\\pm$ {}".format(rep_sNu, rep_serrNu))


        ax.semilogy(x / 1.e6, double_lorentzian(x/1.e6, *bParams), label = 'PRIMARY PEAK:\n  FWHM={}$\\pm${} Hz\n  $\\nu_0$={}$\\pm${} MHz\nSEEDED PEAK:\n  FWHM={}$\\pm${} Hz\n  $\\nu_0$={}$\\pm${} MHz'.format(rep_pFWHM, rep_perrFWHM, rep_pNu, rep_perrNu, rep_sFWHM, rep_serrFWHM, rep_sNu, rep_serrNu)) # Test Dual Fit


        ax.legend(loc='best')
        ax.set_title('Seeded: Mode {}\nPrimary: Mode {}'.format(sMode, pMode))

        plt.savefig(data_dir + 'fitted_modes{}{}.png'.format(pMode, sMode), bbox_inches='tight')

        plt.show(block=True)


def fit_Lorentzian(data_dir, primary_bounds, seeded_bounds):

    x = np.load(data_dir + 'freqs.npy')
    y = np.load(data_dir + 'PSD_data.npy')

    peaks, _ = scipy.signal.find_peaks(y, distance = 5)

    pLboundInd = np.argmin(np.absolute(x/1.e6 - primary_bounds[0]))
    pRboundInd = np.argmin(np.absolute(x/1.e6 - primary_bounds[1]))
    pPeakInd = np.argmax(y[pLboundInd:pRboundInd])

    sLboundInd = np.argmin(np.absolute(x/1.e6 - seeded_bounds[0]))
    sRboundInd = np.argmin(np.absolute(x/1.e6 - seeded_bounds[1]))
    sPeakInd = np.argmax(y[sLboundInd:sRboundInd])


#    tP = np.linspace(x[pLboundInd],x[pRboundInd],1000)/1e6
#    tS = np.linspace(x[sLboundInd],x[sRboundInd],1000)/1e6

#    pBounds = ([0., 0., 0.],[np.inf, np.inf, np.inf])
#    sBounds = ([0., 0., 0.],[np.inf, np.inf, np.inf])

    #
    bothBounds = (0., np.inf) # Test Dual Fit


#    p0P = [y[pLboundInd:pRboundInd][pPeakInd], 10., x[pLboundInd:pRboundInd][pPeakInd]]
#    p0S = [y[sLboundInd:sRboundInd][sPeakInd], 10., x[sLboundInd:sRboundInd][sPeakInd]]
    
    #
    p0Both = [y[pLboundInd:pRboundInd][pPeakInd], 2., x[pLboundInd:pRboundInd][pPeakInd], y[sLboundInd:sRboundInd][sPeakInd], 2., x[sLboundInd:sRboundInd][sPeakInd], 23.] # Test Dual Fit


    
#    poptP, pcovP = scipy.optimize.curve_fit(lorentzian, x[pLboundInd:pRboundInd], y[pLboundInd:pRboundInd], p0 = p0P, bounds = bothBounds)

#    poptS, pcovS = scipy.optimize.curve_fit(lorentzian, x[sLboundInd:sRboundInd], y[sLboundInd:sRboundInd], p0 = p0S, bounds = bothBounds)
    
    #
    poptBoth, pcovBoth = scipy.optimize.curve_fit(double_lorentzian, x,y, p0 = p0Both, bounds = bothBounds) # Test Dual Fit


#    poptP = np.array(poptP)
#    poptS = np.array(poptS)

    #
    poptBoth = np.array(poptBoth) # Test Dual Fit
    p_sigma = np.sqrt(np.diag(pcovBoth))

#    poptP[1:] *= 1.e-6
#    poptS[1:] *= 1.e-6

    #
    poptBoth[1:3] *= 1.e-6 # Test Dual Fit
    poptBoth[4:6] *= 1.e-6 # Test Dual Fit
    p_sigma[1:3] *= 1.e-6
    p_sigma[4:6] *= 1.e-6


#    plt.plot(tP, lorentzian(tP, *poptP), c='purple')
#    plt.plot(tS, lorentzian(tS, *poptS), c='green')
#    plt.show(block=True)

    #
    return [poptBoth, p_sigma]#[poptP, pcovP], [poptS, pcovS], [poptBoth, pcovBoth] # Test Dual Fit



parser = argparse.ArgumentParser()
parser.add_argument('directory', help="Provide the directory containing 'freq.npy' and 'PSD_data.npy' for the desired modes.")

args = parser.parse_args()


primaryBounds, seededBounds = plot_evolution(args.directory)
bothFit = fit_Lorentzian(args.directory, primaryBounds, seededBounds)
#primaryFit, seededFit, bothFit = fit_Lorentzian(args.directory, primaryBounds, seededBounds)
plot_evolution(args.directory, get_Params=False, bParams=bothFit[0], b_sigma = bothFit[1])
#plot_evolution(args.directory, get_Params=False, pParams=primaryFit[0], sParams=seededFit[0], bParams=bothFit[0])
