#### TO DO ####
# Include the axial eigenvalues in get_peaks somehow.  It would be very useful to see which peak is primary and which is seeded
# It seems that when NOT using bounds in curve_fit, we are getting an error "Optimal parameters not found: Number of calls to function has reached maxfev=1600."  wheras specifying bounds results in "Optimal parameters not found: The maximum number of function evaluations is exceeded."  It looks like the same issue is occuring here, but different optimization methods produce different error conditions
# Since it seems I am not converging to a solution with either case, I will try playing around in the no bounds case.

# By end of night 8/10, I seem to have gotten parameters guess close enough for the function to work properly.  Current parameters are [y[pPeakInd], 10., x[pPeakInd], y[sPeakInd], 10., x[sPeakInd], 10.] for primary mode 4, seeded mode 2.
# However, the final plot is not pretty.  It is difficult to distinguish the 'data' from the fitted curve.  CLEAN UP PRESENTATION OF FITTED CURVE.
# Also, how do I get the guesses for the fits to be close enough?  The overall multiplication factor seems to have a wide degree of variance.  Does it depend on the baseline value for the data?  The individual peak multiplication factors are coupled with the FWHM for the given peak.  Is there a way to get this from the user more effectively?
# My time might be better spent going through each individual mode coupling scheme and choosing appropriate parameters.  I could try to write a bash script to run through all of them, and write to a file which datasets do not converge to a solution.
# However, I believe I should spend the rest of the time writing a bash script to create slurm scripts for the supercomputer to run.  Once I have a set of files to run, I should then create a bash script to submit each job to slurm.
# Once this is complete, I can let the cluster get me good data while I play around with how to best fit a generic set of data.

import matplotlib.pyplot as plt
import matplotlib.lines
import numpy as np
import scipy.signal
import scipy.optimize
import mode_analysis_code
import argparse

plt.style.use('ggplot')
plt.rcParams.update({'font.size':18})
plt.ion()

mode_analysis = mode_analysis_code.ModeAnalysis()
nu_z = np.sqrt(2 * mode_analysis.q * mode_analysis.Coeff[2] / mode_analysis.m_Be) / (2 * np.pi)


def lorentzian(x, *params):
    A, FWHM_A, x0 = params
    p = (x0 - x)/(FWHM/2)
    L = A / (1 + p**2)
    return L


def double_lorentzian(x, *params):
    A, FWHM_A, x0_A, B, FWHM_B, x0_B = params
    pA = (x0_A - x) / (FWHM_A / 2)
    pB = (x0_B - x) / (FWHM_B / 2)
    L = A / (1 + pA**2) + B / (1 + pB**2)
    return L


# def double_lorentzian(x, *params):
#     A, FWHM_A, x0_A, B, FWHM_B, x0_B, C = params
#     pA = (x0_A - x) / (FWHM_A / 2)
#     pB = (x0_B - x) / (FWHM_B / 2)
#     L = C * (A / (1 + pA**2) + B / (1 + pB**2))
#     return L

def get_peaks(x, y):
    """
    Has the user specify the bounds for the primary and seeded peaks.  The maximum value is then determined to be the estimated x0 and 0.5*height.
    Next, the user provides position estimates of the FWHM left and right limits.
    This function then shows a plot of the estimated peak and a FWHM bar for the primary and seeded peaks.  If satisfactory, the program returns these values.  If not, the program start over.
    """
    
    # Plot the data needed for getting user-defined bounds
    fig, ax = plt.subplots(figsize = (27,18))
    ax.semilogy( x / 1.e6, y, c = 'blue')

#    ax.axvline(x = axialEvalsE[-pMode] / 1.e6, c = 'red', label = 'Primary: Mode {}'.format(pMode))
#    ax.axvline(x = axialEvalsE[-sMode] / 1.e6, c = 'green', label = 'Seeded: Mode {}'.format(sMode))

    ax.set_xlim(1.56, 1.62)
    ax.set_xlabel(r'$\nu / \rm{MHz}$')
    ax.set_ylabel(r'PSD($z$)')
    ax.legend(loc='best')

    # Get bounds for primary peak

    ax.set_title('Select base for primary peak')
    pLeft, pRight = fig.ginput(2)

    pLboundInd = np.argmin(np.absolute(x / 1.e6 - pLeft[0]))
    pRboundInd = np.argmin(np.absolute(x / 1.e6 - pRight[0]))
    pPeakInd = np.argmax(y[pLboundInd:pRboundInd]) + pLboundInd


    # Get bounds for seeded peak

    ax.set_title('Select base for seeded peak')
    sLeft, sRight = fig.ginput(2)

    sLboundInd = np.argmin(np.absolute(x / 1.e6 - sLeft[0]))
    sRboundInd = np.argmin(np.absolute(x / 1.e6 - sRight[0]))
    sPeakInd = np.argmax(y[sLboundInd:sRboundInd]) + sLboundInd

    plt.close()

    pFWHM = float(input("Enter estimated Primary FWHM in Hz: "))
    sFWHM = float(input("Enter estimated Seeded FWHM in Hz: "))
    C = float(input("Enter estimated overall multiplicative factor: "))

    return pPeakInd, pFWHM, sPeakInd, sFWHM, C, [pLeft[0], pRight[0]], [sLeft[0], sRight[0]]

        
def fit_double_lorentzian(data_dir, axial_Evals_fpath, bounds=False):
    """
    PARAMETERS
    :data_dir: The directory path containing 'freqs.npy' and 'PSD_data.npy' files.  Assumes the final folder is of the format '/modes**' where the first * is the primary mode and the second * is the seeded mode.
    :axial_Evals_fpath: The file containing axialEvalsE.npy, obtained from using the .run() method in mode_analysis_code.py


    RETURNS
    :fit_params: a list of the *7* parameters resulting in the best fit of the double lorentzian to the data

    The function asks the user to click on the plot several times to provide estimates for the fitting parameters.  After performing the fit, asks the user if this is satisfactory.  If not, it allows for inserting/removing bounds, or fitting each peak individually.

    This function starts by trimming the data down to only look at frequencies in the range 1.56 MHz < f < 1.62 MHz.  This is done in an attempt to ensure fits on axial peak modes 1-10ish  This can be adjusted in the first few lines of code.

    POSSIBLE ADDITION:  All parameters should be positive, so we might put in bounds = (0., np.inf).  However, according to scipy documentation, this will prevent the use of the Levenberg-Marquardt algorithm.
    """

    # Specify region of interest
    f_min = 1.56
    f_max = 1.62

    # Load in the relevant data
    x = np.load(data_dir + '/freqs.npy')
    y = np.load(data_dir + '/PSD_data.npy')
    axialEvalsE = np.load(axial_Evals_fpath) / (2 * np.pi)

    #Trim down the relevant data
    lFbound = np.argmin(np.absolute(x / 1.e6 - f_min))
    rFbound = np.argmin(np.absolute(x / 1.e6 - f_max))
    xTrim = x[lFbound:rFbound]
    yTrim = y[lFbound:rFbound]

    # Parse the directory to determine the primary and seeded modes
    mode_list = data_dir.split('/')
    while True:
        if mode_list[-1]=='':
            mode_list.remove('')
        else:
            mode_str = mode_list[-1]
            break

    pMode = int(mode_str[-2])
    sMode = int(mode_str[-1])

    
    # Get User to define initial guesses
    pPeakInd, pFWHM, sPeakInd, sFWHM, C, pBounds, sBounds= get_peaks(xTrim, yTrim)

#    initial_guess = [yTrim[pPeakInd], pFWHM, xTrim[pPeakInd], yTrim[sPeakInd], sFWHM, xTrim[sPeakInd], C]

    initial_guess = [yTrim[pPeakInd], pFWHM, xTrim[pPeakInd], yTrim[sPeakInd], sFWHM, xTrim[sPeakInd]]

    vals = ['A Height: ', 'A FWHM: ', 'A v0: ', 'B Height: ', 'B FWHM: ', 'B v0: ', 'C: ']

    print("\nUser Guess Values:")
    for i in np.arange(len(initial_guess)):
        print('{} {}'.format(vals[i], initial_guess[i]))

    t = np.linspace(xTrim.min(), xTrim.max(), 1000)
    yt = double_lorentzian(t, initial_guess[0], initial_guess[1], initial_guess[2], initial_guess[3], initial_guess[4], initial_guess[5])
#    yt = double_lorentzian(t, initial_guess[0], initial_guess[1], initial_guess[2], initial_guess[3], initial_guess[4], initial_guess[5], initial_guess[6])

    plt.figure(figsize=(27,18))
    plt.title("User Guess")
    plt.semilogy(xTrim,yTrim, label='Data')
    plt.semilogy(t,yt, label='Guess')
    plt.show(block=True)

    init_cond = input("Is guess satisfactory ((y)es/(n)o/(q)uit): ")

    if ((init_cond=='q') or (init_cond=='Q')):
        return None

    elif ((init_cond=='n') or (init_cond=='N')):
        fit_double_lorentzian(data_dir, axial_Evals_fpath, bounds)

    # Determine whether to use bounds
    if bounds:
        parameter_bounds = (0., np.inf)
#        parameter_bounds = ([0., 0., pBounds[0], 0., 0., sBounds[0], 0.], [np.inf, np.inf, pBounds[1], np.inf, np.inf, sBounds[1], np.inf])
    else:
        parameter_bounds = None

    print(parameter_bounds)

    # curve_fit to get optimal parameters
    if parameter_bounds==None:
        dLorentzParams, dLorentzCov = scipy.optimize.curve_fit(double_lorentzian, xTrim, yTrim, p0=initial_guess)
    else:
        dLorentzParams, dLorentzCov = scipy.optimize.curve_fit(double_lorentzian, xTrim, yTrim, p0=initial_guess, bounds=parameter_bounds)

    # Convert covariance matrix to std dev values
    dLorentzSigma = np.sqrt(np.diag(dLorentzCov))

    ### Modification Follows ###

    print('\nFitted Values:')
    for i in np.arange(len(dLorentzParams)):
        print('{} {}'.format(vals[i], dLorentzParams[i]))
    print('\n')

    ############################

    # Convert x-values from Hz to MHz (Ease of use in plotting)
    dLorentzParams = np.array(dLorentzParams)
    dLorentzParams[1] *= 1.e-6 # FWHM_A
    dLorentzParams[2] *= 1.e-6 # x0_A
    dLorentzParams[4] *= 1.e-6 # FWHM_B
    dLorentzParams[5] *= 1.e-6 # x0_B

    # Plot Data next to fitted Curve
    # Create data series for fitted peaks
    x_fit = np.linspace(xTrim.min(), xTrim.max(), 1000)
    y_fit = double_lorentzian(x_fit / 1.e6, dLorentzParams[0],  dLorentzParams[1], dLorentzParams[2], dLorentzParams[3],  dLorentzParams[4], dLorentzParams[5])
#    y_fit = double_lorentzian(x_fit / 1.e6, dLorentzParams[0],  dLorentzParams[1], dLorentzParams[2], dLorentzParams[3],  dLorentzParams[4], dLorentzParams[5], dLorentzParams[6])

    fig, ax = plt.subplots(figsize=(27,18))
    ax.semilogy(xTrim / 1.e6, yTrim, label = 'PSD') # PSD 'data'
    ax.semilogy(x_fit / 1.e6, y_fit, label = 'Fitted Peaks\nPrimary $\\nu\_0$={:.3} MHz\nSeeded $\\nu\_0$={:.3} MHz\nPrimary FWHM={:.3f} Hz\nSeeded FWHM={:.3f} Hz'.format(dLorentzParams[2], dLorentzParams[5], dLorentzParams[1]*1.e6, dLorentzParams[4]*1.e6)) # Fitted curve

    # Plot a background of Harmonic Approximation eigenfrequencies
    for i in np.arange(1,11):
        ax.axvline(x=axialEvalsE[-i] / 1.e6, c='sienna', linewidth=0.5)

    # Plot heavier lines for the primary and seeded frequencies
    ax.axvline(x=axialEvalsE[-pMode] / 1.e6, label='Primary Mode')
    ax.axvline(x=axialEvalsE[-sMode] / 1.e6, label='Seeded Mode')
    ax.axvline(x=nu_z / 1.e6, label='Axial Trap Frequency')
    
    # Set labels, title, etc.
    ax.set_xlim(1.56, 1.62)
    ax.set_xlabel(r'$\nu / \rm{MHz}$')
    ax.set_ylabel(r'PSD($z$)')
    ax.legend(loc='best')
    ax.set_title('Primary: Mode {}\nSeeded: Mode {}'.format(pMode, sMode))

    # When using terminal, prevent the matplotlib window from closing
    plt.show(block=True)

    # Make sure user is satisfied with fit.
    final_word = input("Is fit satisfactory ((y)es/(n)o/(q)uit): ")
    if ((final_word=='q') or (final_word=='Q')):
        return None
    elif ((final_word=='n') or (final_word=='N')): # If not, start over
        print("Previous Fitted Values:\n")
        for i in np.arange(len(dLorentzParams)):
            print("{} {}".format(vals[i], dLorentzParams[i]))

        fit_double_lorentzian(data_dir, axial_Evals_fpath, parameter_bounds)

    else: # Otherwise, save the plot and fitted values to the same directory
        plt.savefig(data_dir + '/fitted_modes{}{}.png'.format(pMode, sMode), bbox_inches='tight')
        np.save(data_dir + '/fit_values{}{}.npy'.format(pMode, sMode), np.array(dLorentzParams))

parser = argparse.ArgumentParser()
parser.add_argument('data_directory', help="Provide the directory containing 'freq.npy' and 'PSD_data.npy' for the desired modes. Should not have a '/' as the final character")
parser.add_argument('evals_fpath', help="Provide the file path to 'axialEvalsE.npy'")

args = parser.parse_args()

fit_double_lorentzian(args.data_directory, args.evals_fpath, True)

# It seems some of the problem lies in the fact that we are fitting a sum of lorentzians.  The maximum values should occur at the individual v0, but sometimes, we are getting anomolous shifted peaks.
# For instance, looking at dataset modes41, guessing the peaks as A(B) and A v0 (B v0) but with A FWHM = 10 Hz, B FWHM = 15 Hz, and C = 50, we get a visually very similar function shape.  However, the optimization swerves away from here to put in peaks shifted right from the real peaks.
# I think the problem may be that the height of each peak is dependent on the height of the other peak, the FWHM of the other peak, AND the frequency spacing of the two peaks in question.
# Maybe I should get rid of the C factor, and just sum up two lorentzians
# Maybe I should do the derivative of the sum lorentzians to see where we expect to find peaks.
# Maybe I should make curve fitting a two step process:  First, curve fit a single lorentzian to get the v0 for each peak, and then input this into the summed (double lorentzian) function, but with these determined v0 as input parameters.  Then, curve fit the double lorentzian as now, but eliminate the v0 (x0) parameters in the parameter space to search for.


# Can we make sure the data used in curve fitting only has the two peaks?  Is this what is throwing off the curve fitting?
