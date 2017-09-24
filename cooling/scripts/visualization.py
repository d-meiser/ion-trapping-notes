import numpy as np
import matplotlib.pyplot as plt


# Enable LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {
    'text.latex.preamble':
    [r'\usepackage{siunitx}', r'\usepackage{amsmath}'],
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
    'axes.labelsize': 'medium'
}
plt.rcParams.update(params)


default_limit = 1.0e6 * 1.2e-4
line_width = 246.0 / 72.0
default_width = 0.95 * line_width 
golden_ratio = 1.61803398875
default_height = default_width


def top_view(x, y, style='o',
             xlim=[-default_limit, default_limit],
             ylim=[-default_limit, default_limit]):
    plt.plot(1.0e6 * x, 1.0e6 * y, style, ms=3)
    plt.xlabel(r'$x / \si{\um}$')
    plt.ylabel(r'$y / \si{\um}$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gcf().set_size_inches([default_width, default_height])
    plt.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.15)
    plt.axes().set_aspect('equal')
