import numpy as np


def read_data(basename, i, j):
    filename = basename + '_' + str(i) + '_' + str(j) + '.dat'
    return np.loadtxt(filename, dtype=np.float64, skiprows=2)

for j in range(9):
    data = read_data('heating_run', 0, j)
    print(np.mean(data[50:, 2]))

print('\n\n')

for j in range(0, 20):
    data = read_data('heating_run', 1, j)
    print(np.mean(data[50:, 2]))

print('\n\n')

for j in range(0, 20):
    data = read_data('heating_run', 2, j)
    print(np.mean(data[50:, 2]))

