import visualization
import coldatoms
import matplotlib.pyplot as plt


ensemble = coldatoms.json_to_ensemble(open('initial_state_180kHz.txt').read())
plt.clf()
visualization.top_view(ensemble.x[:, 0], ensemble.x[:, 1])
plt.savefig('initial_state_top_view.pdf')

