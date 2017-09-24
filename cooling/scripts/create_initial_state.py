import numpy as np
import mode_analysis_code
import coldatoms
import ion_trapping


# Configuration
num_ions = 127
frot = 180.0e3
v_wall = 1.0


# Now compute the steady state configuration of the ions for certain trap
# voltages and rotating wall potential
mode_analysis = mode_analysis_code.ModeAnalysis(N=num_ions,
                                                Vtrap=(0.0, -1750.0, -1970.0),
                                                Vwall=v_wall,
                                                frot=frot)
mode_analysis.run()


# A few additional parameters
m_beryllium = mode_analysis.m_Be
q_beryllium = mode_analysis.q


# Now create the ensemble
ensemble = ion_trapping.create_ensemble(mode_analysis.uE,
                                        2.0 * np.pi * frot,
                                        m_beryllium,
                                        q_beryllium)


# And safe it to disk.
f = open('initial_state_180kHz.txt', 'w')
f.write(coldatoms.ensemble_to_json(ensemble))
f.close()
