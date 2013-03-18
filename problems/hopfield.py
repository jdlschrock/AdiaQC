'''

File: hopfield.py
Author: Hadayat Seddiqi
Date: 3.15.13
Description: Parameters for a Hopfield neural network.

'''
# Ugly import hack to support Python2 and 3
import sys
if sys.version_info[0] > 2: import problems.problems as problems
else: import problems

#
# Output parameters
#
eigspecflag = 1 # Plot eigenspectrum (or not)
outputdir = 'data/'

#
# Simulation parameters
#
T = 100 # Total time
dt = 0.1 # Timestep
neurons = 4 # qubits
memories = [ [1,0,0,1] ]
inputstate = [1,0,1,1]

# Generate the QUBO
nQubits, Q, a = problems.HopfieldNetwork(neurons, memories, inputstate)