'''

File: hopfield_gamma.py
Author: Hadayat Seddiqi
Date: 5.10.14
Description: Do a [deterministic] Hopfield run while taking in
             the bias term, Gamma, as a cmd arg. allowing us to
             study the effects on the eigenspectrum and success
             probabilities.

'''

import os
import scipy as sp
import itertools
import random
import collections

def parameters(cmdargs):
    """
    """

    # The Hopfield parameters
    hparams = {
        'numNeurons': cmdargs['qubits'],
        'inputState': [ 2*sp.random.random_integers(0,1)-1 
                        for k in xrange(cmdargs['qubits']) ],
        'learningRule': cmdargs['simtype'],
        'bias': float(cmdargs['farg']),
        'numMemories': sp.random.random_integers(1,cmdargs['qubits'])
        }

    # Construct memories
    memories = [ [ 2*sp.random.random_integers(0,1)-1 
                   for k in xrange(hparams['numNeurons']) ]
                 for j in xrange(hparams['numMemories']) ]

    # At least one pattern must be one Hamming unit away from the input
    memories[0] = list(hparams['inputState'])
    memories[0][sp.random.random_integers(0,hparams['numNeurons']-1)] *= -1

    # Make sure all other patterns have Hamming distance > 1
    def hamdist(a,b):
        """ Calculate Hamming distance. """
        return sp.sum(abs(sp.array(a)-sp.array(b))/2.0)
    # Loop over additional memories, if there are any
    for imem, mem in enumerate(memories[1:]):
        while hamdist(mem, hparams['inputState']) < 2.0:
            # Flip a random spin
            rndbit = sp.random.random_integers(0,hparams['numNeurons']-1)
            memories[imem+1][rndbit] *= -1

    # Basic simulation params
    nQubits = hparams['numNeurons']
    T = 100.0
    dt = 0.01*T

    # Output parameters
    binary = 1 # Save output files as binary Numpy format
    progressout = 0 # Output simulation progress over anneal timesteps

    eigspecdat = 1 # Output data for eigspec
    eigspecplot = 0 # Plot eigspec
    eigspecnum = 2**(nQubits-1) # Number of eigenvalues to output
    fidelplot = 0 # Plot fidelity
    fideldat = 0 # Output fidelity data
    fidelnumstates = 2**nQubits # Check fidelity with this number of eigenstates
    overlapdat = 0 # Output overlap data
    overlapplot = 0 # Plot overlap
    solveMethod = 'ExpPert' # 'ExpPert', 'SuzTrot', 'ForRuth', 'BCM'

    # Output directory stuff
    probdir = 'data/hopfield_gamma_random/'+hparams['learningRule']
    if isinstance(T, collections.Iterable):
        probdir += 'MultiT'
    if os.path.isdir(probdir):
        outlist = sorted([ int(name) for name in os.listdir(probdir) 
                           if name.isdigit() ])
    else:
        outlist = []
    outnum = outlist[-1] + 1 if outlist else 0
    outputdir = probdir + '/' + str(outnum) + '/'

    probshow = 0 # Print final state probabilities to screen
    probout = 1 # Output probabilities to file
    mingap = 1 # Record the minimum spectral gap

    errchk = 0 # Error-checking on/off (for simulation accuracy)
    eps = 0.01 # Numerical error in normalization condition (1 - norm < eps)

    # Specify a QUBO (convert to Ising = True), or alpha, beta directly 
    # (convert = False), and also specify the signs on the Ising Hamiltonian 
    # terms (you can specify coefficients too for some problems if needed)
    isingConvert = 0
    isingSigns = {'hx': -1, 'hz': -1, 'hzz': -1}

    # Construct network Ising parameters
    neurons = nQubits

    # This is gamma, the appropriate weighting on the input vector
    # isingSigns['hz'] *= 1 - (len(hparams['inputState']) - 
    #                          hparams['inputState'].count(0))/(2*neurons)
    isingSigns['hz'] *= hparams['bias']

    alpha = sp.array(hparams['inputState'])
    beta = sp.zeros((neurons,neurons))
    delta = sp.array([])

    # Construct the memory matrix according to a learning rule
    if hparams['learningRule'] == 'hebb':
        # Construct pattern matrix according to Hebb's rule
        for i in range(neurons):
            for j in range(neurons):
                for p in range(len(memories)):
                    beta[i,j] += ( memories[p][i]*memories[p][j] -
                                   len(memories)*(i == j) )
        beta = sp.triu(beta)/float(neurons)
    elif hparams['learningRule'] == 'stork':
        # Construct the memory matrix according to the Storkey learning rule
        memMat = sp.zeros((neurons,neurons))
        for m, mem in enumerate(memories):
            for i in range(neurons):
                for j in range(neurons):
                    hij = sp.sum([ memMat[i,k]*mem[k] for k in range(neurons) ])
                    hji = sp.sum([ memMat[j,k]*mem[k] for k in range(neurons) ])
                    # Don't forget to make the normalization a float!
                    memMat[i,j] += 1./neurons*(mem[i]*mem[j] - mem[i]*hji - 
                                               hij*mem[j])
        beta = sp.triu(memMat)
    elif hparams['learningRule'] == 'proj':
        # Construct memory matrix according to the Moore-Penrose pseudoinverse rule
        memMat = sp.matrix(memories).T
        beta = sp.triu(memMat * sp.linalg.pinv(memMat))

    # Some outputs
    outputs = {
        'nQubits': nQubits,
        'learningRule': hparams['learningRule'],
        'outdir': probdir,
        'inputState': hparams['inputState'],
        'memories': memories,
        'answer': memories[0],
        'bias': hparams['bias'],
        'annealTime': list(T) if isinstance(T, collections.Iterable) else T
               }

    ############################################################################
    ######## All variables must be specified here, do NOT change the keys ######
    ############################################################################

    return {
        'nQubits': nQubits,
        'Q': None,
        'T': T,
        'dt': dt,
        'outputdir': outputdir,
        'errchk': errchk,
        'eps': eps,
        'isingConvert': isingConvert,
        'isingSigns': isingSigns,
        'outputs': outputs,
        'alpha': alpha,
        'beta': beta,
        'delta': delta,
        'eigdat': eigspecdat,
        'eigplot': eigspecplot,
        'eignum': eigspecnum,
        'fiddat': fideldat,
        'fidplot': fidelplot,
        'fidnumstates': fidelnumstates,
        'overlapdat': overlapdat,
        'overlapplot': overlapplot,
        'outdir': outputdir,
        'binary': binary,
        'progressout': progressout,
        'probshow': probshow,
        'probout': probout,
        'mingap': mingap,
        'stateoverlap': None,
        'solveMethod': solveMethod
        }
