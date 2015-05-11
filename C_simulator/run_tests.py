import subprocess as sp
import random as r
import numpy as np
import sys

def build_config( comments, nq, lrgs, res, t, dt, alpha, beta, delta ):    
    output = ''
    #output += '/*\n'
    #output +i= comments + '\n'
    #output += '*/\n'
    output += '{' 
    output += '"scalars" : {'
    output += '"nq" : ' + str(nq) + ',' 
    output += '"lrgs" : ' + str(lrgs) + ',' 
    output += '"res" : ' + str(res) + ',' 
    output += '"t" : ' + str(t) + ',' 
    output += '"dt" : ' + str(dt) 
    output += '},'
    output += '"coefficients" : {'
    output += '"alpha" : ' + str(alpha) + ',' 
    output += '"beta" : ' + str(beta) + ',' 
    output += '"delta" : ' + str(delta) 
    output += '}'
    output += '}' 
    return output

def hopf_config( nq, alpha, beta, delta, p ):
    comment = ''

    for i in xrange(0, nq):
        alpha.append( 2*r.randint(0, 1) - 1 )
        delta.append( -1.0 )

    comment += "input: " + str(alpha) + "\n"
    
    mem = []
    for i in xrange( 0, p ):
        temp = []
        for j in xrange( 0, nq ):
            temp.append( 2*r.randint(0,1) - 1 )
        mem.append( temp )

    comment += "memories: " + str( np.array(mem) ) + "\n"
    
    memMat = np.matrix(mem).T
    ising_off = np.triu(memMat*memMat.T, 1)/float(nq)
    for i in xrange( 0, nq ):
        for j in xrange( i+1, nq ):
            beta.append( ising_off[i,j] );

    return comment

def toNum( bits ):
    ans = 0
    n = len( bits )-1
    for b in bits:
        if b == 1:
            ans += 2**n
        n -= 1
    return ans

#srun /usr/bin/env OMP_NUM_THREADS=16 ./OMP-SIM2 -s ""
if __name__ == "__main__":
    n = int( sys.argv[1] )
    samples = int( sys.argv[2] )
    lrgs = 1
    dt = 0.1
    alpha = []
    beta = []
    delta = []

    #for n in xrange( 4, 18, 2 ):
    #print >> sys.stderr, n, ":"
    #comments = hopf_config( n, alpha, beta, delta, int(0.14*n) )
    comments = hopf_config( n, alpha, beta, delta, 1 )
    res = toNum( alpha )
    #for t in xrange(1, 11):
    for t in xrange(2, 21):
        print >> sys.stderr, t,
        conf = build_config( comments, n, lrgs, res, 0.1*t, dt, alpha, beta, delta )
        #if t == 1:
        if t == 2:
            print conf
        proc = sp.Popen( [ "srun", "/usr/bin/env", "OMP_NUM_THREADS=16", \
                           "./OMP-SIM2", "-s", conf, str(samples) ], stdout=sp.PIPE )
        #proc = sp.Popen( [ "./OMP-SIM2", "-s", conf, str(samples) ], stdout=sp.PIPE )
        #print >> sys.stderr, cmd
        ccount, temp = proc.communicate()
        #print 10*t, n, ccount.strip()
        print t, n, ccount.strip()
    #print >> sys.stderr 
