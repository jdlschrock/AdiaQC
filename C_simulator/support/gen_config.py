import random as r
import numpy as np
import sys

def toNum( bits ):
    ans = 0
    n = len( bits )-1
    for b in bits:
        if b == 1:
            ans += 2**n
        n -= 1
    return ans


def build_config( comments, nq, lrgs, res, t, dt, alpha, beta, delta ):    
    output = ''
    output += '/*\n'
    output += comments + '\n'
    output += '*/\n'
    output += '{\n' 
    output += '    "scalars" : {\n'
    output += '        "nq" : ' + str(nq) + ',\n' 
    output += '        "lrgs" : ' + str(lrgs) + ',\n' 
    output += '        "res" : ' + str(res) + ',\n' 
    output += '        "t" : ' + str(t) + ',\n' 
    output += '        "dt" : ' + str(dt) + '\n'
    output += '    },\n'
    output += '    "coefficients" : {\n'
    output += '        "alpha" : ' + str(alpha) + ',\n' 
    output += '        "beta" : ' + str(beta) + ',\n' 
    output += '        "delta" : ' + str(delta) + '\n'
    output += '    }\n'
    output += '}\n' 
    return output


def rand_config( nq, alpha, beta, delta ):
    for i in xrange( nq ):
        alpha.append( round( r.random(), 3 ) ) 
        delta.append( 1.0 )

    for i in xrange( nq ):
        for j in xrange( i+1, nq ):
            beta.append( round( r.random(), 3 ) );

    return "Random config file"


def hopf_config( nq, alpha, beta, delta, p ):
    comment = ''

    for i in xrange(0, nq):
        alpha.append( 2*r.randint(0, 1) - 1 )
        #alpha.append( 1 if i < nq//2 else -1 )
        #alpha.append( 1 )
        delta.append( -1.0 )

    comment += "input: " + str(alpha) + "\n"
    
    mem = []
    for i in xrange( 0, p ):
        temp = []
        for j in xrange( 0, nq ):
            temp.append( 2*r.randint(0,1) - 1 )
            #temp.append( 1 )
        mem.append( temp )
    #mem = [ [1,-1]*(nq/2), [-1]*nq ]
    # hebb rule

    comment += "memories: " + str( np.array(mem) ) + "\n"
    
    memMat = np.matrix(mem).T
    #ising_off = (memMat*memMat.T)/float(nq)
    ising_off = np.triu(memMat*memMat.T, 1)/float(nq)
    for i in xrange( 0, nq ):
        for j in xrange( i+1, nq ):
            beta.append( ising_off[i,j] );
    #ising_diag = np.array(inputstate)

    return comment


if __name__ == "__main__":
    nq = int(sys.argv[1])
    lrgs = 10
    t = 10.0
    dt = 0.1 
    alpha = []
    beta = []
    delta = []

    r.seed( 0 ) 
    #comments = rand_config( nq, alpha, beta, delta )
    temp = int( r.uniform(0.0,0.14)*nq )
    p = max( 1, temp )
    #comments = hopf_config( nq, alpha, beta, delta, p )
    comments = hopf_config( nq, alpha, beta, delta, 1 )
    res = toNum( alpha )

    print build_config( comments, nq, lrgs, res, t, dt, alpha, beta, delta )
