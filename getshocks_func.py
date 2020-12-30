#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

def getshocks_fixvar(inputdict, varfgvec, varfgname, shockname, irfperiods = None, pltshow = True, pltsavename = None, postshockperiods = 10):
    """
    varfgvec is the vector of values of the variable I want to fix
    varfgname is the name of the variable I'm fixing.
    shockname is the general name of the shocks that will be set i.e. if shockname is ui_ then the shocks are ui_0, ui_1, ui_2, ldots, ui_10, ui_11, \ldots

    This works better with python 3.7 where you can have large numbers of shocks (otherwise limit of 255 arguments on lambdify comes into play). To overcome this, don't use lambdify.
    
    inputdict is argument returned from running my standard dsge codes

    The name of the shock in the DSGE code should be ui_0, ui_1 etc.
    """
    import numpy as np
    import scipy.linalg

    numshockperiods = len(varfgvec)

    nonshockstates = [state for state in inputdict['states'] if not state.startswith(shockname)]

    # this is the (x,x,...,0)' part of the system
    
    constant = np.append(varfgvec, np.zeros(len(nonshockstates) * (numshockperiods - 1)))

    # [[DELTA_0 DELTA_1], [0, DELTA_0]]
    topleft = np.zeros([numshockperiods, numshockperiods])
    for i in range(0, numshockperiods):
        for j in range(0, numshockperiods):
            if j >= i:
                topleft[i, j] = inputdict['gx'][inputdict['controlposdict'][varfgname], inputdict['stateposdict'][shockname + str(j - i)]]

    if len(nonshockstates) > 0:
        toprightlist = [np.zeros([numshockperiods, numshockperiods - 1]) for state in nonshockstates]
        for statei in range(len(nonshockstates)):
            state = nonshockstates[statei]
            topright = np.zeros([numshockperiods, numshockperiods - 1])
            for i in range(0, numshockperiods - 1):
                toprightlist[statei][i + 1, i] = inputdict['gx'][inputdict['controlposdict'][varfgname], inputdict['stateposdict'][state]]
        topright = np.concatenate(toprightlist, axis = 1)


        # note that I'm including the -1 term by defining a negative identity matrix
        bottomright = np.identity((numshockperiods - 1) * len(nonshockstates)) * -1
        for stateinum in range(len(nonshockstates)):
            for statejnum in range(len(nonshockstates)):
                statei = nonshockstates[stateinum]
                statej = nonshockstates[statejnum]
                # add impact of statej on statei
                for i in range(0, numshockperiods - 2):
                    bottomright[stateinum * (numshockperiods - 1) + i + 1, statejnum * (numshockperiods - 1) + i] = inputdict['hx'][inputdict['stateposdict'][statei], inputdict['stateposdict'][statej]]

        bottomleftlist = [np.zeros([numshockperiods - 1, numshockperiods]) for state in nonshockstates]
        for statei in range(len(nonshockstates)):
            state = nonshockstates[statei]
            for i in range(0, numshockperiods - 1):
                for j in range(0, numshockperiods):
                    if j >= i:
                        bottomleftlist[statei][i, j] = inputdict['hx'][inputdict['stateposdict'][state], inputdict['stateposdict'][shockname + str(j - i)]]
        bottomleft = np.concatenate(bottomleftlist, axis = 0)

    # print('topleft')
    # print(topleft)
    # print('bottomleft')
    # print(bottomleft)
    # print('topright')
    # print(topright)
    # print('bottomright')
    # print(bottomright)

    # if don't have any nonshockstates, only need to include topleft part of matrix
    if len(nonshockstates) > 0:
        left = np.concatenate([topleft, bottomleft], axis = 0)
        right = np.concatenate([topright, bottomright], axis = 0)
        varmatrix = np.concatenate([left, right], axis = 1)
    else:
        varmatrix = topleft

    # get uivec
    coeff = np.linalg.inv(varmatrix).dot(constant)
    uivec = coeff[0: numshockperiods]

    # get X0
    X0 = np.zeros(len(inputdict['states'] + inputdict['shocks']))
    for i in range(numshockperiods):
        X0[inputdict['stateposdict'][shockname + str(i)]] = uivec[i]

    # generate irf
    if irfperiods is None:
        irfperiods = numshockperiods + postshockperiods
    sys.path.append(str(__projectdir__ / Path('submodules/dsge-perturbation/')))
    from dsge_bkdiscrete_func import irmatrix
    XY = irmatrix(inputdict['gx'], inputdict['hx'], X0, T = irfperiods)
    irfvars = [inputdict['stateshockcontrolposdict'][varname] for varname in inputdict['mainvars']]
    irfnames = inputdict['mainvarnames']
    XY2 = XY[:, irfvars]

    if pltsavename is None and inputdict['savefolder'] is not None:
        pltsavename = inputdict['savefolder'] + 'fixvarirf.png'
    # this gives me the option to not save the plot even if I specify a savefolder in inputdict
    if pltsavename is False:
        pltsavename = None
    sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/statespace')))
    from statespace_func import irgraphs
    irgraphs(XY2, names = irfnames, pltshow = pltshow, pltsavename = pltsavename)

    return(uivec)

