#!/usr/bin/env python3
"""
I examine the impact of forward guidance in the standard simple NK model.
"""
# PYTHON_PREAMBLE_START_STANDARD:{{{

# Christopher David Cotton (c)
# http://www.cdcotton.com

# modules needed for preamble
import importlib
import os
from pathlib import Path
import sys

# Get full real filename
__fullrealfile__ = os.path.abspath(__file__)

# Function to get git directory containing this file
def getprojectdir(filename):
    curlevel = filename
    while curlevel is not '/':
        curlevel = os.path.dirname(curlevel)
        if os.path.exists(curlevel + '/.git/'):
            return(curlevel + '/')
    return(None)

# Directory of project
__projectdir__ = Path(getprojectdir(__fullrealfile__))

# Function to call functions from files by their absolute path.
# Imports modules if they've not already been imported
# First argument is filename, second is function name, third is dictionary containing loaded modules.
modulesdict = {}
def importattr(modulefilename, func, modulesdict = modulesdict):
    # get modulefilename as string to prevent problems in <= python3.5 with pathlib -> os
    modulefilename = str(modulefilename)
    # if function in this file
    if modulefilename == __fullrealfile__:
        return(eval(func))
    else:
        # add file to moduledict if not there already
        if modulefilename not in modulesdict:
            # check filename exists
            if not os.path.isfile(modulefilename):
                raise Exception('Module not exists: ' + modulefilename + '. Function: ' + func + '. Filename called from: ' + __fullrealfile__ + '.')
            # add directory to path
            sys.path.append(os.path.dirname(modulefilename))
            # actually add module to moduledict
            modulesdict[modulefilename] = importlib.import_module(''.join(os.path.basename(modulefilename).split('.')[: -1]))

        # get the actual function from the file and return it
        return(getattr(modulesdict[modulefilename], func))

# PYTHON_PREAMBLE_END:}}}

import numpy as np 

def main(Rshock = True, Ishock = False, numshockperiods = 50):
    """
    We see that the only way this yields a solution is if Pi is a state.
    """
    inputdict = {}
    inputdict['equations'] = [
    'Pihat = KAPPA * Xhat + BETA * Pihat_p'
    ,
    'Xhat = Xhat_p - 1/GAMMA*(Ihat - Pihat_p - Rnhat)'
    ,
    'Ihat = PHIpi * Pihat + PHIy * Xhat + Rnhat + ui_0'
    ,
    'Rp = Ihat - Pihat_p'
    ]
    for i in range(numshockperiods - 1):
        # 'ui_0_p = ui_1'
        inputdict['equations'].append('ui_' + str(i) + '_p = ui_' + str(i + 1))
    # 'ui_11_p = 0'
    inputdict['equations'].append('ui_' + str(numshockperiods - 1) + '_p')


    inputdict['paramssdict'] = {'BETA': 0.98, 'KAPPA': 1, 'GAMMA': 1, 'PHIpi': 1.5, 'PHIy': 0}


    inputdict['controls'] = ['Xhat', 'Pihat', 'Ihat', 'Rp']
    inputdict['states'] = [] + ['ui_' + str(i) for i in range(numshockperiods)]
    inputdict['shocks'] = ['Rnhat']

    inputdict['mainvars'] = ['Xhat', 'Pihat', 'Rp', 'Ihat']
    inputdict['showirfs'] = []

    inputdict['loglineareqs'] = True
    importattr(__projectdir__ / Path('submodules/dsge-perturbation/dsge_bkdiscrete_func.py'), 'discretelineardsgefull')(inputdict)

    
    if Rshock is True:
        Rshockvec = [-0.01] * numshockperiods
        importattr(__projectdir__ / Path('getshocks_func.py'), 'getshocks_fixvar')(inputdict, Rshockvec, 'Rp', 'ui_')

    if Ishock is True:
        Ishockvec = [-np.log(1/0.98)] * numshockperiods
        importattr(__projectdir__ / Path('getshocks_func.py'), 'getshocks_fixvar')(inputdict, Ishockvec, 'Ihat', 'ui_')

# Run:{{{1
# main(Rshock = True, Ishock = False, numshockperiods = 10)
main(Rshock = False, Ishock = True, numshockperiods = 10)