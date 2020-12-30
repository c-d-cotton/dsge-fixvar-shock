#!/usr/bin/env python3
"""
I examine the impact of forward guidance in the standard simple NK model.
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

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
    sys.path.append(str(__projectdir__ / Path('submodules/dsge-perturbation/')))
    from dsge_bkdiscrete_func import discretelineardsgefull
    discretelineardsgefull(inputdict)

    
    if Rshock is True:
        Rshockvec = [-0.01] * numshockperiods
        from getshocks_func import getshocks_fixvar
        getshocks_fixvar(inputdict, Rshockvec, 'Rp', 'ui_')

    if Ishock is True:
        Ishockvec = [-np.log(1/0.98)] * numshockperiods
        from getshocks_func import getshocks_fixvar
        getshocks_fixvar(inputdict, Ishockvec, 'Ihat', 'ui_')

# Run:{{{1
# main(Rshock = True, Ishock = False, numshockperiods = 10)
main(Rshock = False, Ishock = True, numshockperiods = 10)
