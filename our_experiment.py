'''
our experiment
'''

import math as m

# import time
# import itertools
import numpy as np
import time
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as lasparse
import pandas as pd

import SBMO
from SPOC import SPOC_get_Z_from_UL, euclidean_proj_simplex
from new_SPOC import new_SPOC_get_Z_from_UL
from generate_SPOC_model import models_param, find_permutation_Theta
pd.set_option('display.width', 1000)

Nsimu = 10
NbInit = 5

Model = "SPOC"



def get_comms(hatTheta, K):
    comm_matr = hatTheta >= 1.0 / K
    return comm_matr

def make_res(name, err, err_theta, emni, args, time):
    result = {'method': name,
              'error Z': err,
              'error fro theta': err_theta,
              'eNMI': emni,
              'time': time,
              'NbInit': NbInit}
    result.update(args)
    return result

def calc_res(name, Zest, theta, Z0, Theta, start):
    fintime = time.time()
    fpos, fneg, err, perm = SBMO.evaluation(Zest, Z0)
    try:
        theta = np.array([euclidean_proj_simplex(x) for x in theta])
        err_theta, _ = find_permutation_Theta(theta, Theta)
    except:
        err_theta = float('nan')
    enmi = SBMO.ENMI(Zest, Z0)
    return make_res(name, err, err_theta, enmi, args, start-fintime)

results = []
k = 0
compt = 0
globstart = time.time()
while k < Nsimu:
    k += 1
    for indx, model in enumerate(models_param()):
        print('{} repeat, model {}: glob time: {}'.format(k, indx, time.time()-globstart))
        A, Theta, B, args = model
        P = Theta.dot(B).dot(Theta.T)
        K = Theta.shape[1]

        # true binary membership
        Z0 = get_comms(Theta, K)
        Ahat = A.astype(np.float)
        degrees = np.sum(Ahat, axis=1)
        dmax = np.max(degrees)
        sumA = np.sum(Ahat)
        Asparse = sparse.csr_matrix(Ahat)
        # eigendecomposition
        ll, vv = lasparse.eigsh(Asparse, K, which='LA')
        indices = ll.argsort()
        L = ll[indices]
        V = vv[:, indices]
        V = np.real(V)
        # SPOC_new
        start = time.time()
        Zest, theta = new_SPOC_get_Z_from_UL(V, L, K)
        results.append(calc_res('SPOC_new', Zest, theta, Z0, Theta, start))
        
	# SC
        start = time.time()
        try:
            Zest, theta = SBMO.ProcessSC(V, degrees, Initoption='smallplusplus', NbInit=NbInit)
        except:
            Zest, theta = float('nan'), float('nan')
        results.append(calc_res('SC', Zest, theta, Z0, Theta, start))
        # CombSC
        start = time.time()
        try:
            Zest, theta = SBMO.ProcessSAAC(V, degrees, Omax=0, Initoption='smallplusplus', NbInit=NbInit)
        except:
            Zest, theta = float('nan'), float('nan')
        results.append(calc_res('SAAC', Zest, theta, Z0, Theta, start))
        # OCCAM
        start = time.time()
        try:
            Zest, theta = SBMO.ProcessOCCAM(L, V, sumA, degrees, Initoption='plusplus', NbInit=NbInit)
        except:
            Zest, theta = float('nan'), float('nan')
        results.append(calc_res('OCCAM', Zest, theta, Z0, Theta, start))
        # SPOC
        start = time.time()
        try:
            Zest, theta = SPOC_get_Z_from_UL(V, L, K, None, new=True)
        except:
            Zest, theta = float('nan'), float('nan')
        results.append(calc_res('SPOC', Zest, theta, Z0, Theta, start))
        compt += 1
        print('\t\t', compt)

        df_results = pd.DataFrame(results)
        df_results.to_csv('temp_results.csv')
        print(df_results[-5:])
