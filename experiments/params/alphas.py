
# coding: utf-8

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from new_SPOC import SPOC, SPOC_bootstrap
from generate_SPOC_model import generate_theta, P_to_A,\
                                find_permutation_Theta, find_permutation_B
import matlab.engine
from tqdm import tqdm

eng = matlab.engine.start_matlab()

n_nodes = 50
rho = 0.7
n_clusters = 3
B = np.diag([0.4, 0.7, 1.])
n_repetitions = 2
pn_number = n_clusters

EXP_DATA = pd.DataFrame(columns=["n_nodes", "n_clusters", "pure_nodes_number",
                                 "alpha", "seed", "method", "matrix", "error"])

methods = {"SPOC": lambda A, n_clusters: SPOC(A, n_clusters, use_ellipsoid=False, use_cvxpy=False),
           "GeoNMF": lambda A, n_clusters: eng.GeoNMF(A, n_clusters, 0.25, 0.95, nargout=2),
           "SPOC_bootstrap": lambda A, n_clusters: SPOC_bootstrap(A, n_clusters, n_repetitions=10, std_num=3),
          }


for alpha in tqdm(np.arange(0.5, 4.5, 0.5)):
    for repeat in tqdm(range(n_repetitions)):

        Theta = generate_theta(n_nodes=n_nodes, n_clusters=n_clusters,
                               pure_nodes_number=pn_number, seed=repeat,
                               alphas=[alpha/n_clusters] * n_clusters)

        P = rho * Theta.dot(B).dot(Theta.T)
        A = P_to_A(P, seed=repeat)

        for method_name in methods:
            if method_name == "GeoNMF":
                A = matlab.double(A.tolist())

            theta, b = methods[method_name](A, n_clusters)
            theta = np.array(theta)
            b = np.array(b)

            err, _ = find_permutation_Theta(Theta, theta)
            EXP_DATA = EXP_DATA.append(
                {"n_nodes": n_nodes, "n_clusters": n_clusters,
                 "pure_nodes_number": pn_number, "alpha": alpha, "seed": repeat,
                 "method": method_name, "matrix": "Theta", "error": err},
                ignore_index=True)

            err, _ = find_permutation_B(B, b)
            EXP_DATA = EXP_DATA.append(
                {"n_nodes": n_nodes, "n_clusters": n_clusters,
                 "pure_nodes_number": pn_number, "alpha": alpha, "seed": repeat,
                 "method": method_name, "matrix": "B", "error": err},
                ignore_index=True)

        EXP_DATA.to_csv("alphas.csv", index=None)


