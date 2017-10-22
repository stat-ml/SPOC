
# coding: utf-8

import numpy as np
import pandas as pd
from new_SPOC import *
import old_SPOC as old_SPOC
from generate_SPOC_model import *
import matlab.engine

from tqdm import tqdm_notebook, tqdm
import random

eng = matlab.engine.start_matlab()

methods = {"SPOC": lambda A, n_clusters: SPOC(A, n_clusters, use_ellipsoid=False, use_cvxpy=False),
           "GeoNMF": lambda A, n_clusters: eng.GeoNMF(A, n_clusters, 0.25, 0.95, nargout=2),
           "SPOC_bootstrap": lambda A, n_clusters: SPOC_bootstrap(A, n_clusters, n_repetitions=10, std_num=3),
          }


n_nodes = 1000
n_clusters = 3
n_repetitions = 10
pn_number = n_clusters
B = np.diag([1, 1, 1]);

EXP_DATA = pd.DataFrame(columns=["n_nodes", "n_clusters", "pure_nodes_number",\
                            "rho", "seed", "method", "matrix", "error"])

for rho in np.arange(0.1, 1.1, 0.1):
    for repeat in tqdm(range(n_repetitions)):
        
        random.seed(repeat)
        np.random.seed(repeat)

        Theta = generate_theta(n_nodes=n_nodes, n_clusters=n_clusters, pure_nodes_number=pn_number, seed=repeat)
        P = rho * Theta.dot(B).dot(Theta.T)
        A = P_to_A(P, seed=repeat)

        for method_name in tqdm(methods): 
            if method_name == "GeoNMF":
                A = matlab.double(A.tolist())

            theta, b = methods[method_name](A, n_clusters)
            theta = np.array(theta)
            b = np.array(b)

            err, _ = find_permutation_Theta(Theta, theta)
            EXP_DATA = EXP_DATA.append({"n_nodes": n_nodes, "n_clusters": n_clusters,\
                    "pure_nodes_number": pn_number, "rho": rho, "seed":repeat,\
                    "method":method_name, "matrix":"Theta", "error":err}, ignore_index=True)

            err, _ = find_permutation_B(B, b)
            EXP_DATA = EXP_DATA.append({"n_nodes": n_nodes, "n_clusters": n_clusters,\
                    "pure_nodes_number": pn_number, "rho": rho, "seed": repeat,\
                    "method": method_name, "matrix": "B", "error": err}, ignore_index=True)

    EXP_DATA.to_csv("rho.csv", index=None)
