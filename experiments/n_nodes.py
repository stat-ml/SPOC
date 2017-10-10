
# coding: utf-8


from time import time

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


n_clusters = 3
n_repetitions = 5
pn_number = n_clusters
random_vec = np.random.rand(3,)

EXP_DATA = pd.DataFrame(columns=["n_nodes", "n_clusters", "pure_nodes_number",                                 "seed", "method", "matrix", "error", "time"])

for n_nodes in np.arange(1000, 6000, 500):
    for repeat in tqdm(range(n_repetitions)):
        
        random.seed(repeat)
        np.random.seed(repeat)
        random_vec = 0.5 * np.random.rand(3,)
        B = np.diag([0.5] * 3 + random_vec)
        B = B / np.max(B)

        Theta = generate_theta(n_nodes=n_nodes, n_clusters=n_clusters, pure_nodes_number=pn_number, seed=repeat)
        P = Theta.dot(B).dot(Theta.T)
        A = P_to_A(P, seed=repeat)

        for method_name in tqdm(methods): 
            if method_name == "GeoNMF":
                A = matlab.double(A.tolist())

            time_start = time()
            theta, b = methods[method_name](A, n_clusters)
            time_end = time()
            theta = np.array(theta)
            b = np.array(b)

            err, _ = find_permutation_Theta(Theta, theta)
            EXP_DATA = EXP_DATA.append({"n_nodes": n_nodes, "n_clusters": n_clusters,                                        "pure_nodes_number": pn_number, "seed":repeat,                                        "method":method_name, "matrix":"Theta", "error":err,                                        "time": time_end - time_start}, ignore_index=True)

            err, _ = find_permutation_B(B, b)
            EXP_DATA = EXP_DATA.append({"n_nodes": n_nodes, "n_clusters": n_clusters,                                        "pure_nodes_number": pn_number, "seed": repeat,                                        "method": method_name, "matrix": "B", "error": err,                                        "time": time_end - time_start}, ignore_index=True)

    EXP_DATA.to_csv("n_nodes.csv", index=None)
