
# coding: utf-8

import numpy as np
import pandas as pd
from new_SPOC import *
import old_SPOC as old_SPOC
from generate_SPOC_model import *
import matlab.engine
from tqdm import tqdm
from IPython.display import clear_output, display

eng = matlab.engine.start_matlab()


n_nodes = 1000
n_clusters = 3
n_repetitions = 10
pn_number = n_clusters

EXP_DATA = pd.DataFrame(columns=["n_nodes", "n_clusters", "pure_nodes_number", "eps",                                 "seed", "method", "matrix", "error"])

for eps in np.arange(0.05, 0.5, 0.05):
    B = np.diag([0.5-eps, 0.5, 0.5+eps])
    B = B / np.max(B)
    
    for repeat in tqdm(range(n_repetitions)):
        Theta = generate_theta(n_nodes=n_nodes, n_clusters=n_clusters, pure_nodes_number=pn_number, seed=repeat)
        P = Theta.dot(B).dot(Theta.T)
        A = P_to_A(P, seed=repeat)
        
        A_mat = matlab.double(A.tolist())
        theta_geo, b_geo = eng.GeoNMF(A_mat, n_clusters, 0.25, 0.95, nargout=2)
        theta_geo = np.array(theta_geo)
        b_geo = np.array(b_geo)
        
        err, _ = find_permutation_Theta(Theta, theta_geo)
        EXP_DATA = EXP_DATA.append({"n_nodes":n_nodes, "n_clusters":n_clusters, "pure_nodes_number":pn_number,                "eps":eps, "seed":repeat, "method":"Geo_NMF", "matrix":"Theta", "error":err}, ignore_index=True)
        err, _ = find_permutation_B(B, b_geo)
        EXP_DATA = EXP_DATA.append({"n_nodes":n_nodes, "n_clusters":n_clusters, "pure_nodes_number":pn_number,                "eps":eps, "seed":repeat, "method":"Geo_NMF", "matrix":"B", "error":err}, ignore_index=True)
        
        theta_spa, b_spa = SPOC(A, n_clusters)
        err, _ = find_permutation_Theta(Theta, theta_spa)
        EXP_DATA = EXP_DATA.append({"n_nodes":n_nodes, "n_clusters":n_clusters, "pure_nodes_number":pn_number,                "eps":eps, "seed":repeat, "method":"basic_SPA", "matrix":"Theta", "error":err}, ignore_index=True)
        err, _ = find_permutation_B(B, b_spa)
        EXP_DATA = EXP_DATA.append({"n_nodes":n_nodes, "n_clusters":n_clusters, "pure_nodes_number":pn_number,                "eps":eps, "seed":repeat, "method":"basic_SPA", "matrix":"B", "error":err}, ignore_index=True)
        
        theta_bootstrap, b_bootstrap = SPOC_bootstrap(A, n_clusters, n_repetitions=100, std_num=3)
        err, _ = find_permutation_Theta(Theta, theta_bootstrap)
        EXP_DATA = EXP_DATA.append({"n_nodes":n_nodes, "n_clusters":n_clusters, "pure_nodes_number":pn_number,                "eps":eps, "seed":repeat, "method":"bootstrap_SPA", "matrix":"Theta", "error":err}, ignore_index=True)
        err, _ = find_permutation_B(B, b_bootstrap)
        EXP_DATA = EXP_DATA.append({"n_nodes":n_nodes, "n_clusters":n_clusters, "pure_nodes_number":pn_number,                "eps":eps, "seed":repeat, "method":"bootstrap_SPA", "matrix":"B", "error":err}, ignore_index=True)

EXP_DATA.to_csv("skewness.csv", index=None)


