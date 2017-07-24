# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:28:08 2015
Contains clustering algorithms (Kmeans, Combinatorial Kmeans, K medians, Clustering by distance)
and various initalization procedures 
@author: ekaufman
"""

import itertools
import math as m
import random as rd
from random import *

import numpy as np
from numpy import linalg as la
from numpy import log as log


# several initalization procedures for K-means, K-median and combinatorial K-means

def CheatInit(V, K, Z):
    '''draw a pure node at random in each community'''
    C = np.zeros((K, K))
    for k in range(K):
        indices = np.where(Z[:, k] == 1)[0]
        i = indices[int(len(indices) * random())]
        C[k, :] = V[i, :]
    return C


def RandomInit(V, K):
    ''' chooses K rows of V at random'''
    (n, R) = V.shape
    init = list(range(n))
    rd.shuffle(init)
    C = V[init[0:K], :]
    return C


def PlusPlusInit(V, K):
    ''' degrees : vector of degrees of the nodes
    K-means ++ initialization on the rows of V'''
    (n, R) = np.shape(V)
    C = np.zeros((K, R), float)
    dist = np.zeros(n, float)
    # pick the first centroid at random
    idx = int(random() * n)
    C[0, :] = V[idx, :]  # first centroid
    j = 1
    while j < K:  # other centroids far from previous ones
        for i in range(n):
            v = V[i, :]
            norms = np.zeros(j)
            for s in range(j):
                norms[s] = la.norm(C[s, :].T - v)
            dd = min(norms)
            dist[i] = dd * dd
        u = random()
        dist = np.cumsum(dist / np.sum(dist))
        i = 0
        while (u > dist[i]):
            i += 1
        C[j, :] = V[i, :]
        j += 1
    return C


def SmallDegInit(V, K, degrees):
    ''' K random rows of V with small degrees'''
    (n, R) = V.shape
    MD = np.median(degrees)
    init = list(range(n))
    rd.shuffle(init)
    listIndex = []
    start = 0
    while (len(listIndex) < K):
        i = init[start]
        if (degrees[i] < MD):
            listIndex.append(i)
        start += 1
    C = V[listIndex, :]
    return C


def SmallPlusPlusInit(V, K, degrees):
    ''' degrees : vector of degrees of the nodes
    K-means ++ initialized with a node of small degree'''
    (n, R) = V.shape
    C = np.zeros((K, R), float)
    dist = np.zeros(n, float)
    MD = np.median(np.array(degrees))
    init = list(range(n))
    rd.shuffle(init)
    start = 0
    i = init[start]
    while (degrees[i] > MD):
        start += 1
        i = init[start]
    C[0, :] = V[i, :]  # first centroid randomly chosen among nodes of small degree
    j = 1
    while j < K:  # other centroids far from previous ones
        for i in range(n):
            v = V[i, :]
            norms = np.zeros(j)
            for s in range(j):
                norms[s] = la.norm(C[s, :].T - v)
            dd = min(norms)
            dist[i] = dd * dd
        u = random()
        dist = np.cumsum(dist / np.sum(dist))
        i = 0
        while (u > dist[i]):
            i += 1
        C[j, :] = V[i, :]
        j += 1
    return C


# K-means algorithm (Loyds)


def Kmeans(M, K, C0):
    '''
    M: matrix whose rows have to be clustered (n by R)
    K: number of clusters to find
    C0 : initial matrix of centroids (K by R) 
    Returns a membership vector z : z[i] gives the cluster to which i belongs, matrix of centroids C and  
    K-means score of the partition found
    '''
    (n, x) = M.shape
    shift = 1
    score = 0
    Centroids = C0  # matrix of centroids
    z = np.zeros(n, dtype=int)  # membership vector
    while (shift > 10 ** (-10)):
        # optimal membership
        oldscore = score
        for i in range(n):
            X = M[i, :]
            index = 0  # index of the closest centroid
            d = 100000  # distance to the closest centroid
            for k in range(K):
                c = Centroids[k, :]
                a = X - c
                dist = np.dot(a, a.T)
                if (dist == d):
                    if (random() < 0.5):
                        index = k
                elif (dist < d):
                    d = dist
                    index = k
            z[i] = index
        # recomputing clusters centroids and computing the score
        score = 0
        for k in range(K):
            # form a vector giving the indices of nodes in cluster k
            indices = np.where(z == k)[0]
            # print(len(indices))
            l = len(indices)
            if (l == 0):
                Centroids[k, :] = np.zeros((1, x))
                # a vanishing centroid is set to a null vector (it could be arbitrairy)
            else:
                Centroids[k, :] = np.mean(M[indices, :], 0)
                A = M[indices, :] - np.tile(Centroids[k, :], (l, 1))
                score = score + np.trace(np.dot(A, A.T))
        shift = score - oldscore
        # print(Centroids)
    return z, Centroids, score


def loss2(V, Z, C, n):
    res = 0
    for i in range(n):
        res += (la.norm(V[i, :] - np.dot(Z[i, :], C))) ** 2
    return res


def loss(V, Z, C, n):
    res = 0
    for i in range(n):
        res += la.norm(V[i, :] - np.dot(Z[i, :], C))
    return res


def Kmedians(V, K, C0):
    ''' V: matrix n by K whose rows are clustered into K clusters
    C0 : initial centroids'''
    (n, R) = np.shape(V)
    Z = np.zeros((n, K))  # membership vector
    C = C0
    # loop If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
    precision = 10
    valueM = 10000000
    compt = 0
    while (precision > 0.0001) & (compt < 50):
        # optimization in Z  
        valuemax = 0
        for i in range(n):
            vv = np.tile(V[i, :], (K, 1))
            mat = vv - C
            mat2 = np.dot(mat, mat.T)
            norm = np.sqrt(mat2[range(K), range(K)])  # diagonal gives the distance
            index = np.argmin(norm)
            Z[i, :] = np.zeros(K)
            Z[i, index] = 1
            valuemax = valuemax + np.amin(norm)
        # continuation test
        precision = valueM - valuemax
        valueM = valuemax
        # optimization in C 
        s = 0
        ecart = 1
        step = 1
        valueCurrent = valueM
        while (ecart > 0.0001) & (s < 50):
            # computing the gradient
            A = V - np.dot(Z, C)
            for i in range(n):
                a = A[i, :]
                A[i, :] = a / (max(la.norm(a), 0.00001))
            Gradient = -np.dot(Z.T, A)
            # gradient step
            C0 = C - (step) * Gradient
            value = loss(V, Z, C0, n)
            v = 0
            while (value > valueCurrent) & (v < 30):
                # reduce the step until we have a smaller value of 
                # the function to minimize
                step = 0.5 * step
                C0 = C - (step) * Gradient
                value = loss(V, Z, C0, n)
                v += 1
            # print("ajusting stepsize : %d" %v)
            if (v == 30):
                C0 = C  # no update
            valueCurrent = value
            step = 1.2 * step
            ecart = la.norm(C - C0)
            C = C0
            s += 1
        # print(" gradient iterations : %d" %s)
        compt += 1
    # print("total iteration : %d" %compt)
    # Z turned into a vector 
    z = np.zeros(n)
    for k in range(K):
        indices = list(np.where(Z[:, k] == 1)[0])
        z[indices] = k
    return z, C, valueM


# Combinatorial K-means algorithm


def CombKmeans(V, C0, Omax=0):
    ''' V: matrix n by K whose rows are clustered into K clusters
    C0 : initial centroids
    Omax: maximum overlap present (to speed up)
    combinatorial Kmeans clustering of the rows of V into K overlapping clusters'''
    (n, R) = V.shape
    (K, x) = np.shape(C0)
    if Omax == 0:
        # all possibilities
        Omax = 2 ** K - 1
    Z = np.zeros((n, K))
    C = C0
    # storage of the matrix of possible configurations
    KK = 2 ** K - 1
    Comb = np.zeros((KK, K))
    j = 0
    for k in range(Omax):
        for l in itertools.combinations(range(K), k + 1):
            z = np.zeros(K)
            z[list(l)] = 1
            Comb[j] = z
            j = j + 1
    Comb = Comb[range(j), :]  # reduced size j <= KK
    # loop 
    precision = 10
    valueM = 10000000
    compt = 0
    while (precision > 0.0001):
        # optimization in Z      
        valuemax = 0
        for i in range(n):
            vv = V[i, :]
            norms = np.apply_along_axis(la.norm, 1, np.dot(Comb, C) - vv)
            index = np.argmin(norms)
            Z[i, :] = Comb[index, :]
            valuemax = valuemax + np.amin(norms)
        # continuation test  
        precision = valueM - valuemax
        valueM = valuemax
        # optimization in C 
        mat1 = np.dot(Z.T, Z)
        mat2 = np.dot(Z.T, V)
        try:
            C = np.dot(la.inv(mat1), mat2)
        except:
            # random re-initalization 
            init = list(range(n))
            rd.shuffle(init)
            C = V[init[0:K], :]
            precision = 1
            print("there was a re-initialization")
    return Z, C, valueM


def CombKmeans2(V, C0, Omax=0):
    ''' (small speedup) V: matrix n by K whose rows are clustered into K clusters
    C0 : initial centroids
    Omax: maximum overlap present (to speed up)
    combinatorial Kmeans clustering of the rows of V into K overlapping clusters'''
    (n, R) = V.shape
    (K, x) = np.shape(C0)
    if Omax == 0:
        # all possibilities
        Omax = 2 ** K - 1
    Z = np.zeros((n, K))
    C = C0
    # storage of the matrix of possible configurations
    KK = 2 ** K - 1
    Comb = np.zeros((KK, K))
    j = 0
    for k in range(Omax):
        for l in itertools.combinations(range(K), k + 1):
            z = np.zeros(K)
            z[list(l)] = 1
            Comb[j] = z
            j = j + 1
    Comb = Comb[range(j), :]  # reduced size j <= KK
    # loop 
    precision = 10
    valueM = 10000000
    while (precision > 0.0001):
        # optimization in Z 
        Z = np.zeros((K, n), int)
        Z[0, :] = np.ones(n)
        ZZ = np.zeros((K, 1), int)
        ZZ[0, :] = 1
        dd = np.apply_along_axis(la.norm, 0, (np.dot(C.T, ZZ) - V.T))
        for jj in range(j):
            ZZ = np.zeros((K, 1))
            ZZ[:, 0] = Comb[jj, :]
            delta = dd - np.apply_along_axis(la.norm, 0, (np.dot(C.T, ZZ) - V.T))
            change = np.maximum(np.sign(delta), 0)
            dd = dd - change * delta
            Z = Z - change * Z + change * ZZ
        Z = Z.T
        valuemax = np.sum(dd)
        # continuation test  
        precision = valueM - valuemax
        valueM = valuemax
        # optimization in C 
        mat1 = np.dot(Z.T, Z)
        mat2 = np.dot(Z.T, V)
        try:
            C = np.dot(la.inv(mat1), mat2)
        except:
            # random re-initalization 
            init = list(range(n))
            rd.shuffle(init)
            C = V[init[0:K], :]
            precision = 1
            print("there was a re-initialization")
    return Z, C, valueM


# clustering by distance


def ClusteringDistance(M, thres):
    ''' clusters the rows of M depending on whether their 
    distance exceed or not the threshold thres 
    returns the vector indicating the clusters and 
    the number of clusters found'''
    (n, K) = np.shape(M)
    # form the distance matrix
    D = 2 * np.ones((n, n)) - 2 * np.dot(M, M.T)
    # clustering by distance 
    z = np.zeros(n)  # indicates the labels of the nodes
    nodes = list(range(n))  # nodes that remains to be clustered
    l = len(nodes)
    c = 0  # number of clusters already found
    while (l > 0):
        # pick a node at random a remaining node        
        r = int(l * random())
        i = nodes[r]
        d = D[i, :]
        index = np.where(d < thres)[0]  # potential community of node i
        if (len(index) == 0):
            # the node is alone 
            z[i] = c
            nodes.remove(i)
        else:
            # there are several nodes
            ind = list(index)
            # indF=[j in ind if j in nodes]
            # z[indF]=c
            # nodes=[j in nodes if j not in indF]
            for j in ind:
                if j in nodes:
                    nodes.remove(j)
                    z[j] = c
        c += 1
        l = len(nodes)
    return z, c
