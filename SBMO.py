'''
code from kaufman
Tests of the algorithm proposed for SBM-O 
'''

import itertools
import math as m
from random import *

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as lasparse
from numpy import linalg as la
from numpy import log as log

import Clustering as Cluster


# random models


def randomCom(alpha):
    '''random community taken from alpha'''
    l = 0
    u = random()
    while (u > alpha[l]):
        u -= alpha[l]
        l += 1
    return l


def randomGraph(n, K, nt, Omax, AvgD, alpha):
    '''create a random instance of SBMgraph with given n, K, nt (sum of Z), 
    AvgD and alpha (distribution of communities)
    pairwise overlap only'''

    Z = np.zeros((n, K), int)
    for i in range(n):
        Z[i][randomCom(alpha)] = 1
        k = 1
        while (random() * nt > n and k < Omax):  # overlap
            l = randomCom(alpha)
            while (Z[i][l] == 1):
                l = randomCom(alpha)
            Z[i][l] = 1
            k += 1
    q = la.norm(alpha)
    p = 2 * AvgD * n / nt / nt / q / q
    B = np.dot(Z, np.transpose(Z))
    A = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i):
            if (random() < p * B[i][j]):
                A[i][j] = 1
                A[j][i] = 1
    return Z, A


from operator import itemgetter


def RandomZ(n, K, p, Omax=0):
    '''randommy generate a membershiph matrix Z of size (n,K) with overlaps
    and such that the proportion of pure nodes is p 
    (each community containing the same number of pure nodes)'''
    if (Omax < 2):
        Omax = K
    Z = np.zeros((n, K))
    npur = int(p * n / float(K))
    npurtot = K * npur
    for i in range(K):
        Z[i * npur:(i + 1) * npur, i] = 1
    for i in range(npurtot, n):
        tab = np.arange(K)
        np.random.shuffle(tab)
        nbcomm = 2 + int(random() * (Omax - 1))
        indices = tab[0:nbcomm]
        Z[i, indices] = 1
    Z3 = list(Z)
    Z3 = sorted(Z3, key=itemgetter(*list(range(K))))
    Z3 = np.array(Z3)
    return Z3


def DrawRG(A):
    '''draws a random graph with mean adjacency matrix A'''
    (n, l) = np.shape(A)
    Ahat = np.random.random((n, n))
    Ahat = np.tril(Ahat) + np.tril(Ahat, -1).T
    return 1.0 * (Ahat < A)


# Algorithms


def SAAC(Adj, K, Omax=0, Initoption='smallplusplus', NbInit=1, TypeEig='LA'):
    '''given the adjacency matrix, return the community membership table,
    K has to be known 
    initialization options : 'random' 'smallDeg' 'plusplus'
    TypeEig : 'LA' leading algebraic eigenvalues, 'LM' leading eigenvalues in module
    '''
    (n, w) = np.shape(Adj)
    # compute leading eigenvectors
    Asparse = sparse.csr_matrix(Adj)
    ll, vv = lasparse.eigsh(Asparse, K, which=TypeEig)
    index = ll.argsort()
    ll = ll[index]  # sorted in increasing order
    V = vv[:, index]  # V is the matrix to use
    V = np.real(V)
    # initialization for CombKMeans
    C0 = np.zeros((K, K))
    C = np.zeros((K, K))
    # combinatorial K-means clustering 
    score = 1000000
    Z = np.zeros((n, K))
    for s in range(NbInit):
        if Initoption == 'random':
            C0 = Cluster.RandomInit(V, K)
        elif Initoption == 'plusplus':
            C0 = Cluster.PlusPlusInit(V, K)
        elif Initoption == 'smallDeg':
            degrees = np.dot(Adj, np.ones((n, 1)))
            C0 = Cluster.SmallDegInit(V, K, degrees)
        elif Initoption == 'smallplusplus':
            degrees = np.dot(Adj, np.ones((n, 1)))
            C0 = Cluster.SmallPlusPlusInit(V, K, degrees)
        Z1, C1, score1 = Cluster.CombKmeans(V, C0, Omax)
        if score1 < score:
            score = score1
            Z = Z1
            C = C1
    return Z, C, score


def OCCAM(A, K, threshold, Initoption='plusplus', NbInit=1):
    ''' spectral algorithm of Levina et al 2014'''
    (n, y) = A.shape
    Asparse = sparse.csr_matrix(A)
    L, V = lasparse.eigsh(Asparse, K, which='LA')
    L = np.real(L)
    V = np.real(V)
    ind = np.where(L < 0)[0]
    L[list(ind)] = 0
    L = np.diag(L)
    X = np.dot(V, np.sqrt(L))
    alpha = np.sum(A) / (n * (n - 1) * K)
    tau = 0.1 * alpha ** (0.2) * K ** (1.5) / (n ** (0.3))
    for i in range(n):
        x = X[i, :]
        X[i, :] = x / (la.norm(x) + tau)
    score = 100000
    Cest = np.zeros((K, K))
    C0 = np.zeros((K, K))
    for k in range(NbInit):
        if Initoption == 'random':
            C0 = Cluster.RandomInit(V, K)
        elif Initoption == 'plusplus':
            C0 = Cluster.PlusPlusInit(V, K)
        elif Initoption == 'smallDeg':
            degrees = np.dot(A, np.ones((n, 1)))
            C0 = Cluster.SmallDegInit(V, K, degrees)
        elif Initoption == 'smallplusplus':
            degrees = np.dot(A, np.ones((n, 1)))
            C0 = Cluster.SmallPlusPlusInit(V, K, degrees)
        z, C, score1 = Cluster.Kmedians(X, K, C0)
        if score1 < score:
            Cest = C
            score = score1
    Ztmp = np.dot(X, la.inv(Cest))
    ZLev = np.zeros((n, K))
    for i in range(n):
        z = Ztmp[i, :]
        z2 = z / (la.norm(z))
        indices = list(np.where(z2 > threshold)[0])
        # another possible modification
        # indices=list(np.where(np.abs(z2)>threshold)[0]) 
        if (len(indices) == 0):
            ind = np.argmax(z2)
            ZLev[i, ind] = 1
        else:
            ZLev[i, indices] = 1
    return ZLev, Cest, score


# Process spectral algorithm (the eigendecomposition being given)


def ProcessSC(V, degrees, Initoption='smallplusplus', NbInit=1):
    '''Normalized Spectral clustering
    given the matrix of leading eigenvectors V, 
    return the community membership table,
    K has to be known 
    several initialization option : 'random' 'smallDeg' 'plusplus'
    TypeEig : 'LA' leading algebraic eigenvalues, 'LM' leading eigenvalues in module
    '''
    (n, K) = np.shape(V)
    Vnorm = np.zeros((n, K))
    for i in range(n):
        v = V[i, :]
        Vnorm[i, :] = v / la.norm(v)
        # initialization for CombKMeans
    C0 = np.zeros((K, K))
    C = np.zeros((K, K))
    # combinatorial K-means clustering 
    score = 1000000
    z = np.zeros(n)
    for s in range(NbInit):
        if Initoption == 'random':
            C0 = Cluster.RandomInit(Vnorm, K)
        elif Initoption == 'plusplus':
            C0 = Cluster.PlusPlusInit(Vnorm, K)
        elif Initoption == 'smallDeg':
            C0 = Cluster.SmallDegInit(Vnorm, K, degrees)
        elif Initoption == 'smallplusplus':
            C0 = Cluster.SmallPlusPlusInit(Vnorm, K, degrees)
        z1, C1, score1 = Cluster.Kmeans(Vnorm, K, C0)
        if score1 < score:
            score = score1
            z = z1
            C = C1
        Z = np.zeros((n, K))
        Z[range(n), list(z)] = 1
    return Z, Z


def ProcessSAAC(V, degrees, Omax=0, Initoption='smallplusplus', NbInit=1):
    '''given the spectral embedding V, return the community membership table,
    K has to be known 
    several initialization option : 'random' 'smallDeg' 'plusplus'
    TypeEig : 'LA' leading algebraic eigenvalues, 'LM' leading eigenvalues in module
    '''
    (n, K) = np.shape(V)
    # initialization for CombKMeans
    C0 = np.zeros((K, K))
    C = np.zeros((K, K))
    # combinatorial K-means clustering 
    score = 1000000
    Z = np.zeros((n, K))
    for s in range(NbInit):
        if Initoption == 'random':
            C0 = Cluster.RandomInit(V, K)
        elif Initoption == 'plusplus':
            C0 = Cluster.PlusPlusInit(V, K)
        elif Initoption == 'smallDeg':
            C0 = Cluster.SmallDegInit(V, K, degrees)
        elif Initoption == 'smallplusplus':
            C0 = Cluster.SmallPlusPlusInit(V, K, degrees)
        Z1, C1, score1 = Cluster.CombKmeans(V, C0, Omax)
        if score1 < score:
            score = score1
            Z = Z1
            C = C1
    return Z, Z


def ProcessOCCAM(ll, V, sumA, degrees, Initoption='plusplus', NbInit=1):
    '''given the matrix of K leading eigenvectors V and 
    the associated eigenvalues ll, returns the community membership table'''
    (n, K) = V.shape
    ind = np.where(ll < 0)[0]
    L = ll
    L[list(ind)] = 0
    L = np.diag(L)
    X = np.dot(V, np.sqrt(L))
    alpha = sumA / float(n * (n - 1) * K)
    tau = 0.1 * alpha ** (0.2) * K ** (1.5) / (n ** (0.3))
    for i in range(n):
        x = X[i, :]
        X[i, :] = x / (la.norm(x) + tau)
    score = 100000
    Cest = np.zeros((K, K))
    C0 = np.zeros((K, K))
    for k in range(NbInit):
        if Initoption == 'random':
            C0 = Cluster.RandomInit(V, K)
        elif Initoption == 'plusplus':
            C0 = Cluster.PlusPlusInit(V, K)
        elif Initoption == 'smallDeg':
            C0 = Cluster.SmallDegInit(V, K, degrees)
        elif Initoption == 'smallplusplus':
            C0 = Cluster.SmallPlusPlusInit(V, K, degrees)
        z, C, score1 = Cluster.Kmedians(X, K, C0)
        if score1 < score:
            Cest = C
            score = score1
    Ztmp = np.dot(X, la.inv(Cest))
    ZLev = np.zeros((n, K))
    for i in range(n):
        z = Ztmp[i, :]
        z2 = z / (la.norm(z))
        indices = list(np.where(z2 > 1 / float(K))[0])
        if (len(indices) == 0):
            ind = np.argmax(z2)
            ZLev[i, ind] = 1
        else:
            ZLev[i, indices] = 1
    return ZLev, Ztmp


# Performance measures

def evaluation(Z1, Z2):
    '''
    Z1 and Z2 two membership matrices with Z1 = Zest and Z2 = Ztrue
    returns
    fpos, false positive rate = number of 1 in Z1 not in Z2 /  number of 1 in Z2 (you add a community)
    fneg, false negative rate = number of 0 in Z1 not in Z2 / number of 0 in Z2  (you forget a community)
    err, error rate = number of different entries between Z1 and Z2 / number of entries in Z1
    perm, permutation of the label in Z1 necessary (based on min error)
    '''
    (n, K) = np.shape(Z1)
    diff = 10000000
    perm = list()
    for sigma in itertools.permutations(range(K)):
        indices = list(sigma)
        # form the associated permutation matrix
        MP = np.zeros((K, K))
        MP[range(K), indices] = 1
        # compute the Froebenius norm of the difference
        mat = np.dot(Z1, MP) - Z2
        diff1 = np.sum(np.absolute(mat))
        if diff1 < diff:
            diff = diff1
            perm = indices
    MP = np.zeros((K, K))
    MP[range(K), perm] = 1
    mat = np.dot(Z1, MP) - Z2
    fpos = np.sum(np.maximum(mat, np.zeros((n, K), int))) / np.sum(Z2)
    fneg = np.sum(np.maximum(-mat, np.zeros((n, K), int))) / (n * K - np.sum(Z2))
    err = diff / float(n * K)
    return fpos, fneg, err, perm


def evaluationVector(v1, v2):
    '''
    v1 and v2 are two arrays of same length, with entries indicating communities in O,K1 and 0,K2
    returns nb of nodes that are misclassified
    '''
    c1 = np.max(v1) + 1  # number of communities in v1
    c1 = int(c1)
    c2 = np.max(v2) + 1  # number of communities in v2
    c2 = int(c2)
    # permutations of the vector with the largest number of communities
    if (c1 >= c2):
        misc = 10000000
        # listP=[l for l in itertools.permutations(range(c1))]
        # for k in range(len(listP)):
        for sigma in itertools.permutations(range(c1)):
            # sigma=np.array(listP[k])
            sigma = np.array(sigma)
            v = sigma[list(v1)]
            index = np.where(v != v2)[0]
            misc1 = len(index)
            if misc1 < misc:
                misc = misc1
    else:
        misc = 10000000
        for sigma in itertools.permutations(range(c2)):
            sigma = np.array(sigma)
            v = sigma[list(v2)]
            index = np.where(v != v1)[0]
            misc1 = len(index)
            if misc1 < misc:
                misc = misc1
    return misc


def exNVI(Z1, Z2):
    '''compute exNVI for two n by K matrices with binaries entries'''
    (n, K) = Z1.shape
    e = np.ones(K)
    s2 = np.sum(Z2, axis=0)
    H2 = np.zeros(K)
    for l in range(K):
        if (s2[l] > 0 and s2[l] < n):
            p2 = 1. * s2[l] / n
            H2[l] = -p2 * log(p2) - (1 - p2) * log(1 - p2)
    nvi = 0.
    perm = list(range(K))
    for sigma in itertools.permutations(range(K)):
        indices = list(sigma)
        MP = np.zeros((K, K))
        MP[range(K), indices] = 1
        Z1P = np.dot(Z1, MP)
        s1 = np.sum(Z1P, axis=0)
        t0 = np.sum(Z1P * Z2, axis=0)  # i : z1_i,z2_i=(1,1)
        t1 = np.sum(Z1P - Z1P * Z2, axis=0)  # i : z1_i,z2_i=(1,0)
        t2 = np.sum(Z2 - Z1P * Z2, axis=0)  # i : z1_i,z2_i=(0,1)
        t3 = n * e - np.sum(Z1P + Z2 - Z1P * Z2, axis=0)  # i : z1_i,z2_i=(0,0)
        sums = 0.
        for l in range(K):
            H1 = 0
            if (s1[l] > 0 and s1[l] < n):
                p1 = 1. * s1[l] / n
                H1 = -p1 * log(p1) - (1 - p1) * log(1 - p1)
            H = 0  # joint entropy
            if (t0[l] > 0):
                p0 = 1. * t0[l] / n
                H += -p0 * log(p0)
            if (t1[l] > 0):
                p1 = 1. * t1[l] / n
                H += -p1 * log(p1)
            if (t2[l] > 0):
                p2 = 1. * t2[l] / n
                H += -p2 * log(p2)
            if (t3[l] > 0):
                p3 = 1. * t3[l] / n
                H += -p3 * log(p3)
            if (s1[l] > 0 and s1[l] < n):
                sums += (H - H2[l]) / H1
            if (s2[l] > 0 and s2[l] < n):
                sums += (H - H1) / H2[l]
        nvi1 = 1. - sums / (float(2 * K))
        if (nvi1 > nvi):
            nvi = nvi1
            perm = indices
    return nvi, perm


def ENMI(Z1, Z2):
    '''compute ENMI for two n by K matrices with binaries entries'''
    (n, K) = Z1.shape
    # compute entropy 
    s1 = np.sum(Z1, axis=0)
    H1 = np.zeros(K)
    for l in range(K):
        if (s1[l] > 0 and s1[l] < n):
            p1 = 1. * s1[l] / n
            H1[l] = -p1 * log(p1) - (1 - p1) * log(1 - p1)
    #print(H1)
    s2 = np.sum(Z2, axis=0)
    H2 = np.zeros(K)
    for l in range(K):
        if (s2[l] > 0 and s2[l] < n):
            p2 = 1. * s2[l] / n
            H2[l] = -p2 * log(p2) - (1 - p2) * log(1 - p2)
    #print(H2)
    # compute the joint entropy
    Joint = np.zeros((K, K))
    Condition = np.ones((K, K))
    for k in range(K):
        for l in range(K):
            Xk = Z1[:, k]
            Yl = Z2[:, l]
            p = np.zeros(4)
            p[0] = np.sum(Xk * Yl) / float(n)  # (1,1)
            p[1] = np.sum((1 - Xk) * (1 - Yl)) / float(n)  # (0,0)
            p[2] = np.sum((Xk) * (1 - Yl)) / float(n)  # (1,0)
            p[3] = 1 - p[0] - p[1] - p[2]  # (0,1)
            L = np.zeros(4)
            for i in range(4):
                if (p[i] > 0):
                    L[i] = -p[i] * m.log(p[i])
            if (L[0] + L[1] < L[2] + L[3]):
                Condition[k, l] = 0
            Joint[k, l] = np.sum(L)
    #print(Joint)
    #print(Condition)
    # compute ENMI
    enmi = 0.
    for k in range(K):
        indices = list(np.where(Condition[k, :] == 1)[0])
        if (len(indices) == 0):
            enmi += 1
        else:
            J = Joint[k, indices]
            if H1[k] > 0:
                enmi += np.min(J - H2[indices]) / H1[k]
        indices = list(np.where(Condition[:, k] == 1)[0])
        if (len(indices) == 0):
            enmi += 1
        else:
            J = Joint[indices, k]
            if H2[k] > 0:
                enmi += np.min(J - H1[indices]) / H2[k]
    return 1 - (1 / float(2 * K)) * enmi


def Modularity(A, Z):
    '''compute the following quantity (NG-modularity) for communities (non necessarly overlapping) given by Z :  
    (number of internal edges - number of expected internal edges under the configuration model)
    /(total number of edges)'''
    (n, K) = Z.shape
    M = np.sum(A) / 2  # total number of edges
    mod = 0
    for k in range(K):
        indices = np.where(Z[:, k] == 1)[0]
        nk = len(indices)
        Ak = A[indices, :]
        degk = np.sum(Ak, axis=1)
        Matdegk = np.zeros((nk, nk))
        for i in range(nk):
            for j in range(nk):
                Matdegk[i, j] = degk[i] * degk[j]
        Ak = Ak[:, indices]
        nbinternal = np.sum(Ak) / 2  # nb of internal edges
        expectedinternal = (np.sum(Matdegk)) / (float(4 * M))
        mod += nbinternal - expectedinternal
    mod = mod / (M)
    return mod


# Visualization tools


def printResults(Zest, Z):
    (n, K) = np.shape(Z)
    AvgCom = np.sum(Z) / float(n)
    AvgComEst = np.sum(Zest) / float(n)
    n0 = np.sum(Z, axis=1)
    Omax = np.max(n0)
    n1 = np.sum(Zest, axis=1)
    OmaxEst = np.max(n1)
    fpos, fneg, err, perm = evaluation(Zest, Z)
    print("c (true, estimated) = %3.2f, %3.2f" % (AvgCom, AvgComEst))
    print("m (true, estimated) = %d, %d" % (Omax, OmaxEst))
    print("false positive, false negative, error = %4.2f, %4.2f, %4.2f" % (fpos, fneg, err))


def printCommunities(Z):
    (n, K) = Z.shape
    for k in range(K):
        C = list(np.where(Z[:, k] == 1)[0] + 1)
        print("community number %d is" % k)
        print(C)


# Pre-processing of a network


def PPComm(Z0, A0, p):
    ''' pre processing on the communities inspired by the paper by Levina et al 2014 : 
        iteratively drop communities whose pure nodes are less than p% of the network size'''
    (ninit, Kinit) = Z0.shape
    # remove nodes with zero degree
    degrees = np.sum(A0, axis=1)
    nuls = list(np.where(degrees == 0)[0])
    ToKeep = [i for i in range(ninit) if i not in nuls]
    Z0 = Z0[ToKeep, :]
    A0 = A0[ToKeep, :]
    A0 = A0[:, ToKeep]
    condition = True
    (ninit, Kinit) = Z0.shape
    n = ninit
    K = Kinit
    Z = Z0
    A = A0
    while condition:
        # compute the number of pure nodes in each community, and keep the smallest  
        NbPurs = np.zeros(K)
        nO = np.dot(Z, np.ones((K, 1)))
        indP = np.where(nO == 1)[0]
        for i in list(indP):
            k = np.argmax(Z[i, :])
            NbPurs[k] = NbPurs[k] + 1
        minPur = np.min(NbPurs)
        Krm = np.argmin(NbPurs)
        if (K > 1) & (minPur < p * n):
            KeepComm = list(range(K))
            KeepComm.remove(Krm)
            # remove pure nodes in community Krm            
            ToKeep = []
            for i in range(n):
                if not ((i in indP) & (Z[i, Krm] == 1)):
                    ToKeep.append(i)
            Z = Z[:, KeepComm]
            Z = Z[ToKeep, :]
            A = A[ToKeep, :]
            A = A[:, ToKeep]
            (n, K) = Z.shape
        else:
            # stop pre-processing
            condition = False
    return Z, A


def PPDegrees(n, A, Z, dmin):
    degrees = np.sum(A, axis=1)
    indices = np.where(degrees < dmin)[0]
    ToKeep = [i for i in range(n) if i not in indices]
    Z = Z[ToKeep, :]
    A = A[ToKeep, :]
    A = A[:, ToKeep]
    (n, K) = Z.shape
    # removing new nodes of zero degrees
    degrees = np.sum(A, axis=1)
    indices = np.where(degrees == 0)[0]
    ToKeep = [i for i in range(n) if i not in indices]
    Z = Z[ToKeep, :]
    A = A[ToKeep, :]
    A = A[:, ToKeep]
    return A, Z
