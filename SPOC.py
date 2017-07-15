import scipy as sp
import scipy.sparse.linalg
import numpy as np
import seaborn as sns
from cvxpy import *

eigs = sp.sparse.linalg.eigs

def get_U_L(matrix, k):
    '''
    eigenvalue decomposition of symmetric matrix (U^-1 = U.T)
    matrix = U * L * U.T

    k = n_clusters

    returns
    _______________________________________________
    U: np.array with shape (n_nodes, n_clusters)
    L: np.array with shape (n_clusters, n_clusters), L is diagonal
    '''
    Lambda, U = eigs(matrix, k=k, which='LR')
    Lambda, U = np.diag(np.real(Lambda)), np.real(U)
    return U, Lambda


def get_Q(U):
    '''
    get positive semidefinite Q matrix of ellipsoid from U[i,:] vectors
    for any u_i in range(U[i,:], U.shape[0] <- number of nodes):

            abs(u_i.T * Q * u_i) < 1
    returns
    _______________________________________________
    Q: np.array with shape (U.shape[1], U.shape[1])

    requires: cvxpy (http://www.cvxpy.org/en/latest/)
    '''
    k = U.shape[1]
    Q = Semidef(n=k)
    constraints = [ abs( U[i,:].reshape((1,k))*Q*U[i,:].reshape((k,1)) ) < 1 for i in range(U.shape[0]) ]
    obj = Minimize(-log_det(Q))
    prob = Problem(obj, constraints)
    _ = prob.solve()
    Q = np.array(Q.value)
    return Q


def transform_U(U, Q, return_transform_matrix = False):
    '''
    U is matrix (n,k) shape, n = number of nodes, k is number of clusters
    transform U[i,:] coordinates to new basis where ellipsoid is a sphere

    Lambda = S.T * Q * S (Lambda is diagonal matrix (k,k)), where
    S is basis transformation matrix
    e' = e*S (for basis vectors)
    coordinates in new basis = S^-1 * coordinates in old basis
    Q is symmetric matrix ==> S^-1 = S.T
    new_coord = S.T * old_coord
    in order to make Lambda = Identity it is necessary to multiply S.T to np.sqrt(L)

    returns
    _______________________________________________
    new_U: np.array with shape == U.shape
    transform_matrix: np.array with shape == Q.shape
    '''
    L, S = sp.linalg.eig(Q)
    S, L = np.real(S), np.diag(np.real(L))
    transform_matrix = (S.T).dot(np.sqrt(L))
    new_U = np.dot(transform_matrix, U.T).T
    if return_transform_matrix == True:
        return new_U, transform_matrix
    else:
        return new_U


def get_F_B(U, Lambda, cut=True):

    '''
    compute F, B from U matrix
    via using get_Q() to find Q matrix of ellipsoid
    and using transform_U(U,Q) to calculate new coordinate of u_i vectors

    U.shape[0] is a number of nodes
    U.shape[1] is a number of clusters

    returns
    _______________________________________________
    F: np.array with shape (U.shape[1], U.shape[1])
    B: np.array with shape == F.shape
    '''
    k = U.shape[1]
    Q = get_Q(U)
    new_U, transform_matrix = transform_U(U, Q, return_transform_matrix=True)

    ### find k the biggest vectors in u_i
    ### and define f_j = u_i for
    J = set()
    i = 0
    while i < k:
        j_indexes = np.argsort(-np.sum(new_U ** 2, axis=1))
        r = 0
        j_ = j_indexes[r]
        while j_ in J:
            j_ = j_indexes[r]
            r += 1
        u = new_U[j_, :]
        u = u.reshape((u.shape[0], 1))
        new_U = (np.diag([1.] * new_U.shape[1]) - u.dot(u.T) / (u.T.dot(u))).dot(new_U.T)
        new_U = new_U.T
        J.add(j_)
        i += 1
    F = U[list(J), :]
    B = F.dot(Lambda).dot(F.T)
    B = (B - np.min(B))/(np.max(B) - np.min(B))
    return F, B


def get_Theta(U, F):
    '''
    function to find Theta from U and F
    via convex optimization in cvxpy

    returns
    _______________________________________________
    Theta: np.array with shape (n_nodes, n_clusters)
    where n_nodes == U.shape[0], n_clusters == U.shape[1]

    requires: cvxpy (http://www.cvxpy.org/en/latest/)
    '''

    assert U.shape[1] == F.shape[0] == F.shape[1], "U.shape[1] != F.shape"
    n_nodes = U.shape[0]
    n_clusters = U.shape[1]

    Theta = Variable(rows = n_nodes, cols = n_clusters)
    constraints = [sum_entries(Theta[i,:])==1 for i in range(n_nodes)]
    constraints  += [Theta[i,j] >= 0 for i in range(n_nodes) for j in range(n_clusters)]
    obj = Minimize(norm(U - Theta*F, 'fro'))
    prob = Problem(obj, constraints)
    prob.solve()
    return np.array(Theta.value)

def basic_get_Theta(U, L, F):
    '''
    basic function to find Theta from U, L and F
    via ordinary least squares

    returns
    _______________________________________________
    Theta: np.array with shape (n_nodes, n_clusters)
    '''

    assert U.shape[1] == F.shape[0] == F.shape[1], "U.shape[1] != F.shape"
    Theta: np.array
    projector = F.T.dot(np.linalg.inv(F.dot(F.T)))
    Theta = U.dot(projector)
    return Theta
