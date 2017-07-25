import scipy as sp
import scipy.sparse.linalg
from scipy.spatial import ConvexHull
import numpy as np
from cvxpy import *

eigs = sp.sparse.linalg.eigs

def get_U_L(matrix, k):
    '''
    eigenvalue decomposition of matrix
    k = n_clusters

    returns
    _______________________________________________
    U: nd.array with shape (n_nodes, n_clusters)
    L: nd.array with shape (k,)
    '''
    Lambda, U = eigs(matrix, k=k, which='LR')
    Lambda, U = np.real(Lambda), np.real(U)
    return U, Lambda


def get_Q(U, use_convex_hull = True):
    '''
    get positive semidefinite Q matrix of ellipsoid from U[i,:] vectors

    for any u_i in U[i,:] :

            abs(u_i.T * Q * u_i) <= 1

    if flag use_convex_Hull == True (default) then scipy.spatial.ConvexHull
    optimization is used to reduce a number of points in U[i,:]

    returns
    _______________________________________________
    Q: nd.array with shape (U.shape[1], U.shape[1])

    requires:
    cvxpy (http://www.cvxpy.org/en/latest/)
    scipy.spatial.ConvexHull
    '''
    n_nodes = U.shape[0]
    k = U.shape[1]
    Q = Semidef(n=k)
    
    if use_convex_hull:
        hull = ConvexHull(U)
        constraints = [abs(U[i, :].reshape((1, k)) * Q * U[i, :].reshape((k, 1))) <= 1 for i in hull.vertices]
    else:
        constraints = [abs(U[i, :].reshape((1, k)) * Q * U[i, :].reshape((k, 1))) <= 1 for i in range(n_nodes)]
        
    obj = Minimize(-log_det(Q))
    prob = Problem(obj, constraints)
    _ = prob.solve(SCS, max_iters=100)
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
    new_U: nd.array with shape == U.shape
    transform_matrix: nd.array with shape == Q.shape
    '''
    L, S = sp.linalg.eig(Q)
    S, L = np.real(S), np.diag(np.real(L))
    transform_matrix = (S.T).dot(np.sqrt(L))
    new_U = np.dot(transform_matrix, U.T).T

    if return_transform_matrix == True:
        return new_U, transform_matrix
    else:
        return new_U


def get_F_B(U, Lambda, use_ellipsoid=True, use_convex_hull=True):

    '''
    compute F and B matrices from U matrix and L eigenvalues
    
    If use_ellipsoid == True then ellipsoid transformation 
    is used to transform coordinates
    
    Else basic method is used.
    
    Flag use_convex_hull is only for ellipsoid transformation
    (check get_Q function)

    U.shape[0] is a number of nodes
    U.shape[1] is a number of clusters

    returns
    _______________________________________________
    F: nd.array with shape (U.shape[1], U.shape[1])
    B: nd.array with shape == F.shape
    '''
    k = U.shape[1]
    
    if use_ellipsoid:
        Q = get_Q(U, use_convex_hull)
        new_U, transform_matrix = transform_U(U, Q, return_transform_matrix=True)
    else:
        new_U = U.copy()

    ### find k the biggest vectors in u_i
    ### and define f_j = u_i for
    J = set()
    i = 0
    while i < k:
        j_indexes = np.argsort(-np.sum(new_U**2, axis=1))
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
    B = F.dot(np.diag(Lambda)).dot(F.T)
    B[B < 1e-8] = 0
    B[B > 1] = 1
    return F, B


def get_Theta(U, F, use_cvxpy = True, Lambda = None):
    '''
    function to find Theta from U and F
    via convex optimization in cvxpy or euclidean_proj_simplex

    use_cvxpy is a flag for selecting a way of finding Theta
    if use_cvxpy == True then Theta is computed via cvxpy
    else Theta is computed by basic algorithm

    Lambda: nd.array with shape (U.shape[1],) which
    contains eigenvalues

    returns
    _______________________________________________
    Theta: nd.array with shape (n_nodes, n_clusters)
    where n_nodes == U.shape[0], n_clusters == U.shape[1]

    requires: cvxpy (http://www.cvxpy.org/en/latest/)
    '''

    assert U.shape[1] == F.shape[0] == F.shape[1], "U.shape[1] != F.shape"
    n_nodes = U.shape[0]
    n_clusters = U.shape[1]

    if use_cvxpy:
        Theta = Variable(rows=n_nodes, cols=n_clusters)
        constraints = [sum_entries(Theta[i, :]) == 1 for i in range(n_nodes)]
        constraints += [Theta[i, j] >= 0 for i in range(n_nodes) for j in range(n_clusters)]
        obj = Minimize(norm(U - Theta * F, 'fro'))
        prob = Problem(obj, constraints)
        prob.solve()
        return np.array(Theta.value)

    else:
        assert Lambda is not None, "Lambda is None object"
        projector = F.T.dot(np.linalg.inv(F.dot(F.T)))
        theta = U.dot(projector)
        theta_simplex_proj = np.array([euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj


def SPOC(A, n_clusters, use_ellipsoid=True, use_convex_hull=True, use_cvxpy=True):
    '''
    Entire SPOC method
    '''
    U, Lambda = get_U_L(A, k=n_clusters)
    F, B = get_F_B(U, Lambda, use_ellipsoid, use_convex_hull)
    theta = get_Theta(U, F, use_cvxpy, Lambda)
    return theta, B

def SPOC_from_UL(U, Lambda, use_ellipsoid=True, use_convex_hull=True, use_cvxpy=True):
    '''
    @author: kslavnov
    modified: rushakov
    '''
    F, B = get_F_B(U, Lambda, use_ellipsoid, use_convex_hull)
    theta = get_Theta(U, F, use_cvxpy, Lambda)
    return theta, B

def SPOC_get_Z_from_UL(U, L, n_clusters, use_ellipsoid=True, use_convex_hull=True, use_cvxpy=True):
    '''
    @author: kslavnov
    modified: rushakov
    '''
    theta, B = SPOC_from_UL(U, L, use_ellipsoid, use_convex_hull, use_cvxpy)
    return 1.0 * (theta > 1 / n_clusters), theta

def SPOC_get_Z(A, n_clusters, use_ellipsoid=True, use_convex_hull=True, use_cvxpy=True):
    theta, B = SPOC(A, n_clusters, use_ellipsoid, use_convex_hull, use_cvxpy)
    return 1.0 * (theta > 1 / n_clusters)

def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w
