"""
SPOC method realization
@author: kslavnov

You need to refactor this.
Make a class and etc.
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

eigs = sp.sparse.linalg.eigs

S2 = None
axs = []
NEW = False
Draw = False  # Only for data with 2 or 3 communities
dims = 2 # or 3 for drawing


def get_F_B(U, Lambda, r, cut=True):

    if NEW:
        X = U
    else:
        X = U.dot(np.sqrt(np.diag(Lambda)))
    R = X.T
    J = set()
    i = 0
    col = None
    while i < r:
        if Draw:
            drawR(i, R)
        j_indexes = np.argsort(-np.sum(R ** 2, axis=0))
        k = 0
        j_ = j_indexes[k]
        while j_ in J:
            j_ = j_indexes[k]
            k += 1
        u = R[:, j_]
        u = u.reshape(u.shape[0], 1)
        R = (np.diag([1.] * R.shape[0]) - u.dot(u.T) / (u.T.dot(u))).dot(R)
        J.add(j_)
        i += 1

    F = X[list(J), :]

    if Draw:
        drawF(F)

    if NEW:
        B = F.dot(np.diag(Lambda).dot(F.T))  # TODO: optimise calc
    else:
        B = F.dot(F.T)
    if np.any(B < 0):
        pass
        # print('I need to cut B!')

    if cut:
        B[B < 1e-8] = 0
        B[B > 1] = 1

    return F, B


def get_Theta(U, L, F):
    if NEW:
        X = U
    else:
        X = U.dot(np.diag(np.sqrt(L)))

    projector = F.T.dot(np.linalg.inv(F.dot(F.T)))
    theta = X.dot(projector)
    return theta


def get_U_L(matrix, k):
    Lambda, U = eigs(1.0 * matrix, k, which="LR")
    Lambda, U = np.real(Lambda), np.real(U)
    U = np.sign(U[0, :])[None, :] * U
    return U, Lambda


def SPOC(A, n_clusters, Theta_true=None):
    if Theta_true is not None and Draw:
        n = A.shape[0]
        global S2
        S2 = np.hstack((Theta_true, np.array([(0, 0.8)] * n)))

    U, Lambda = get_U_L(A, k=n_clusters)
    F, B = get_F_B(U, Lambda, n_clusters)
    theta = get_Theta(U, Lambda, F)
    theta_simplex_proj = np.array([euclidean_proj_simplex(x) for x in theta])
    return theta_simplex_proj, B

def SPOC_from_UL(U, Lambda, n_clusters, Theta_true=None):
    if Theta_true is not None and Draw:
        n = U.shape[0]
        global S2
        S2 = np.hstack((Theta_true, np.array([(0, 0.8)] * n)))

    F, B = get_F_B(U, Lambda, n_clusters)
    theta = get_Theta(U, Lambda, F)
    theta_simplex_proj = np.array([euclidean_proj_simplex(x) for x in theta])
    return theta_simplex_proj, B


def SPOC_get_Z_from_UL(U, L, n_clusters, Theta_true, new=True):
    global NEW
    NEW = new
    theta, B = SPOC_from_UL(U, L, n_clusters, Theta_true)
    return 1.0 * (theta > 1 / n_clusters), theta

def SPOC_get_Z(A, n_clusters, Theta_true, new=True):
    global NEW
    NEW = new
    theta, B = SPOC(A, n_clusters, Theta_true)
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


def drawF(F):
    if dims == 2:
        axs[0].scatter(F[:, 0], F[:, 1], c='k', s=25, marker='*')
    else:
        axs[0].scatter(F[:, 0], F[:, 1], F[:, 2], c='k', s=200, )


def drawR(i, R):
    global axs
    if i >= len(axs):
        fig = plt.figure(i + 1)
        if dims == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
        axs.append(ax)
        max_size = 20
        col = None #'k'
    else:
        ax = axs[i]
        max_size = 100
    sizes = np.sum(R ** 2, axis=0)
    sizes -= np.min(sizes)
    sizes /= np.max(sizes) + 1e-6
    sizes *= max_size
    sizes += 5
    if dims == 2:
        ax.scatter(1e-16 + np.abs(R[0]), 1e-16 + np.abs(R[1]), c=S2 if col is None else col, s=sizes)
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
    else:
        ax.scatter(R[0], R[1], R[2], c=S2 if col is None else col, s=sizes)
        ax.scatter(0, 0, 0, c='k', s=100)
        ax.scatter(0, 0, c=(0.5, 0.5, 0.5, 0.5), s=100)
    plt.tight_layout(pad=1.02)
    plt.draw()
    plt.hold('on')
