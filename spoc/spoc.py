# coding: utf-8

import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs, svds
from scipy.spatial import ConvexHull

from cvxpy import abs, log_det, sum, norm, Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
#from cvxpy.expressions.variables.semidef_var import Semidef

from spoc.equality_test import EqualityTester


class SPOC(object):
    """
    Entire classical SPOC method and bootstrap realization of SPOC algorithm

    Parameters
    ---------
    A:  nd.array with shape (n_nodes, n_nodes)
        Adjacency matrix of a graph.

    n_clusters: int
        Number of clusters.

    use_bootstrap: boolean, default value is False
        Whether to use bootstrap realization of SPOC.

    n_repetitions: int, default value is 30
        Number of repetitions in bootstrap.

    std_num: float, default value is 3
        Influence on size of ellipsoid.

    return_pure_nodes_indices: boolean, default value is False
        If True then also returns a list of indices of selected pretenders
        to be pure nodes.

    use_cvxpy: boolean, optional, default value is False
        Whether to use cvxpy optimization in order to compute Theta matrix.
        
    solver: string, default value is "CVXOPT",
        solver for cvxpy ellipsoid optimization

    use_ellipsoid: boolean, optional, default value is False
        If True then ellipsoid transformation is used to transform coordinates
        else basic method is used.

    use_convex_hull: boolean, optional, default value is False
        If True then scipy.spatial.ConvexHull optimization is used to reduce a
        number of points in U[i,:].

    use_averaging: boolean, optional, default value is False
        If True then averaging procedure is run as described in SPOC++ algorithm

    bootstrap_type: string, optional, default value is 'random_weights'
        If 'random_indices' then algorithm uses random indices
        If 'random_weights' then algorithm uses random weights (default).

    return_bootstrap_matrix: boolean, optional, default value is False
        If True then also returns bootstrap result.
    """

    def __init__(self, use_bootstrap=False, use_ellipsoid=False,
                 use_convex_hull=False, use_cvxpy=False, solver="SCS",#"CVXOPT",
                 use_averaging=False,
                 bootstrap_type='random_weights', n_repetitions=30,
                 std_num=3.0, return_bootstrap_matrix=False,
                 return_pure_nodes_indices=False):

        self.std_num = std_num
        self.use_cvxpy = use_cvxpy
        self.solver = solver
        self.n_repetitions = n_repetitions
        self.use_ellipsoid = use_ellipsoid
        self.use_bootstrap = use_bootstrap
        self.bootstrap_type = bootstrap_type
        self.use_convex_hull = use_convex_hull
        self.use_averaging = use_averaging
        self.return_bootstrap_matrix = return_bootstrap_matrix
        self.return_pure_nodes_indices = return_pure_nodes_indices

    def _update_params(self, **kwargs):
        """
        Parameters
        ---------
        kwargs: dict
            Set the parameters of this estimator;
            Valid parameter keys can be listed with get_params().
        """

        parameters = [
            'A', 'n_clusters', 'n_repetitions', 'std_num', 'use_bootstrap',
            'return_pure_nodes_indices', 'use_ellipsoid', 'bootstrap_type',
            'return_bootstrap_matrix', 'use_convex_hull', 'use_cvxpy'
        ]

        for param in parameters:
            setattr(self, param, kwargs.get(param, getattr(self, param)))

    def fit(self, A, n_clusters, sym=True, **kwargs):
        """
        Parameters
        ---------
        kwargs: dict
            Set the parameters of this estimator;
            Valid parameter keys can be listed with get_params().


        Returns
        -------
        Theta: nd.array with shape (n_nodes, n_clusters)

        B: nd.array with shape (n_clusters, n_clusters)

        J: list with selected nodes (check **kwargs)

        repeat_matrix_array: nd.array with shape
                             (n_repetitions, n_nodes, n_clusters)
            the result of bootstrap
        """

        self.A = A
        self.n_clusters = n_clusters
        self._update_params(**kwargs)

        self.A = 1.0 * self.A
        if sym:
            U, Lambda = self._get_U_L(self.A, self.n_clusters)
        else:
            U, Lambda, V = self._get_U_L_V(self.A, self.n_clusters)

        if self.use_bootstrap:
            C = self._calculate_C_from_UL(U, Lambda)
            U_mean, repeats = self._calculate_mean_cov_U(self.A, C)
            F, B, J = self._get_F_B_bootstrap(U, Lambda, repeats)
            Theta = self._get_Theta(U=U_mean, F=F)
            if self.return_bootstrap_matrix and self.return_pure_nodes_indices:
                return Theta, B, J, repeats
            elif self.return_bootstrap_matrix \
                    and self.return_pure_nodes_indices is False:
                return Theta, B, repeats
            elif self.return_bootstrap_matrix is False and \
                    self.return_pure_nodes_indices:
                return Theta, B, J
            else:
                return Theta, B
        else:
            F, B, J = self._get_F_B(U, Lambda)
            Theta = self._get_Theta(U, F)
            if self.return_pure_nodes_indices:
                return Theta, B, J
            else:
                return Theta, B

    @staticmethod
    def _get_U_L(matrix, n_clusters):
        """
        Eigenvalue decomposition of matrix

        Parameters
        ---------
        matrix: array-like with shape (n_nodes, n_nodes)
            Matrix which is decomposed

        n_clusters: int
            Number of n_clusters

        fix the signs of eigenvectors for reproducibility


        Returns
        -------
        U: nd.array with shape (n_nodes, n_clusters)

        L: nd.array with shape (n_clusters,)
        """

        Lambda, U = eigs(matrix, k=n_clusters, which='LR')
        Lambda, U = np.real(Lambda), np.real(U)

        for index in range(n_clusters):
            if U[0, index] < 0:
                U[:, index] = -1 * U[:, index]
        return U, Lambda
    
    @staticmethod
    def _get_U_L_V(matrix, n_clusters):
        """
        Singular value decomposition of matrix

        Parameters
        ---------
        matrix: array-like with shape (n_nodes_1, n_nodes_2)
            Matrix which is decomposed
            Two types of nodes -- extension of _get_U_L for non symmetric matrix

        n_clusters: int
            Number of n_clusters

        fix the signs of singular vectors for reproducibility


        Returns
        -------
        U: nd.array with shape (n_nodes_1, n_clusters)

        L: nd.array with shape (n_clusters,)
        
        V: nd.array with shape (n_nodes_2, n_clusters)
        """
        
        U, Lambda, V = svds(matrix, k=n_clusters, which='LM')
        return U, Lambda, V.T
        

    def _get_Q(self, U):
        """
        Get positive semidefinite Q matrix of ellipsoid from U[i,:] vectors
        for any u_i in U[i,:] : abs(u_i.T * Q * u_i) <= 1

        Parameters
        ---------
        U: array-like


        Returns
        -------
        Q: nd.array with shape (U.shape[1], U.shape[1])


        Requires:
        ---------
        cvxpy (http://www.cvxpy.org/en/latest/)
        scipy.spatial.ConvexHull
        """

        n_nodes = U.shape[0]
        k = U.shape[1]
        Q = Variable((k,k), symmetric=True)
        constraints = [Q >> 0]
        #Q = Variable((k,k), PSD=True)

        if self.use_convex_hull:
            hull = ConvexHull(U)
            constraints = [
                abs(U[i, :].reshape((1, k)) @ Q @ U[i, :].reshape((k, 1))) \
                <= 1 for i in hull.vertices
            ]
        else:
            constraints = [
                abs(U[i, :].reshape((1, k)) @ Q @ U[i, :].reshape((k, 1))) \
                <= 1 for i in range(n_nodes)
            ]

        obj = Minimize(-log_det(Q))
        prob = Problem(obj, constraints)
        _ = prob.solve(solver=self.solver)
        Q = np.array(Q.value)
        return Q

    @staticmethod
    def _transform_U(U, Q, return_transform_matrix=False):
        """
        Transform U[i,:] coordinates to new basis where ellipsoid is a sphere.

        Parameters
        ---------
        U: array-like with shape (n_nodes, n_clusters)

        Q: array-like with shape (n_clusters, n_clusters)
        _________


        Returns
        -------
        new_U: nd.array with shape == U.shape

        transform_matrix: nd.array with shape == Q.shape


        Notes
        ------
        Lambda = S.T * Q * S (), where S is basis transformation matrix
        e' = e*S (for basis vectors) coordinates in
        new basis = S^-1 * coordinates in old basis
        Q is symmetric matrix ==> S^-1 = S.T
        new_coord = S.T * old_coord
        in order to make Lambda = Identity it is necessary to multiply
        S.T to np.sqrt(L)
        """

        L, S = sp.linalg.eig(Q)
        L, S = np.diag(np.real(L)), np.real(S)
        transform_matrix = np.linalg.inv(S)
        new_U = np.dot(transform_matrix, U.T).T

        if return_transform_matrix:
            return new_U, transform_matrix
        else:
            return new_U

    def _get_F_B(self, U, Lambda):
        """
        Compute F and B matrices from U matrix and L eigenvalues.

        Parameters
        ---------
        U: array-like with shape (n_nodes, n_clusters)

        Lambda: n_clusters with shape (n_clusters,)
            Which contains k the most significant eigenvalues


        Returns
        -------
        F: nd.array with shape (n_clusters, n_clusters)

        B: nd.array with shape == F.shape

        J: list of pretenders to be pure nodes (check **kwargs argument)
        """

        k = U.shape[1]

        if self.use_ellipsoid:
            Q = self._get_Q(U)
            new_U, transform_matrix = self._transform_U(
                U, Q, return_transform_matrix=True)
        else:
            new_U = U.copy()

        # find k the biggest vectors in u_i
        # and define f_j = u_i for
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
            new_U = (np.diag([1.] * new_U.shape[1]) - u.dot(u.T) / \
                     (u.T.dot(u))).dot(new_U.T)
            new_U = new_U.T
            J.add(j_)
            i += 1

        if self.use_averaging:
            F = np.zeros((self.n_clusters, self.n_clusters))
            tester = EqualityTester()
            tester.fit(
                self.A,
                self.n_clusters,
                U=U,
                Lambda=Lambda
            )
            T, pvalues = tester.test(np.array(list(J)))
            for j in range(len(J)):
                # in the paper it demands that threshold leads to infinity
                # but in real life exact such growth function is not clear
                # thus to use a quantile seems to be a best way
                F[j] = np.mean(U[pvalues[j, :] < 0.05, :], axis=0)
        else:
            F = U[list(J), :]
        B = F.dot(np.diag(Lambda)).dot(F.T)
        B[B < 1e-8] = 0
        B[B > 1] = 1

        return F, B, list(J)

    def _get_Theta(self, U, F):
        """
        Function to find Theta from U and F via convex optimization in cvxpy
        or euclidean_proj_simplex

        Parameters
        ---------
        U: array-like with shape (n_nodes, n_clusters)

        F: nd.array with shape (n_clusters, n_clusters)
            with coordinates of k pretenders to be pure nodes


        Returns
        -------
        Theta: nd.array with shape (n_nodes, n_clusters)

        where n_nodes == U.shape[0], n_clusters == U.shape[1]


        Requires
        --------
        cvxpy (http://www.cvxpy.org/en/latest/)
        """

        assert U.shape[1] == F.shape[0] == F.shape[1], \
            "U.shape[1] != F.shape"

        n_nodes = U.shape[0]
        n_clusters = U.shape[1]

        if self.use_cvxpy:
            Theta = Variable(shape=(n_nodes, n_clusters))
            constraints = [
                sum(Theta[i, :]) == 1 for i in range(n_nodes)
            ]
            constraints += [
                Theta[i, j] >= 0 for i in range(n_nodes)
                for j in range(n_clusters)
            ]
            obj = Minimize(norm(U - Theta @ F, 'fro'))
            prob = Problem(obj, constraints)
            prob.solve()
            return np.array(Theta.value)
        else:
            projector = F.T.dot(np.linalg.inv(F.dot(F.T)))
            theta = U.dot(projector)
            theta_simplex_proj = np.array([
                self._euclidean_proj_simplex(x) for x in theta
            ])
            return theta_simplex_proj

    @staticmethod
    def _euclidean_proj_simplex(v, s=1):
        """
        Compute the Euclidean projection on a positive simplex
        Solves the optimisation problem (using the algorithm from [1]):
            min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0

        Parameters
        ----------
        v: nd.array with shape (n,)
           n-dimensional vector to project

        s: int, optional, default: 1,
           radius of the simplex


        Returns
        -------
        w: (n,) numpy array,
           Euclidean projection of v on the simplex


        Notes
        -----
        The complexity of this algorithm is in O(n log(n)) as it
        involves sorting v. Better alternatives exist for high-dimensional
        sparse vectors (cf. [1]). However, this implementation still easily
        scales to millions of dimensions.


        References
        ----------
        [1] Efficient Projections onto the .1-Ball
                for Learning in High Dimensions
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

    @staticmethod
    def _calculate_C_from_UL(U, Lambda):
        """
        Calculate C matrix from matrices U and Lambda

        Parameters
        ---------
        U: array-like with shape (n_nodes, n_clusters)

        Lambda: n_clusters with shape (n_clusters,)
            Which contains k the most significant eigenvalues


        Returns
        -------
        C: nd.array with shape (n_nodes, n_clusters)
        """

        return np.dot(U, np.diag(1 / Lambda))

    def _calculate_mean_cov_U(self, A, C):
        """
        Function to make bootstrap with U coordinates

        Parameters
        ---------
        A: array-like
            Adjacency matrix of a graph

        C: array-like
            check calculate_C function

        Returns
        -------
        U: nd.array with shape (n_nodes, n_clusters)
            Array with mean value of coordinates for each node. Mean value is
            calculated by averaging bootstrap coordinates

        repeat_matrix_array: nd.array with shape (n_repetitions,
                                                  n_nodes,
                                                  n_clusters)
            The result of bootstrap
        _______


        Notes

        n_repetitions: int, optional, default: 30
            Number of repetitions in bootstrap

        bootstrap_type: string, oprional, default: 'random_weights'
            Type on bootstrap realization of SPOC spoc;
            If 'random_indices' then algorithm use random indeces;
            If 'random_weights' then algorithm use random weights (default).
        """

        if A.shape[0] != C.shape[0]:
            raise ValueError("A.shape[0] != U.shape[0]")

        n_nodes = A.shape[0]
        n_clusters = C.shape[1]
        n_repetitions = self.n_repetitions

        if self.bootstrap_type == "random_weights":
            A = A.copy()
            if hasattr(A, "toarray"):
                A = A.toarray()

            repeat_matrix_array = np.zeros((n_repetitions,
                                            n_nodes,
                                            n_clusters))
            nonzeros = np.nonzero(A)
            non_zero_count = nonzeros[0].shape[0]
            for i in range(n_repetitions):
                A[nonzeros] = np.random.normal(loc=1, scale=1,
                                               size=(non_zero_count,))
                repeat_matrix_array[i] = np.dot(A, C)
        else:
            repeat_matrix_array = []
            for i in range(n_repetitions):
                bootstrap_indices = np.random.randint(low=0, high=n_nodes,
                                                      size=n_nodes)
                repeat_matrix_array.append(
                    np.dot(A[:, bootstrap_indices], C[bootstrap_indices, :]))
            repeat_matrix_array = np.array(repeat_matrix_array)

        U = np.mean(repeat_matrix_array, axis=0)
        return U, repeat_matrix_array

    def _get_F_B_bootstrap(self, U, Lambda, repeats):
        """
        Compute F and B matrices from U matrix and L eigenvalues

        Parameters
        ---------
        U: array-like with shape (n_nodes, n_clusters)

        Lambda: n_clusters with shape (n_clusters,)
            Which contains k the most significant eigenvalues


        Returns
        -------
        F: nd.array with shape (n_clusters, n_clusters)

        B: nd.array with shape == F.shape

        J: nd.array
            List of pretenders to be pure nodes (check **kwargs argument)
        """

        k = U.shape[1]
        # find k the biggest vectors in u_i
        # and define f_j = u_i for
        new_U = U.copy()
        J = []
        F = np.zeros((k, k))
        i = 0

        while i < k:
            j_ = np.argsort(-np.sum(new_U ** 2, axis=1))[0]
            J.append(j_)

            cov = np.cov(repeats[:, j_].T)
            v = U[j_, :]
            diffs = [v - u for u in U]
            indices = [
                index for (index, dif) in enumerate(diffs) \
                if dif.dot(np.linalg.inv(cov)).dot(dif.T) < self.std_num ** 2
            ]
            F[i, :] = U[indices].mean(axis=0)
            u = new_U[indices].mean(axis=0)
            u = u.reshape((u.shape[0], 1))
            new_U = (np.diag([1.] * new_U.shape[1]) - u.dot(u.T) /
                     (u.T.dot(u))).dot(new_U.T)
            new_U = new_U.T
            i += 1

        B = F.dot(np.diag(Lambda)).dot(F.T)
        B[B < 1e-8] = 0
        B[B > 1] = 1
        return F, B, J
