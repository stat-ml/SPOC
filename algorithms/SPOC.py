import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs
from scipy.spatial import ConvexHull

from cvxpy import abs, log_det, sum_entries, norm
from cvxpy.settings import SCS, CVXOPT
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
from cvxpy.expressions.variables.semidef_var import Semidef

class SPOC(object):
    '''
    Entire classical SPOC method and bootstrap realization of SPOC algorithm

    A: adjacency matrix of a graph
    n_clusters: number of clusters
    n_repetitions: number of repetitions in bootstrap
    std_num: influence on size of ellipsoid
    
    **kwargs:

    use_bootstrap: flag
        if use_bootstrap == True then bootstrap realization of SPOC is activated
        else Theta is run by classical algorithm 

    return_pure_nodes_indices: flag
        if True then also returns a list of indices of selected pretenders to be pure nodes
    
    use_cvxpy: flag
        if use_cvxpy == True then Theta is computed via cvxpy
        else Theta is computed by basic algorithm (euclidean_proj_simplex)

    use_ellipsoid: flag
        if True then ellipsoid transformation is used to transform coordinates else basic method is used.

    use_convex_hull: flag
        if True then scipy.spatial.ConvexHull optimization is used to reduce a number of points in U[i,:]

    bootstrap_type: flag
        if 'random_indices' then algorithm use random indeces
        if 'random_weights' then algorithm use random weights (default parametrs)

    return_bootstrap_matrix: flag
        if True then also returns bootstrap result

    returns
    _______________________________________________
    Theta: nd.array with shape (n_nodes, n_clusters)
    B: nd.array with shape (n_clusters, n_clusters)
    J: list with selected nodes (check **kwargs)

    '''
    __version__ = '1.0'

    def __init__(self, A = None, n_clusters = None, **kwargs):
        
        self.A = A        
        self.n_clusters = n_clusters
        
        self.std_num = kwargs.get('std_num', 3)
        self.use_cvxpy = kwargs.get('use_cvxpy', False)
        self.n_repetitions = kwargs.get('n_repetitions', 30)
        self.use_ellipsoid = kwargs.get('use_ellipsoid', False)
        self.use_bootstrap = kwargs.get('use_bootstrap', False)
        self.bootstrap_type = kwargs.get('bootstrap_type', 'random_weights')
        self.use_convex_hull = kwargs.get('use_convex_hull', False)
        self.return_bootstrap_matrix = kwargs.get('return_bootstrap_matrix', False)
        self.return_pure_nodes_indices = kwargs.get('return_pure_nodes_indices', False)
        
    def fit(self, **kwargs):

        self.A = kwargs.get('A', self.A)
        self.n_clusters = kwargs.get('n_clusters', self.n_clusters)
        
        self.std_num = kwargs.get('std_num', self.std_num)
        self.use_cvxpy = kwargs.get('use_cvxpy', self.use_cvxpy)
        self.use_ellipsoid = kwargs.get('use_ellipsoid', self.use_ellipsoid)
        self.use_bootstrap = kwargs.get('use_bootstrap', self.use_bootstrap)
        self.n_repetitions = kwargs.get('n_repetitions', self.n_repetitions)
        self.bootstrap_type = kwargs.get('bootstrap_type', self.bootstrap_type)
        self.use_convex_hull = kwargs.get('use_convex_hull', self.use_convex_hull)
        self.return_bootstrap_matrix = kwargs.get('return_bootstrap_matrix', self.return_bootstrap_matrix)
        self.return_pure_nodes_indices = kwargs.get('return_pure_nodes_indices', self.return_pure_nodes_indices)
        
        assert np.all(self.A != None) or self.n_clusters != None, "Parametrs A or n_clusters is None"
        self.A = 1.0 * self.A
        U, Lambda = self.get_U_L(self.A, self.n_clusters)

        if self.use_bootstrap == False:
            F, B, J = self.get_F_B(U, Lambda)
            theta   = self.get_Theta(U, F, Lambda)
            if self.return_pure_nodes_indices == True:
                return theta, B, J
            else:
                return theta, B
        else:
            C = self.calculate_C_from_UL(U, Lambda)
            U_mean, std, repeats = self.calculate_mean_cov_U(self.A, C, self.n_repetitions)
            F, B, J = self.get_F_B_bootstrap(U, Lambda, repeats, std_num=self.std_num)
            Theta = self.get_Theta(U=U_mean, F=F, Lambda = Lambda, use_cvxpy=False)
            if (self.return_bootstrap_matrix == True) and (self.return_pure_nodes_indices == True):
                return Theta, B, repeats, J
            elif (self.return_bootstrap_matrix == True) and (self.return_pure_nodes_indices == False):
                return Theta, B, repeats
            elif (self.return_bootstrap_matrix == False) and (self.return_pure_nodes_indices == True):
                return Theta, B, J
            else:
                return Theta, B



    def get_U_L(self, matrix, k):
        '''
        eigenvalue decomposition of matrix
        k = n_clusters

        fix the signs of eigenvectors for reproducibility

        returns
        _______________________________________________
        U: nd.array with shape (n_nodes, n_clusters)
        L: nd.array with shape (k,)
        '''
        Lambda, U = eigs(matrix, k=k, which='LR')
        Lambda, U = np.real(Lambda), np.real(U)

        for index in range(k):
            if U[0, index] < 0:
                U[:, index] = -1 * U[:, index]
        return U, Lambda

    def get_Q(self, U):
        '''
        get positive semidefinite Q matrix of ellipsoid from U[i,:] vectors

        for any u_i in U[i,:] :

                abs(u_i.T * Q * u_i) <= 1

        **kwargs:
        use_convex_hull: flag
        if True then scipy.spatial.ConvexHull optimization is used to reduce a number of points in U[i,:]

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

        if self.use_convex_hull == True:
            hull = ConvexHull(U)
            constraints = [abs(U[i, :].reshape((1, k)) * Q * U[i, :].reshape((k, 1))) <= 1 for i in hull.vertices]
        else:
            constraints = [abs(U[i, :].reshape((1, k)) * Q * U[i, :].reshape((k, 1))) <= 1 for i in range(n_nodes)]

        obj = Minimize(-log_det(Q))
        prob = Problem(obj, constraints)
        # _ = prob.solve(SCS, max_iters=100)
        _ = prob.solve()
        Q = np.array(Q.value)
        return Q

    def transform_U(self, U, Q, return_transform_matrix=False):
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

    def get_F_B(self, U, Lambda):
        '''
        compute F and B matrices from U matrix and L eigenvalues

        U: matrix (n,k) shape, n = number of nodes, k is number of clusters
        U.shape[0] is a number of nodes
        U.shape[1] is a number of clusters

        Lambda: nd.array with shape (U.shape[1],) which contains k the most significant eigenvalues

        **kwargs:
        use_ellipsoid: flag
        if True then ellipsoid transformation is used to transform coordinates else basic method is used.

        use_convex_hull: flag
        if True then scipy.spatial.ConvexHull optimization is used to reduce a number of points in U[i,:]
        (check get_Q function)
        Flag use_convex_hull is only for ellipsoid transformation

        return_pure_nodes_indices: flag
        if True then returns a list with indices of selected nodes which are pretenders to be pure nodes

        returns
        _______________________________________________
        F: nd.array with shape (k, k)
        B: nd.array with shape == F.shape
        J: list of pretenders to be pure nodes (check **kwargs argument)
        '''
        k = U.shape[1]

        if self.use_ellipsoid:
            Q = self.get_Q(U)
            new_U, transform_matrix = self.transform_U(U, Q, return_transform_matrix=True)
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

        return F, B, list(J)


    def get_Theta(self, U, F, Lambda = None, **kwargs):
        '''
        function to find Theta from U and F
        via convex optimization in cvxpy or euclidean_proj_simplex

        U: matrix (n,k) shape, n = number of nodes, k is number of clusters
        U.shape[0] is a number of nodes
        U.shape[1] is a number of clusters

        Lambda: nd.array with shape (k,) which contains k the most significant eigenvalues

        F: nd.array with shape (k, k) with coordinates of k pretenders to be pure nodes

        **kwargs:
        use_cvxpy: flag for selecting a way of finding Theta
        if use_cvxpy == True then Theta is computed via cvxpy
        else Theta is computed by basic algorithm (euclidean_proj_simplex)

        returns
        _______________________________________________
        Theta: nd.array with shape (n_nodes, n_clusters)
        where n_nodes == U.shape[0], n_clusters == U.shape[1]

        requires: cvxpy (http://www.cvxpy.org/en/latest/)
        '''
        assert U.shape[1] == F.shape[0] == F.shape[1], "U.shape[1] != F.shape"
        n_nodes = U.shape[0]
        n_clusters = U.shape[1]

        if self.use_cvxpy:
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
            theta_simplex_proj = np.array([self.euclidean_proj_simplex(x) for x in theta])
            return theta_simplex_proj

    def euclidean_proj_simplex(self, v, s=1):
        '''
        Compute the Euclidean projection on a positive simplex
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
        '''
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

    def calculate_C_from_UL(self, U, Lambda):
        '''
        calculate C matrix from matrices U and Lambda 

        U: matrix (n,k) shape, n = number of nodes, k is number of clusters
        U.shape[0] is a number of nodes
        U.shape[1] is a number of clusters

        Lambda: nd.array with shape (U.shape[1],) which contains k the most significant eigenvalues

        returns
        _______________________________________________
        C: nd.array with shape (n_nodes, n_clusters)
        '''
        return np.dot(U, np.diag(1 / Lambda))

    def calculate_mean_cov_U(self, A, C, n_repetitions = 30):
        '''
        function to make bootstrap with U coordinates

        A: adjacency matrix of a graph
        C: check calculate_C function
        n_repetitions: number of repetitions in bootstrap
        bootstrap_type: type on bootstrap realization of SPOC algorithms

        returns
        _______________________________________________
        U: nd.array with shape (n_nodes, n_clusters) with mean value of coordinates
        for each node. Mean value is calculated by averaging bootstrap coordinates

        std_array: std for every node which is calculated from bootstrap

        repeat_matrix_array: nd.array with shape (n_repetitions, n_nodes, n_clusters)
        the result of bootstrap
        '''
        assert A.shape[0] == C.shape[0], "A.shape[0] != U.shape[0]"
        n_nodes = A.shape[0]
        n_clusters = C.shape[1]

        if self.bootstrap_type == "random_weights":
            A = A.copy()
            if hasattr(A, "toarray"):
                A = A.toarray()
            
            repeat_matrix_array = np.zeros((n_repetitions, A.shape[0], C.shape[1]))
            nonzeros = np.nonzero(A)
            non_zero_count = nonzeros[0].shape[0]
            for i in range(n_repetitions):
                A[nonzeros] = np.random.normal(loc=1, scale=1, size=(non_zero_count,))
                repeat_matrix_array[i] = np.dot(A, C)
        else:
            repeat_matrix_array = []
            for i in range(n_repetitions):
                bootstrap_indices = np.random.randint(low=0, high=n_nodes, size=n_nodes)
                repeat_matrix_array.append(np.dot(A[:, bootstrap_indices], C[bootstrap_indices, :]))
            repeat_matrix_array = np.array(repeat_matrix_array) 

        U = np.mean(repeat_matrix_array, axis=0)
        std_array = []

        return U, np.array(std_array), repeat_matrix_array

    def get_F_B_bootstrap(self, U, Lambda, repeats, std_num = 3):
        '''
        compute F and B matrices from U matrix and L eigenvalues

        U: matrix (n,k) shape, n = number of nodes, k is number of clusters
        U.shape[0] is a number of nodes
        U.shape[1] is a number of clusters

        Lambda: nd.array with shape (U.shape[1],) which contains k the most significant eigenvalues

        **kwargs:
        use_ellipsoid: flag
        if True then ellipsoid transformation is used to transform coordinates else basic method is used.

        use_convex_hull: flag
        if True then scipy.spatial.ConvexHull optimization is used to reduce a number of points in U[i,:]
        (check get_Q function)
        Flag use_convex_hull is only for ellipsoid transformation

        return_pure_nodes_indices: flag
        if True then returns a list with indices of selected nodes which are pretenders to be pure nodes

        returns
        _______________________________________________
        F: nd.array with shape (k, k)
        B: nd.array with shape == F.shape
        J: list of pretenders to be pure nodes (check **kwargs argument)
        '''
        k = U.shape[1]
        ### find k the biggest vectors in u_i
        ### and define f_j = u_i for
        new_U = U.copy()
        J = []
        F = np.zeros((k, k))
        i = 0

        while i < k:
            j_ = np.argsort(-np.sum(new_U**2, axis=1))[0]
            J.append(j_)

            cov = np.cov(repeats[:, j_].T)
            v = U[j_, :]
            diffs = [v - u for u in U]
            indices = [index for (index, dif) in enumerate(diffs) if dif.dot(np.linalg.inv(cov)).dot(dif.T) < std_num**2]
            
            F[i,:] = U[indices].mean(axis=0)
            u = new_U[indices].mean(axis=0)
            u = u.reshape((u.shape[0], 1))
            new_U = (np.diag([1.] * new_U.shape[1]) - u.dot(u.T) / (u.T.dot(u))).dot(new_U.T)
            new_U = new_U.T
            i += 1
            
        B = F.dot(np.diag(Lambda)).dot(F.T)
        B[B < 1e-8] = 0
        B[B > 1] = 1
        return F, B, J

