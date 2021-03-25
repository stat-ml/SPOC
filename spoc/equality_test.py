import numba
import numpy as np
from scipy.sparse.linalg import eigs
import scipy.stats as sps

@numba.jit(nopython=True, parallel=True)
def equality_statistic(U, W_var, Lambda, n_nodes, simplex_vertices):
    '''
    Parameters:
    U: n x K matrix, where U[:, k] is the k-th eigenvector of A
    W_var: estimator of element-wise variance of W
    Lambda: estimator of eigenvalues
    n_nodes: number of nodes

    Return:
    T: K x n values of statistic for every simplex_vertex and every node
    pavlue: pvalue of corresponded test
    '''
    statistic_matrix = np.zeros((n_nodes, len(simplex_vertices)))
    U_T = U.T.copy()

    for i in numba.prange(n_nodes):
        for j, vertex_index in enumerate(simplex_vertices):
            W_var_diag = (W_var[i, :] + W_var[vertex_index, :]).copy()
            v_i = U[i, :].copy().reshape(-1, 1)
            v_j = U[vertex_index, :].copy().reshape(-1, 1)
            difference = (Lambda * (U[i, :] - U[vertex_index, :])).copy().reshape(-1, 1)

            # numba's @ is faster on contiguous arrays
            difference_T = difference.T.copy()
            v_i_T = v_i.T.copy()
            v_j_T = v_j.T.copy()

            positive_part = U_T @ (U * W_var_diag.reshape(-1, 1))
            Sigma = positive_part - W_var[i, vertex_index] * (v_j @ v_i_T + v_i @ v_j_T)
            Sigma_inv = np.linalg.inv(Sigma)
            statistic_matrix[i, j] = (
                difference_T @ Sigma_inv @ difference
            )[0, 0]
    return statistic_matrix


class EqualityTester:

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


    def __init__(self):
        pass


    def fit(self, A, n_clusters, U=None, Lambda=None):
        """
        Gets an adjacency matrix A and number of clusters K

        Parameters:
        -------------
        A: agjacency matrix
        n_clusters: number of clusters
        U: (optional) eigenvectors of A
        Lambda: (optional) first n_clusters of eigenvalues of A
        """
        self.n_clusters = n_clusters

        # get rank-K spectral decomposition
        if (type(U) == type(None)):
            self.U, self.Lambda = self._get_U_L(A, n_clusters)
        else:
            self.U = U
            self.Lambda = Lambda

        # we need only diag of ordinary noise matrix estimator
        noise_0 = np.diag(
            A - self.U @ (
                self.Lambda.reshape(-1, 1) * self.U.T
            )
        )

        # to correct eigenvalues
        correction = np.sum(self.U * (np.diag(noise_0) @ self.U), axis=0)
        self.Lambda = 1 / (1 / self.Lambda + correction / self.Lambda ** 3)

        # correct noise matrix estimator
        W = A - self.U @ (self.U.T * self.Lambda.reshape(-1, 1))

        # estimate element-wise variance of W 
        self.W_var = W ** 2


    def test(self, simplex_vertices):
            """
            Run equality test for each vertex from simplex_vertices and each node

            Parameters:
            ------------
            simplex_vertices: np.array of vertices which test is run for

            Returns:
            -----------
            statistic: K x n values of statistic
            pvalue: K x n array of pvalues
            """
            T = equality_statistic(
                self.U, 
                self.W_var,
                self.Lambda, 
                self.U.shape[0],
                simplex_vertices
            ).T
            return T, 1 - sps.chi2(self.n_clusters).cdf(T)


    