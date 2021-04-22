import pytest
from spoc.equality_test import EqualityTester
from spoc import generate_spoc_model
import numpy as np

@pytest.fixture()
def A():
    n_nodes = 1000
    n_clusters = 5
    pure_nodes = 100
    A, _, _ = generate_spoc_model.generate_a(n_nodes, n_clusters, pure_nodes)
    return A 

def test_fit(A):
    tester = EqualityTester()
    tester.fit(A, 5)
    n_nodes = A.shape[0]
    assert tester.W_var.shape == (n_nodes, n_nodes)
    assert tester.Lambda.shape == (5, )

def test_statistic(A):
    tester = EqualityTester()
    n_nodes = A.shape[0]
    tester.fit(A, 5)
    T, pvalue = tester.test(
        np.array([1, 2, 3, 4, 5])
    )
    assert T.shape == (5, n_nodes)
    assert pvalue.shape == (5, n_nodes)
