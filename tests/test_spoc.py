import pytest
from spoc import spoc, generate_spoc_model


@pytest.fixture()
def A():
    n_nodes = 300
    n_clusters = 5
    pure_nodes = 10
    A, _, _ = generate_spoc_model.generate_a(n_nodes, n_clusters, pure_nodes)
    return A


def test_simple(A):
    spoc_obj = spoc.SPOC()
    theta, b = spoc_obj.fit(A, 5)
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)


def test_ellipsoid(A):
    spoc_obj = spoc.SPOC(use_ellipsoid=True)
    theta, b = spoc_obj.fit(A, 5)
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)


def test_use_cvxpy(A):
    spoc_obj = spoc.SPOC(use_cvxpy=True)
    theta, b = spoc_obj.fit(A, 5)
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)


def test_use_convex_hull(A):
    spoc_obj = spoc.SPOC(use_convex_hull=True)
    theta, b = spoc_obj.fit(A, 5)
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)


def test_bootstrap(A):
    spoc_obj = spoc.SPOC(use_bootstrap=True)
    theta, b = spoc_obj.fit(A, 5)
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)

def test_averaging(A):
    spoc_obj = spoc.SPOC(use_averaging=True)
    theta, b = spoc_obj.fit(A, 5)
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)
