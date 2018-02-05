import pytest
from spoc import spoc, generate_spoc_model


@pytest.fixture()
def get_A():
    n_nodes = 300
    n_clusters = 5
    pure_nodes = 10
    A, _, _ = generate_spoc_model.generate_a(n_nodes, n_clusters, pure_nodes)
    return A


def test_simple():
    A = get_A()
    spoc_obj = spoc.SPOC(A, 5)
    theta, b = spoc_obj.fit()
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)


def test_ellipsoid():
    A = get_A()
    spoc_obj = spoc.SPOC(A, 5, use_ellipsoid=True)
    theta, b = spoc_obj.fit()
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)


def test_use_cvxpy():
    A = get_A()
    spoc_obj = spoc.SPOC(A, 5, use_cvxpy=True)
    theta, b = spoc_obj.fit()
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)


def test_use_convex_hull():
    A = get_A()
    spoc_obj = spoc.SPOC(A, 5, use_convex_hull=True)
    theta, b = spoc_obj.fit()
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)


def test_bootstrap():
    A = get_A()
    spoc_obj = spoc.SPOC(A, 5, use_bootstrap=True)
    theta, b = spoc_obj.fit()
    assert theta.shape == (300, 5)
    assert b.shape == (5, 5)
