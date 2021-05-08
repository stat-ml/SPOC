import pytest
from spoc import spoc, generate_spoc_model, generate_lda_model


@pytest.fixture()
def A():
    n_nodes = 300
    n_clusters = 5
    pure_nodes = 10
    A, _, _ = generate_spoc_model.generate_a(n_nodes, n_clusters, pure_nodes)
    return A


@pytest.fixture()
def A_asym():
    return generate_lda_model.gen_freq_matrix(
        N=10000,
        n=1000,
        p=50,
        k=0,
        alpha=(0.15,0.15,0.85)
    )


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


def test_asym(A_asym):
    spoc_obj = spoc.SPOC(model_type='topic_plsi')
    theta, freq_matrix = spoc_obj.fit(A_asym, 3)
    assert theta.shape == (1000, 3)
    assert freq_matrix.shape == (3, 50)