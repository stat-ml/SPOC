import sys
import pytest
import numpy as np
from spoc import generate_spoc_model


def test_theta_error_zero():
    right_theta = np.array([[0, 1], [1, 0]])
    exp_theta = np.array([[1, 0], [0, 1]])
    error, perm = generate_spoc_model.find_permutation_Theta(right_theta,
                                                             exp_theta)
    assert abs(error - 0.0) < sys.float_info.epsilon
    assert np.allclose(perm, np.array([[0, 1], [1, 0]]))


def test_theta_error():
    right_theta = np.array([[0, 1], [1, 0]])
    exp_theta = np.array([[1, 1], [1, 1]])

    error, perm = generate_spoc_model.find_permutation_Theta(right_theta,
                                                             exp_theta)
    assert abs(error - 1.0) < sys.float_info.epsilon
    assert np.allclose(perm, np.array([[1, 1], [1, 1]]))


def test_b_error_zero():

    right_b = np.diag([1, 2, 3, 4, 5])
    exp_b = np.diag([3, 2, 1, 5, 4])

    error, perm = generate_spoc_model.find_permutation_B(right_b, exp_b)
    assert abs(error - 0.0) < sys.float_info.epsilon
    assert np.allclose(perm, right_b)


def test_create_pure_node_row():

    with pytest.raises(AssertionError):
        generate_spoc_model.create_pure_node_row(5, 5)

    first_row = generate_spoc_model.create_pure_node_row(5, 0)
    right_row = np.array([1, 0, 0, 0, 0])

    assert np.allclose(first_row, right_row)

    second_row = generate_spoc_model.create_pure_node_row(10, 1)
    right_row = np.array([0, 1] + [0] * 8)

    assert np.allclose(second_row, right_row)


def test_generate_theta():
    with pytest.raises(AssertionError):
        generate_spoc_model.generate_theta(3, 5, 1)

    with pytest.raises(AssertionError):
        generate_spoc_model.generate_theta(10, 5, 11)

    shape_theta = generate_spoc_model.generate_theta(1145, 13, 1)
    assert shape_theta.shape == (1145, 13)

    right_theta = np.diag([1] * 5)

    generated_theta = generate_spoc_model.generate_theta(5, 5, 5)
    err, perm =  generate_spoc_model.find_permutation_Theta(right_theta,
                                                            generated_theta)
    assert (err - 0.0) < sys.float_info.epsilon


def test_generate_theta_pure_node_indices():

    pure_rows_theta = np.array([[1, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 0, 1, 0, 0]])

    theta = generate_spoc_model.generate_theta(10, 5, 3,
                                               pure_nodes_indices=[4, 5, 6])
    assert np.allclose(pure_rows_theta, theta[4:7])


def test_generate_theta_seed():

    theta_1 = generate_spoc_model.generate_theta(10, 5, 3, seed=0)
    theta_2 = generate_spoc_model.generate_theta(10, 5, 3, seed=0)
    assert np.allclose(theta_1, theta_2)

    theta_1 = generate_spoc_model.generate_theta(100, 10, 15, seed=42)
    theta_2 = generate_spoc_model.generate_theta(100, 10, 15, seed=42)
    assert np.allclose(theta_1, theta_2)

    theta_1 = generate_spoc_model.generate_theta(100, 10, 15, seed=42)
    theta_2 = generate_spoc_model.generate_theta(100, 10, 15, seed=43)
    assert (np.allclose(theta_1, theta_2) is False)


def test_generate_p_shape():
    p, theta, b = generate_spoc_model.generate_p(100, 15, 1)
    assert p.shape == (100, 100)
    assert theta.shape == (100, 15)
    assert b.shape == (15, 15)
    assert np.allclose(p, theta.dot(b).dot(theta.T))


def test_generate_p_seed():
    first_p, first_theta, first_b = generate_spoc_model.generate_p(200, 30, 5,
                                                                   seed=42)

    sec_p, sec_theta, sec_b = generate_spoc_model.generate_p(200, 30,
                                                             5, seed=42)

    assert np.allclose(first_p, sec_p)
    assert np.allclose(first_theta, sec_theta)
    assert np.allclose(first_b, sec_b)
    assert np.allclose(first_p, first_theta.dot(first_b).dot(first_theta.T))

    sec_p, sec_theta, sec_b = generate_spoc_model.generate_p(200, 30,
                                                             5, seed=43)
    assert (np.allclose(first_p, sec_p) is False)


def test_generate_p_pure_indices():
    p, theta, b = generate_spoc_model.generate_p(
        100, 5, 5, pure_nodes_indices=[30, 31, 32, 33, 34])

    assert np.allclose(theta[30:35], np.diag([1] * 5))
    assert np.allclose(p, theta.dot(b).dot(theta.T))


def test_p_2_a_seed():
    P, Theta, B = generate_spoc_model.generate_p(500, 15, 10)
    first_A = generate_spoc_model.P_to_A(P, seed=32)
    second_A = generate_spoc_model.P_to_A(P, seed=32)
    assert np.allclose(first_A, second_A)

    second_A = generate_spoc_model.P_to_A(P, seed=42)
    assert (np.allclose(first_A, second_A) is False)


def test_p_to_a():
    P, Theta, B = generate_spoc_model.generate_p(100, 15, 5)
    A = generate_spoc_model.P_to_A(P)
    assert np.allclose(A, A.T)





