import sys
import pytest
import numpy as np
from spoc import generate_lda_model

def test_freq_matrix():
    X_0, P, W = generate_lda_model.gen_freq_matrix(
        N=10000,
        n=1000,
        p=50,
        k=0,
        alpha=(0.15, 0.15, 0.85),
        fact=True
    )
    assert X_0.shape == (1000, 50)


def test_document_topic_error():
    right_theta = np.array([[0, 1], [1, 0]])
    exp_theta = np.array([[1, 1], [1, 1]])

    error = generate_lda_model.find_topic_document_error(right_theta,
                                                 exp_theta)
    assert abs(error - np.sqrt(2)) < sys.float_info.epsilon