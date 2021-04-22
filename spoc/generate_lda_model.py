import numpy as np
import scipy as sp
import scipy.stats as sps
from itertools import permutations


def dirichlet_matrix(size, alpha):
    """
    Generates a matrix with independent rows from Dirichlet distribution

    Parameters:
    alpha --- the parameter of Dirichlet distribution

    ------------
    Return:
    A matrix of shape (size, len(alpha))
    """
    matD = sps.dirichlet(alpha).rvs(size=(size, ))
    return matD



def topic_document(n,k=0,alpha=0):
    """
    For len(alpha) topics generates n distributions of topics conditioned on a document

    It is supposed that P(topic|document) doesn't depend on document, and for each document 
    topic|document ~ Dirichlet(alpha)

    Parameters:
    n --- the number of documents
    k --- the number of topics if k != 0
    alpha --- the parameter of Dirichlet distribution, if 0 then rows are generated 
    from uniform distribution on simplex

    -----------
    Return
    A matrix of shape (n, len(alpha))
    """
    if (k == 0 and alpha==0):
        raise Exception(
            'Either k or alpha should be non-zero'
        )
    if k==0:
        k = len(alpha)
        matIdk = np.eye(k)
        matW = dirichlet_matrix(n-k,alpha)
        matW = np.concatenate((matIdk,matW))
        #matW = np.random.permutation(matW)
        return(matW)
    else:
        matIdk = np.eye(k)
        if (alpha == 0):
            matW = np.random.rand(n-k,k)
        else:
            matW = dirichlet_matrix(n-k, alpha)
        matW = np.concatenate((matIdk,matW))
        matW = matW/matW.sum(axis=1)[:,None]
        #matW = np.random.permutation(matW)
        return(matW)



def word_topic(p, k):
    """
    Generates a word-topic matrix with entries p(w|t) from the uniform distribution

    Parameters:
    p --- the number of words
    k --- the number of topics

    -------
    Return:
    A --- matrix of shape(k, p)
    """
    matA = np.random.rand(k,p)
    matA = matA/matA.sum(axis=1)[:,None]
    return matA



def word_document(n,p,k,alpha=0, fact=False):
    """
    Generates a word-document matrix as a product of word-topic and topic-document matrices,
    the word-topic matrix is generated from the uniform distribution

    Parameters:
    n --- the number of documents
    p --- the number of words
    k --- the number of topics
    alpha --- the parameter of Dirichlet distribution for a topic conditioned on a document,
    if 0 then a word conditioned on a document generates from the uniform distribution
    fact --- if True, word-topic and topic-document matrices are returned

    --------
    A matrix of shape (p, n)
    """
    matW = topic_document(n,k,alpha)
    if k == 0:
        matA = word_topic(p, len(alpha))
    else:
        matA = word_topic(p, k)
    if fact:
        return(matW.dot(matA), matW, matA)
    return(matW.dot(matA))



def gen_freq_matrix(n,p,N,k=0,alpha=0,fact=False,loop=True):
    """
    Generates frequency matrix for word-doctument

    Parameters:
    n --- number of documents
    p --- number of words
    N --- number of words in each document
    alpha --- parameter of Dirichlet distribution
    fact --- if True, word-topic and topic-document matrices returned

    ----------
    Return:
    matrix of shape (p, n)
    """
    if fact:
        P,W,A = word_document(n,p,k,alpha,fact)
    else:
        P = word_document(n,p,k,alpha)
    X = np.zeros((n,p))
    for i in range(n):
        X[i,:] = sps.multinomial.rvs(N,P[i,:])
    if fact:
        return ((1/N)*X,P,W)
    return ((1/N)*X)


def gen_freq_matrix_given_P(N, P):
    """
    Given the matrix P, samples a frequency matrix and returns it
    
    N --- the number of words we sample in each document
    P --- the matrix which each row is multinomial distribution

    -------------
    Return:
    A matrix with the same shape as P
    """
    n = P.shape[0]
    p = P.shape[1]
    X = np.zeros((n,p))
    
    for i in range(n):
        X[i,:] = sps.multinomial.rvs(N, P[i,:])
    
    return ((1/N)*X) 


def find_topic_document_error(Theta, Theta_exp):
    """
    function to find permutation of Theta cols which minimize
    Frobenius norm:

        || Theta - Theta_exp ||

    
    Parameters:
    Theta --- the estimator of the matrix of Theta_exp
    Theta_exp --- the estimated matrix

    -------------
    Returns:
    The error of estimation
    """

    assert Theta.shape == Theta_exp.shape, "Theta.shape != Theta_exp.shape"
    error = np.inf

    for perm in permutations(range(Theta.shape[1])):
        temple_error = np.linalg.norm(Theta - Theta_exp[:, perm], ord='fro')
        if temple_error < error:
            error = temple_error

    return error