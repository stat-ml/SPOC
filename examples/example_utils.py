import numpy as np
import spoc.spoc as spoc
import matplotlib as mp
from matplotlib.patches import Ellipse
mp.rcParams.update({'font.size': 18, 'font.family': 'serif'})
import matplotlib.pyplot as plt


def plot_spectrum(A, n_clusters,
                  pure_inds=None, sym=True, draw_selected=False):

    if sym:
        u, lambd = spoc.SPOC._get_U_L(A, n_clusters)
    else:
        u, lambd, v = spoc.SPOC._get_U_L_V(A, n_clusters)

    fig, axes = plt.subplots(figsize=(12, 8))
    axes.scatter(u[:, 0], u[:, 1], c='b',
                 alpha=0.5, label='usual nodes')
    if pure_inds:
        axes.scatter(u[pure_inds, 0],
                     u[pure_inds, 1],
                     c='r', marker='s', s=100,
                     label='pure nodes')

    if draw_selected:
        theta, b, selected = spoc.SPOC().fit(A, n_clusters, sym=sym,
                                             return_pure_nodes_indices=True)

        axes.scatter(u[selected, 0],
                     u[selected, 1],
                     c='g', marker='*', s=350,
                     label='selected nodes')

    axes.set_title('eigenvectors')
    axes.set_xlabel(r'$x$')
    axes.set_ylabel(r'$y$')
    axes.grid()
    axes.legend()


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip






