

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

#set colour palette
palette = sns.color_palette()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
        
def plot_gmm(gmm, Z, ax):
    labels = gmm.fit(Z).predict(Z) #do the fit
    for ii in np.flip(np.unique(labels)):
        iplot = labels==ii
        ax.scatter(Z[iplot, 0], Z[iplot, 1], color=palette[ii], zorder=2,
                   alpha=0.1, marker='.', edgecolor='none')
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max() #opacity of the elipses
    
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha = w*w_factor)
