import numpy as np
import matplotlib.pyplot as plt
# default matplotlib colors
cs_mpl = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# a nice blue/red color
cblue = '#66ccff'
cred = '#cc0000'
cmap = plt.get_cmap('coolwarm_r')


def _get_diverging_colors_centered_at_zero(coefs):
    vabs = max(np.abs(coefs))
    coefs /= vabs  # now is in range -1, 1
    coefs += 1  # now is in range 0, 2
    coefs /= 2  # now is in range 0, 1
    colors = cmap(coefs)  # norm(coefs))
    return colors
