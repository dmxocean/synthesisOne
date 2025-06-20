# src/utils/matplotlib.py

"""
Matplotlib configuration for the Translator Assignment System
"""

import matplotlib.pyplot as plt

plt.rcParams.update({
    # Figure settings
    'figure.figsize': (10, 7),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    
    # Font settings
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    
    # Grid settings
    'axes.grid': False,
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
    
    # Layout
    'figure.autolayout': True,
    'figure.constrained_layout.use': True,
    
    # Professional color scheme
    'axes.prop_cycle': plt.cycler('color', plt.cm.tab10.colors),
})