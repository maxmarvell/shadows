"""Plotting utilities and configuration for Shadow CI scripts.

This module provides consistent matplotlib configuration across all scripts,
with LaTeX rendering, serif fonts, and seaborn styling.
"""

import matplotlib.pyplot as plt


def setup_plotting_style():
    """Configure matplotlib with LaTeX fonts and seaborn dark grid style.

    This function should be called at the beginning of any script that produces
    plots to ensure consistent styling across the package.

    Features:
    - LaTeX text rendering with amsmath and bbold packages
    - Serif fonts throughout
    - Seaborn dark grid style for clean backgrounds
    - Grid lines below plot elements for clarity
    - Consistent font sizes for labels, ticks, and legends

    Example:
        >>> from plotting_config import setup_plotting_style
        >>> setup_plotting_style()
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> plt.show()
    """
    # LaTeX font configuration
    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }

    # Apply seaborn dark grid style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Configure LaTeX preamble with additional packages
    plt.rc('text.latex', preamble=R'\usepackage{amsmath} \usepackage{bbold}')

    # Apply font settings
    plt.rcParams.update(tex_fonts)

    # Ensure grid lines are drawn below other plot elements
    plt.rcParams['axes.axisbelow'] = True


def save_figure(filename, dpi=300, bbox_inches='tight', **kwargs):
    """Save figure with consistent high-quality settings.

    Args:
        filename: Output filename (with extension, e.g., 'plot.pdf')
        dpi: Resolution in dots per inch (default: 300)
        bbox_inches: Bounding box setting (default: 'tight')
        **kwargs: Additional arguments passed to plt.savefig()

    Example:
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure('my_plot.pdf')
    """
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"Figure saved to: {filename}")
