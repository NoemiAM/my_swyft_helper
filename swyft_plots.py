import numpy as np
import pylab as plt
from scipy.integrate import simps
import swyft
import swyft.lightning.utils

from typing import Sequence, Union

# Plotting routines based on swyft's plotting routines


def grid_interpolate_samples(x, y, bins=1000, return_norm=False):
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    x_grid = np.linspace(x[0], x[-1], bins)
    y_grid = np.interp(x_grid, x, y)
    norm = simps(y_grid, x_grid)
    y_grid_normed = y_grid / norm
    if return_norm:
        return x_grid, y_grid_normed, norm
    else:
        return x_grid, y_grid_normed


def get_HDI_thresholds(x, cred_level=[0.68268, 0.95450, 0.99730]):
    x = x.flatten()
    x = np.sort(x)[::-1]  # Sort backwards
    total_mass = x.sum()
    enclosed_mass = np.cumsum(x)
    idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
    levels = np.array(x[idx])
    return levels


def plot_2d(
    logratios,
    parname1,
    parname2,
    truth=None,
    ax=None,
    bins=100,
    color="k",
    cmap="gray_r",
    smooth=0.0,
    bounds=None,
    imshow=True,
):
    """Plot 2-dimensional posteriors."""
    counts, xy = swyft.lightning.utils.get_pdf(
        logratios, [parname1, parname2], bins=bins, smooth=smooth
    )
    xbins = xy[:, 0]
    ybins = xy[:, 1]

    if ax is None:
        ax = plt.gca()

    levels = sorted(get_HDI_thresholds(counts))
    ax.contour(
        counts.T,
        extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
        levels=levels,
        linestyles=[":", "--", "-"],
        colors=color,
    )
    if imshow:
        ax.imshow(
            counts.T,
            extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
            cmap=cmap,
            origin="lower",
            aspect="auto",
        )
    if bounds is not None:
        ax.set_xlim([bounds[parname1][0], bounds[parname1][1]])
        ax.set_ylim([bounds[parname2][0], bounds[parname2][1]])
    else:
        ax.set_xlim([xbins.min(), xbins.max()])
        ax.set_ylim([ybins.min(), ybins.max()])
        
    if truth is not None:
        ax.axvline(truth[parname1], color="r", ls='--')
        ax.axhline(truth[parname2], color="r", ls='--')


def plot_1d(
    logratios,
    parname,
    truth=None,
    ax=None,
    bins=100,
    color="k",
    contours=True,
    smooth=0.0,
    bounds=None
):
    """Plot 1-dimensional posteriors."""
    v, zm = swyft.lightning.utils.get_pdf(logratios, parname, bins=bins, smooth=smooth)
    zm = zm[:, 0]

    if ax is None:
        ax = plt.gca()

    levels = sorted(get_HDI_thresholds(v))
    if contours:
        contour1d(zm, v, levels, ax=ax, color=color)
    ax.plot(zm, v, color=color)
    if bounds is not None:
        ax.set_xlim([bounds[parname][0], bounds[parname][1]])
    else:
        ax.set_xlim([zm.min(), zm.max()])
    ax.set_ylim([-v.max() * 0.05, v.max() * 1.1])
    
    if truth is not None:
        ax.axvline(truth[parname], color="r", ls='--')


def corner(
    logratios,
    parnames,
    bins=100,
    truth=None,
    figsize=(10, 10),
    color="k",
    cmap="gray_r",
    labels=None,
    label_args={},
    contours_1d: bool = True,
    fig=None,
    labeler=None,
    smooth=0.0,
    bounds=None,
) -> None:
    """Make a beautiful corner plot.

    Args:
        samples: Samples from `swyft.Posteriors.sample`
        pois: List of parameters of interest
        truth: Ground truth vector
        bins: Number of bins used for histograms.
        figsize: Size of figure
        color: Color
        labels: Custom labels (default is parameter names)
        label_args: Custom label arguments
        contours_1d: Plot 1-dim contours
        fig: Figure instance
    """
    K = len(parnames)
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=figsize)
    else:
        axes = np.array(fig.get_axes()).reshape((K, K))
    lb = 0.125
    tr = 0.9
    whspace = 0.1
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    diagnostics = {}

    if labeler is not None:
        labels = [labeler.get(k, k) for k in parnames]
    else:
        labels = parnames

    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            # Switch off upper left triangle
            if i < j:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                continue

            # Formatting labels
            if j > 0 or i == 0:
                ax.set_yticklabels([])
                # ax.set_yticks([])
            if i < K - 1:
                ax.set_xticklabels([])
                # ax.set_xticks([])
            if i == K - 1:
                ax.set_xlabel(labels[j], **label_args)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], **label_args)

            # 2-dim plots
            if j < i:
                try:
                    ret = plot_2d(
                        logratios,
                        parnames[j],
                        parnames[i],
                        truth=truth,
                        ax=ax,
                        color=color,
                        cmap=cmap,
                        bins=bins,
                        smooth=smooth,
                        bounds=bounds
                    )
                except swyft.SwyftParameterError:
                    pass

            if j == i:
                try:
                    ret = plot_1d(
                        logratios,
                        parnames[i],
                        truth=truth,
                        ax=ax,
                        color=color,
                        bins=bins,
                        contours=contours_1d,
                        smooth=smooth,
                        bounds=bounds
                    )
                except swyft.SwyftParameterError:
                    pass

    return fig, axes


def contour1d(z, v, levels, ax=plt, color=None, **kwargs):
    y0 = -1.0 * v.max()
    y1 = 5.0 * v.max()
    for level in levels:
        ax.fill_between(z, y0, y1, where=v > level, color=color, alpha=0.1)
        

def plot_zz(
    coverage_samples,
    params: Union[str, Sequence[str]],
    z_max: float = 3.5,
    bins: int = 50,
    ax=None,
):
    """Make a zz plot.

    Args:
        coverage_samples: Collection of CoverageSamples object
        params: Parameters of interest
        z_max: Maximum value of z.
        bins: Number of discretization bins.
    """
    cov = swyft.estimate_coverage(coverage_samples, params, z_max=z_max, bins=bins)
    ax = ax if ax else plt.gca()
    swyft.plot.mass.plot_empirical_z_score(ax, cov[:, 0], cov[:, 1], cov[:, 2:])


def plot_pp(
    coverage_samples,
    params: Union[str, Sequence[str]],
    z_max: float = 3.5,
    bins: int = 50,
    ax=None,
):
    """Make a pp plot."""
    cov = swyft.estimate_coverage(coverage_samples, params, z_max=z_max, bins=bins)
    alphas = 1 - swyft.plot.mass.get_alpha(cov)
    ax = ax if ax else plt.gca()
    ax.fill_between(alphas[:, 0], alphas[:, 2], alphas[:, 3], color="0.8")
    ax.plot(alphas[:, 0], alphas[:, 1], "k")
    plt.plot([0, 1], [0, 1], "g--")
    plt.xlabel("Nominal credibility [$1-p$]")
    plt.ylabel("Empirical coverage [$1-p$]")
    # swyft.plot.mass.plot_empirical_z_score(ax, cov[:,0], cov[:,1], cov[:,2:])