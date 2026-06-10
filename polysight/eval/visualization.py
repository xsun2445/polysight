"""SAR visualization and batch figure generation."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from itertools import cycle


def visualize_sar(sar_list, radar_list, labels=None, title=None, vmax_ratio=5):
    """Visualize SAR images (amplitude + phase + H/V ratio).

    Parameters
    ----------
    sar_list : list of ndarray
        Complex SAR images.
    radar_list : list of str
        Radar names, e.g. ['RH', 'RV'].
    labels : dict or None
        Bounding box labels {name: (x0, y0, x1, y1)}.
    title : str or None
    vmax_ratio : float
        Clamp for H/V amplitude display.
    """
    color_list = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    sar_dict = {name: img for name, img in zip(radar_list, sar_list)}
    has_hv = 'RH' in sar_dict and 'RV' in sar_dict

    if has_hv:
        sar_h, sar_v = sar_dict['RH'], sar_dict['RV']
        div = sar_h / sar_v
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes[0, 0].imshow(np.abs(sar_h), cmap='viridis')
        axes[0, 0].set_title('Amplitude H')
        axes[0, 1].imshow(np.abs(sar_v), cmap='viridis')
        axes[0, 1].set_title('Amplitude V')
        axes[0, 2].imshow(np.abs(div), cmap='viridis',
                          norm=mpl.colors.Normalize(vmin=0, vmax=vmax_ratio))
        axes[0, 2].set_title('Amplitude H/V')
        axes[1, 0].imshow(np.angle(sar_h), cmap='viridis')
        axes[1, 0].set_title('Phase H')
        axes[1, 1].imshow(np.angle(sar_v), cmap='viridis')
        axes[1, 1].set_title('Phase V')
        axes[1, 2].imshow(np.angle(div), cmap='viridis')
        axes[1, 2].set_title('Phase H/V')
        for ax in axes.ravel():
            ax.axis('off')
        if labels:
            color_cycle = cycle(color_list)
            for name, bbox in labels.items():
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = bbox
                c = next(color_cycle)
                for ax in axes.ravel():
                    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                             linewidth=1, edgecolor=c, facecolor='none')
                    ax.add_patch(rect)
                    ax.text((x0 + x1) / 2, (y0 + y1) / 2, name,
                            color=c, fontsize=8, va='bottom', ha='center')
    else:
        n = len(sar_list)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), squeeze=False)
        for j, (name, img) in enumerate(zip(radar_list, sar_list)):
            axes[0, j].imshow(np.abs(img), cmap='viridis')
            axes[0, j].set_title(f'Amplitude {name}')
            axes[0, j].axis('off')
            axes[1, j].imshow(np.angle(img), cmap='viridis')
            axes[1, j].set_title(f'Phase {name}')
            axes[1, j].axis('off')

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


def make_sar_figure(sar_h, sar_v, labels, title, subtitle, vmax_ratio=7, dpi=150):
    """Create a 2x3 SAR figure (amplitude + phase) with labels.

    Returns the matplotlib figure object.
    """
    color_list = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    div = sar_h / sar_v

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=dpi)
    axes[0, 0].imshow(np.abs(sar_h), cmap='viridis')
    axes[0, 0].set_title('Amplitude H')
    axes[0, 1].imshow(np.abs(sar_v), cmap='viridis')
    axes[0, 1].set_title('Amplitude V')
    axes[0, 2].imshow(np.abs(div), cmap='viridis',
                      norm=mpl.colors.Normalize(vmin=0, vmax=vmax_ratio))
    axes[0, 2].set_title('Amplitude H/V')
    axes[1, 0].imshow(np.angle(sar_h), cmap='viridis')
    axes[1, 0].set_title('Phase H')
    axes[1, 1].imshow(np.angle(sar_v), cmap='viridis')
    axes[1, 1].set_title('Phase V')
    axes[1, 2].imshow(np.angle(div), cmap='viridis')
    axes[1, 2].set_title('Phase H/V')

    for ax in axes.ravel():
        ax.axis('off')

    if labels and isinstance(labels, dict) and len(labels) > 0:
        color_cycle = cycle(color_list)
        for name, bbox in labels.items():
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = bbox
            c = next(color_cycle)
            for ax in axes.ravel():
                rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                         linewidth=1, edgecolor=c, facecolor='none')
                ax.add_patch(rect)
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, name,
                        color=c, fontsize=7, va='bottom', ha='center',
                        fontweight='bold')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.text(0.5, 0.93, subtitle, ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig
