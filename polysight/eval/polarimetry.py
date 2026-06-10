"""Polarimetric SAR evaluation and material characterization.

Provides functions for loading SAR data, computing polarimetric features,
extracting material properties, and preparing datasets.
"""

import numpy as np
import json
import os
import copy

from polysight.eval import permittivity


def load_sar_fromfile(collection_name, radar_list, labels_dir=None, cfg_load=None):
    """Load pre-saved SAR images from a labeled collection.

    Returns:
        (label_dict, sar_list)
    """
    radar_name_to_savename = {
        'LS': 'sar_l.npy',
        'RH': 'sar_h.npy',
        'RV': 'sar_v.npy',
    }

    save_folder = os.path.join(labels_dir, collection_name)
    config_path = os.path.join(save_folder, f'{collection_name}.json')
    with open(config_path, 'r') as f:
        label_dict = json.load(f)

    sar_list = []
    for radar_name in radar_list:
        sar = np.load(os.path.join(save_folder, radar_name_to_savename[radar_name]))
        if cfg_load is not None:
            from scipy.ndimage import gaussian_filter
            sar = gaussian_filter(sar, sigma=cfg_load['sigma'])
        sar_list.append(sar)
    return label_dict, sar_list


def visualize_labeled_sar(labels, sar_h, sar_v):
    """Visualize SAR H/V amplitude and phase with bounding box labels."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from itertools import cycle

    color_list = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    color_cycle = cycle(color_list)

    div = sar_h / sar_v
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    axes[0, 0].imshow(np.abs(sar_h), cmap='viridis')
    axes[0, 0].set_title('Amplitude H')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(np.abs(sar_v), cmap='viridis')
    axes[0, 1].set_title('Amplitude V')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(np.abs(div), cmap='viridis',
                      norm=mpl.colors.Normalize(vmin=0, vmax=7))
    axes[0, 2].set_title('Amplitude H / V')
    axes[0, 2].axis('off')
    axes[1, 0].imshow(np.angle(sar_h), cmap='viridis')
    axes[1, 0].set_title('Phase H')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(np.angle(sar_v), cmap='viridis')
    axes[1, 1].set_title('Phase V')
    axes[1, 1].axis('off')
    axes[1, 2].imshow(np.angle(div), cmap='viridis')
    axes[1, 2].set_title('Phase H / V')
    axes[1, 2].axis('off')

    color_cycle = cycle(color_list)
    for (x0, y0, x1, y1) in labels.values():
        c = next(color_cycle)
        for ax in axes.ravel():
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=1, edgecolor=c, facecolor='none')
            ax.add_patch(rect)
    fig.suptitle([x for x in labels.keys()], fontsize=10)
    plt.tight_layout()
    plt.show()


def get_percentile_from_array(data, p_low=25, p_high=75, to_be_filtered=None):
    """Filter array values by percentile range."""
    q25 = np.percentile(np.abs(data), p_low)
    q75 = np.percentile(np.abs(data), p_high)
    if to_be_filtered is None:
        return data[(np.abs(data) >= q25) & (np.abs(data) <= q75)]
    return to_be_filtered[(np.abs(data) >= q25) & (np.abs(data) <= q75)]


def load_materials(class_list, collection_list, labels_dir=None, cfg_load=None):
    """Load and crop SAR material patches by class label."""
    class_dict = {c: [] for c in class_list}
    for fd in collection_list:
        label_dict, [sar_h, sar_v] = load_sar_fromfile(fd, ['RH', 'RV'], labels_dir=labels_dir)
        if cfg_load is not None:
            from scipy.ndimage import gaussian_filter
            sar_h = gaussian_filter(sar_h, sigma=cfg_load['sigma'])
            sar_v = gaussian_filter(sar_v, sigma=cfg_load['sigma'])
        div = sar_h / sar_v
        for c in class_list:
            if c in label_dict['labels']:
                x0, y0, x1, y1 = label_dict['labels'][c]
                class_dict[c].append([sar_h[y0:y1, x0:x1], sar_v[y0:y1, x0:x1], div[y0:y1, x0:x1]])
    return class_dict


def load_materials_from_collection(collection_list, labels_dir=None, cfg_load=None):
    """Load all labeled materials from a list of collections."""
    list_of_materials = []
    for fd in collection_list:
        label_dict, [sar_h, sar_v] = load_sar_fromfile(fd, ['RH', 'RV'], labels_dir=labels_dir)
        if cfg_load is not None and 'sigma' in cfg_load:
            from scipy.ndimage import gaussian_filter
            sar_h = gaussian_filter(sar_h, sigma=cfg_load['sigma'])
            sar_v = gaussian_filter(sar_v, sigma=cfg_load['sigma'])
        div = sar_h / sar_v
        material_dict = {}
        for name, [x0, y0, x1, y1] in label_dict['labels'].items():
            if cfg_load is not None and 'shrink_w' in cfg_load:
                x0 += cfg_load['shrink_w']
                x1 -= cfg_load['shrink_w']
                y0 += cfg_load['shrink_w']
                y1 -= cfg_load['shrink_w']
            material_dict[name] = [sar_h[y0:y1, x0:x1], sar_v[y0:y1, x0:x1], div[y0:y1, x0:x1]]
        list_of_materials.append(material_dict)
    return list_of_materials


def calc_polarimetry(material, ref, config=None):
    """Calculate polarization angle and referenced phase from H/V ratio."""
    if config is None:
        config = {'AR': 0.1, 'percentile_range': [30, 70]}
    AR = config.get('AR', 0)
    if 'percentile_range' in config:
        ref = get_percentile_from_array(ref, config['percentile_range'][0],
                                        config['percentile_range'][1])
        material = get_percentile_from_array(material, config['percentile_range'][0],
                                              config['percentile_range'][1])
    temp = np.abs(material.ravel()) / np.mean(np.abs(ref)) / 0.8
    polarization_angle = permittivity.remove_axial_ratio_2(temp, AR=AR) / np.pi * 180
    ref_angle = np.angle(np.mean(ref))
    material_angle_refed = material.ravel() * np.exp(-1j * ref_angle)
    return polarization_angle, material_angle_refed


def calc_polarimetry_from_label(material_label_dict, config=None):
    """Calculate permittivity from a material label dictionary."""
    if config is None:
        config = {'AR': 0.1, 'percentile_range': [30, 70]}
    polarization_angle, _ = calc_polarimetry(
        material_label_dict['values'][2],
        material_label_dict['ref_values'][2],
        config=config)
    pol_val = np.tan(np.deg2rad(polarization_angle))
    return permittivity.ratio_to_epsilon(pol_val, material_label_dict['incident_angle'])


def load_dataset(label_name, labels_dir=None):
    """Load a pickled material dataset."""
    import pickle
    save_name = os.path.join(labels_dir, 'materials', f'{label_name}.pkl')
    with open(save_name, 'rb') as f:
        return pickle.load(f)


def prepare_dataset_by_label(label_dict, config=None):
    """Prepare feature vectors from a label dictionary."""
    if config is None:
        config = {'percentile_range': [0, 100], 'subsets': ['h', 'v', 'p'], 'scaler': [1, 1, 1]}
    curr_data = label_dict['values']
    _vector_list = []
    if 'h' in config['subsets']:
        sar_h = get_percentile_from_array(curr_data[-1], config['percentile_range'][0],
                                           config['percentile_range'][1],
                                           to_be_filtered=curr_data[0])
        _vector_list.append(sar_h * config['scaler'][0])
    if 'v' in config['subsets']:
        sar_v = get_percentile_from_array(curr_data[-1], config['percentile_range'][0],
                                           config['percentile_range'][1],
                                           to_be_filtered=curr_data[1])
        _vector_list.append(sar_v * config['scaler'][1])
    if 'p' in config['subsets']:
        sar_div = get_percentile_from_array(curr_data[-1], config['percentile_range'][0],
                                             config['percentile_range'][1])
        _vector_list.append(sar_div * config['scaler'][2])

    if config.get('remain_complex', False):
        curr_data = np.vstack([x.ravel() for x in _vector_list]).T
        curr_data = np.hstack([curr_data.real, curr_data.imag])
    else:
        curr_data = np.vstack([np.abs(x.ravel()) for x in _vector_list]).T

    return curr_data, [label_dict['name']] * curr_data.shape[0]


def prepare_dataset(collection_list, labels_dir=None, config=None):
    """Prepare feature matrix from a list of material collections."""
    if config is None:
        config = {'percentile_range': [0, 100], 'subsets': ['h', 'v', 'p'], 'scaler': [1, 1, 1]}
    if 'scaler' not in config:
        config['scaler'] = [1, 1, 1]
    material_dict_list = [load_dataset(x, labels_dir=labels_dir) for x in collection_list]
    data_list = []
    label_list = []
    for m_dict in material_dict_list:
        curr_data, curr_label = prepare_dataset_by_label(m_dict, config=config)
        data_list.append(curr_data)
        label_list += curr_label
    return np.vstack(data_list), label_list
