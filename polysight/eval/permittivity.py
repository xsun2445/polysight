"""Fresnel reflection coefficient calculations and material permittivity.

Provides functions for computing dielectric permittivity from polarimetric
SAR measurements using Fresnel equations.
"""

import numpy as np


# Common material permittivities (real, imaginary) at ~77 GHz
MATERIALS = {
    'Aluminum': (1e5, 1e10),
    'Dry Air': (1.0, 0),
    'Water': (9.87, 18.23),
    'Alumina': (9.8, 0.001),
    'Dry Wood': (1.98, 0.083),
    'Wet Wood': (10, 2),
    'Glass': (5, 0.02),
    'Concrete': (2.55, 0.084),
    'Rock Wool': (1.75, 0.013),
    'Plasterboard': (1.74, 0.023),
    'Human Skin': (41, 18),
    'PC': (4, 0.005),
    'Acrylic': (4.5, 0.5),
    'PET': (3.4, 0.2),
}


def rs_to_epsilon(rs, theta_i):
    """Compute permittivity from s-polarized reflectivity.

    Args:
        rs: S-polarized complex reflectivity.
        theta_i: Incident angle in degrees.
    """
    theta_i = np.radians(theta_i)
    return np.sin(theta_i)**2 * (1 + ((1 - rs) / (np.tan(theta_i) * (rs + 1)))**2)


def ratio_to_epsilon(p, theta_i):
    """Compute permittivity from polarimetric ratio rs/rp.

    Args:
        p: Complex ratio rs/rp.
        theta_i: Incident angle in degrees.
    """
    theta_i = np.radians(theta_i)
    return (1 + 4 * p / (1 - p)**2 * np.sin(theta_i)**2) * np.tan(theta_i)**2


def fresnel_reflectivity(epsilon_complex, theta_inc_deg):
    """Fresnel s- and p-polarized complex reflection coefficients.

    Args:
        epsilon_complex: Complex relative permittivity.
        theta_inc_deg: Incident angle in degrees.

    Returns:
        (rs, rp): Complex reflection coefficients.
    """
    theta_i = np.radians(theta_inc_deg)
    n1 = 1  # Air
    n2 = np.sqrt(epsilon_complex)
    sin_theta_t = n1 / n2 * np.sin(theta_i)
    theta_t = np.arcsin(sin_theta_t)
    cos_theta_i = np.cos(theta_i)
    cos_theta_t = np.cos(theta_t)
    rs = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
    rp = (n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)
    return rs, rp


def fresnel_reflectivity_power(epsilon_complex, theta_inc_deg):
    """Fresnel power reflectivity (|rs|^2, |rp|^2)."""
    theta_i = np.radians(theta_inc_deg)
    rs = ((np.cos(theta_i) - np.sqrt(epsilon_complex - np.sin(theta_i)**2))
          / (np.cos(theta_i) + np.sqrt(epsilon_complex - np.sin(theta_i)**2)))
    rp = (-(epsilon_complex * np.cos(theta_i) - np.sqrt(epsilon_complex - np.sin(theta_i)**2))
          / (epsilon_complex * np.cos(theta_i) + np.sqrt(epsilon_complex - np.sin(theta_i)**2)))
    return np.abs(rs)**2, np.abs(rp)**2


def fresnel_reflectivity_complex(epsilon_complex, theta_inc_deg):
    """Fresnel complex reflection coefficients (alternative formulation)."""
    theta_i = np.radians(theta_inc_deg)
    rs = ((np.cos(theta_i) - np.sqrt(epsilon_complex - np.sin(theta_i)**2))
          / (np.cos(theta_i) + np.sqrt(epsilon_complex - np.sin(theta_i)**2)))
    rp = (-(epsilon_complex * np.cos(theta_i) - np.sqrt(epsilon_complex - np.sin(theta_i)**2))
          / (epsilon_complex * np.cos(theta_i) + np.sqrt(epsilon_complex - np.sin(theta_i)**2)))
    return rs, rp


def reflectivity_at_angles(epsilon, angles):
    """Compute reflectivity across an array of angles.

    Returns:
        (Rs_list, Rp_list): Arrays of complex reflectivity values.
    """
    Rs_list, Rp_list = [], []
    for a in angles:
        Rs, Rp = fresnel_reflectivity_complex(epsilon, a)
        Rs_list.append(Rs)
        Rp_list.append(Rp)
    return np.array(Rs_list), np.array(Rp_list)


def remove_axial_ratio(sh, sv, AR=0.1):
    """Remove axial ratio effect from H/V amplitude measurements."""
    return -np.arctan((AR * sv - sh) / (sv - AR * sh))


def remove_axial_ratio_2(p, AR=0.1):
    """Remove axial ratio from polarimetric ratio."""
    return -np.arctan((AR - p) / (1 - AR * p))


def remove_axial_ratio_3(p, AR=0.1):
    """Remove axial ratio (linear approximation)."""
    return -(AR - p) / (1 - AR * p)
