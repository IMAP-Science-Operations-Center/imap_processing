"""Module for constants and useful shared classes used in I-ALiRT processing."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IalirtSwapiConstants:
    """
    Constants for I-ALiRT SWAPI which can be used across different levels or classes.

    Attributes
    ----------
    BOLTZ: float
        Boltzmann constant [J/K]
    AT_MASS: float
        Atomic mass [kg]
    PROT_MASS: float
        Mass of proton [kg]
    EFF_AREA: float
        Instrument effective area [m^2]
    AZ_FOV: float
        Azimuthal width of the field of view for solar wind [radians]
    FWHM_WIDTH: float
        Full Width at Half Maximum of energy width [unitless]
    SPEED_EW: float
        Speed width of energy passband [unitless]
    """

    # Scientific constants used in optimization model
    boltz = 1.380649e-23  # Boltzmann constant, J/K
    at_mass = 1.6605390666e-27  # atomic mass, kg
    prot_mass = 1.007276466621 * at_mass  # mass of proton, kg
    eff_area = 3.3e-5 * 1e-4  # effective area, meters squared
    az_fov = np.deg2rad(30)  # azimuthal width of the field of view, radians
    fwhm_width = 0.085  # FWHM of energy width
    speed_ew = 0.5 * fwhm_width  # speed width of energy passband
