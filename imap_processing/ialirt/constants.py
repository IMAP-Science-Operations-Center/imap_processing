"""Module for constants and useful shared classes used in I-ALiRT processing."""

from dataclasses import dataclass
from typing import NamedTuple

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
    eff_area = 1.633e-4 * 1e-4  # effective area, cm2 to meters squared
    az_fov = np.deg2rad(30)  # azimuthal width of the field of view, radians
    fwhm_width = 0.085  # FWHM of energy width
    speed_ew = 0.5 * fwhm_width  # speed width of energy passband
    e_charge = 1.602176634e-19  # electronic charge, [C]
    speed_coeff = np.sqrt(2 * e_charge / prot_mass) / 1e3


class StationProperties(NamedTuple):
    """Class that represents properties of ground stations."""

    longitude: float  # longitude in degrees
    latitude: float  # latitude in degrees
    altitude: float  # altitude in kilometers
    min_elevation_deg: float  # minimum elevation angle in degrees


# Verified by Kiel and KSWC Observatory staff.
# Notes: the KSWC station is not yet operational,
# but will have the following properties:
STATIONS = {
    "Kiel": StationProperties(
        longitude=10.1808,  # degrees East
        latitude=54.2632,  # degrees North
        altitude=0.1,  # approx 100 meters
        min_elevation_deg=5,  # 5 degrees is the requirement
    ),
    "Korea": StationProperties(
        longitude=126.2958,  # degrees East
        latitude=33.4273,  # degrees North
        altitude=0.1,  # approx 100 meters
        min_elevation_deg=5,  # 5 degrees is the requirement
    ),
    "Manaus": StationProperties(
        longitude=-59.969334,  # degrees East (negative = West)
        latitude=-2.891257,  # degrees North (negative = South)
        altitude=0.1,  # approx 100 meters
        min_elevation_deg=5,  # 5 degrees is the requirement
    ),
    "SANSA": StationProperties(
        longitude=27.714,  # degrees East (negative = West)
        latitude=-25.888,  # degrees North (negative = South)
        altitude=1.542,  # approx 1542 meters
        min_elevation_deg=2,  # 5 degrees is the requirement
    ),
}
