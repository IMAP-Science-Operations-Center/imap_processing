"""Calculate ULTRA Level 2 (L2) ENA Map Product."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.map_properties import (
    DEFAULT_ULTRA_L2_MAP_PROPERTIES,
    MapProperties,
)
from imap_processing.spice import geometry

logger = logging.getLogger(__name__)
logger.info("Importing ultra_l2 module")

# Set some default values for the map properties
PSET_SPICE_FRAME = geometry.SpiceFrame.IMAP_DPS
DEFAULT_L2_HEALPIX_NSIDE = 32
DEFAULT_L2_HEALPIX_NESTED = False
DEFAULT_L2_MAP_PROPERTIES = DEFAULT_ULTRA_L2_MAP_PROPERTIES

# These variables must always be present in each L1C dataset
REQUIRED_L1C_VARIABLES = [
    "counts",
    "exposure_factor",
    "sensitivity",
    "background_rates",
]

# These variables are projected to the map as the mean of pointing set pixels value,
# weighted by that pointing set pixel's exposure and solid angle
VARIABLES_TO_WEIGHT_BY_POINTING_SET_EXPOSURE_TIMES_SOLID_ANGLE = [
    "sensitivity",
    "background_rates",
    "observation_time",
]

# These variables are dropped after they are used to calculate flux and flux uncertainty
# They will not be present in the final map
VARIABLES_TO_DROP_AFTER_FLUX_CALCULATION = [
    "counts",
    "background_rates",
    "pointing_set_exposure_times_solid_angle",
    "num_pointing_set_pixel_members",
    "corrected_count_rate",
]


def read_into_pointing_set(
    input_data: xr.Dataset | str | Path, inplace: bool = False
) -> ena_maps.UltraPointingSet:
    """
    Read a path or Dataset into an UltraPointingSet.

    Parameters
    ----------
    input_data : xr.Dataset | str | Path
        Path to the CDF file or xarray Dataset containing the L1C dataset.
    inplace : bool
        If True and if input_data is a Dataset, modify input_data in place.
        Default is False, in which case a copy is made.
        Does not affect str or Path type input_data.

    Returns
    -------
    ena_maps.UltraPointingSet
        An UltraPointingSet object containing the L1C dataset.

    Raises
    ------
    ValueError
        If input_data is neither an xarray Dataset nor a path to a CDF file.
    KeyError
        If any of the required variables are missing from the input data.
    """
    # Allow for passing in EITHER xarray Datasets (preferable for testing)
    if isinstance(input_data, xr.Dataset):
        if not inplace:
            input_data = input_data.copy(deep=True)
        ultra_pointing_set = ena_maps.UltraPointingSet(
            l1c_dataset=input_data, spice_reference_frame=PSET_SPICE_FRAME
        )
    # OR paths to CDF files (preferable for projecting many PointingSets)
    elif isinstance(input_data, str | Path):
        ultra_pointing_set = ena_maps.UltraPointingSet(
            l1c_dataset=load_cdf(input_data), spice_reference_frame=PSET_SPICE_FRAME
        )
    else:
        raise ValueError(
            f"Input data must be either an xarray Dataset or a path to a CDF file "
            "containing the L1C dataset.\n"
            f"Found {type(input_data)} instead."
        )
    # Check that the required variables are present in the dataset
    for var in REQUIRED_L1C_VARIABLES:
        if var not in ultra_pointing_set.data.data_vars:
            raise KeyError(
                f"Missing required variable '{var}' in input data. "
                "Please ensure the dataset contains all required variables."
            )
    return ultra_pointing_set


def generate_ultra_healpix_skymap(
    ultra_l1c_psets: list[str | xr.Dataset],
    output_map_properties: MapProperties = DEFAULT_L2_MAP_PROPERTIES,
) -> ena_maps.HealpixSkyMap:
    """
    Generate a Healpix skymap from ULTRA L1C pointing sets.

    This function combines IMAP Ultra L1C pointing sets into a single L2 HealpixSkyMap.
    It handles the projection of values from pointing sets to the map,applies necessary
    weighting and background subtraction, and calculates flux and flux uncertainty.

    Parameters
    ----------
    ultra_l1c_psets : list[str | xr.Dataset]
        List of paths to ULTRA L1C pointing set files or xarray Datasets containing
        pointing set data.
    output_map_properties : MapProperties, optional
        Properties defining the output map configuration. If not provided, default L2
        map properties will be used.

    Returns
    -------
    ena_maps.HealpixSkyMap
        HealpixSkyMap object containing the combined data from all pointing sets,
        with calculated flux and flux uncertainty values.

    Notes
    -----
    The structure of this function goes as follows:
    1. Initialize the HealpixSkyMap object with the specified properties.
    2. Iterate over the input pointing sets and read them into UltraPointingSet objects.
    3. For each pointing set, weight certain variables by exposure and solid angle of
    the pointing set pixels.
    4. Project the pointing set values to the map using the push method.
    5. Perform subsequent processing for weighted quantities at the SkyMap level
    (e.g., divide weighted quantities by their summed weights to
    get their weighted mean)
    6. Calculate corrected count rate with background subtraction applied.
    7. Calculate flux and flux uncertainty.
    8. Drop unnecessary variables from the map.
    """
    if output_map_properties.sky_tiling_type is ena_maps.SkyTilingType.HEALPIX:
        map_nside, map_nested = (
            output_map_properties.nside,
            output_map_properties.nested,
        )
    else:
        map_nside, map_nested = (DEFAULT_L2_HEALPIX_NSIDE, DEFAULT_L2_HEALPIX_NESTED)

    # Initialize the HealpixSkyMap object
    skymap = ena_maps.HealpixSkyMap(
        nside=map_nside,
        nested=map_nested,
        spice_frame=output_map_properties.spice_reference_frame,
    )

    # Add additional data variables to the map
    output_map_properties.values_to_push_project.extend(
        [
            "observation_time",
            "pointing_set_exposure_times_solid_angle",
            "num_pointing_set_pixel_members",
        ]
    )

    for ultra_l1c_pset in ultra_l1c_psets:
        pointing_set = read_into_pointing_set(ultra_l1c_pset)
        logger.info(f"PSET epoch: {pointing_set.epoch}")

        pointing_set.data["num_pointing_set_pixel_members"] = xr.DataArray(
            np.ones(pointing_set.num_points, dtype=int),
            dims=(CoordNames.HEALPIX_INDEX.value),
        )
        pointing_set.data["observation_time"] = xr.DataArray(
            np.full(pointing_set.num_points, pointing_set.epoch),
            dims=(CoordNames.HEALPIX_INDEX.value),
        )
        # Add solid_angle * exposure of pointing set as data_var
        # so this quantity is projected to map pixels for use in weighted averaging
        pointing_set.data["pointing_set_exposure_times_solid_angle"] = (
            pointing_set.data["exposure_factor"] * pointing_set.solid_angle
        )

        # Initial processing for weighted quantities at PSET level
        # Weight the values by exposure and solid angle
        # (in that order to avoid double weighting by solid angle)
        for (
            quantity_to_weight
        ) in VARIABLES_TO_WEIGHT_BY_POINTING_SET_EXPOSURE_TIMES_SOLID_ANGLE:
            pointing_set.data[quantity_to_weight] = (
                pointing_set.data[quantity_to_weight]
                * pointing_set.data["pointing_set_exposure_times_solid_angle"]
            )

        skymap.project_pset_values_to_map(
            pointing_set=pointing_set,
            value_keys=set(
                output_map_properties.values_to_push_project + REQUIRED_L1C_VARIABLES
            ),
            index_match_method=ena_maps.IndexMatchMethod.PUSH,
        )

    # Subsequent processing for weighted quantities at SkyMap level
    for (
        quantity_to_weight
    ) in VARIABLES_TO_WEIGHT_BY_POINTING_SET_EXPOSURE_TIMES_SOLID_ANGLE:
        skymap.data_1d[quantity_to_weight] = (
            skymap.data_1d[quantity_to_weight]
            / skymap.data_1d["pointing_set_exposure_times_solid_angle"]
        )

    # TODO: Ask Ultra team about this - I think this is a decent
    # but imperfect approximation for meaning the exposure:
    # (dividing by the 1/(number of PSETs)) to fix the exposure being mean-ed over
    # all pixels in all PSETs which feed into a map superpixel,
    # rather than being mean-ed over pixels in a PSET and summed over PSETs
    skymap.data_1d["exposure_factor"] /= skymap.data_1d[
        "num_pointing_set_pixel_members"
    ] / len(ultra_l1c_psets)

    # TODO: Ask Ultra team about background rates - I think they should increase when
    # binned to larger pixels, as I've done here, but that was never explicitly stated
    skymap.data_1d["background_rates"] *= skymap.solid_angle / pointing_set.solid_angle

    # Get the energy bin widths from a PointingSet (they will all be the same)
    delta_energy = pointing_set.data["energy_bin_delta"]

    # Core calculations of flux and flux uncertainty for L2
    # Get corrected count rate with background subtraction applied
    skymap.data_1d["corrected_count_rate"] = (
        skymap.data_1d["counts"] / skymap.data_1d["exposure_factor"]
    ) - skymap.data_1d["background_rates"]

    # Calculate flux as corrected_counts / (sensitivity * solid_angle * delta_energy)
    skymap.data_1d["flux"] = skymap.data_1d["corrected_count_rate"] / (
        skymap.data_1d["sensitivity"] * skymap.solid_angle * delta_energy
    )

    skymap.data_1d["flux_uncertainty"] = (skymap.data_1d["counts"] ** 0.5) / (
        skymap.data_1d["exposure_factor"]
        * skymap.data_1d["sensitivity"]
        * skymap.solid_angle
        * delta_energy
    )

    # Drop the variables that are no longer needed
    skymap.data_1d = skymap.data_1d.drop_vars(
        VARIABLES_TO_DROP_AFTER_FLUX_CALCULATION,
        errors="ignore",
    )

    return skymap


def ultra_l2(
    data_dict: dict[str, xr.Dataset | str],
    data_version: str,
    output_map_properties: MapProperties = DEFAULT_L2_MAP_PROPERTIES,
) -> list[xr.Dataset]:
    """
    Generate and format Ultra L2 ENA Map Product from L1C Products.

    Parameters
    ----------
    data_dict : dict[str, xr.Dataset]
        Dict mapping l1c product identifiers to paths/Datasets containing l1c psets.
    data_version : str
        Version of the data product being created.
    output_map_properties : MapProperties, optional
        Properties of the map to be generated.
        Default is defined in `map_properties.py`.

    Returns
    -------
    list[xarray.Dataset,]
        L2 output dataset containing map of the counts on the sky.
        Wrapped in a list for consistency with other product levels.
    """
    l1c_products = data_dict.values()
    num_l1c_products = len(l1c_products)
    logger.info(f"Running ultra_l2 processing on {num_l1c_products} L1C products")

    # Regardless of the output sky tiling type, we will directly
    # project the PSET values into a healpix map. However, if we are outputting
    # a Healpix map, we can go directly to map with desired nside, nested params
    healpix_skymap = generate_ultra_healpix_skymap(
        ultra_l1c_psets=list(l1c_products),
        output_map_properties=output_map_properties,
    )

    # Output formatting for HEALPIX tiling
    if output_map_properties.sky_tiling_type is ena_maps.SkyTilingType.HEALPIX:
        map_dataset = healpix_skymap.to_dataset()
        # Add attributes related to the map
        map_attrs = {
            "HEALPix_nside": output_map_properties.nside,
            "HEALPix_nest": output_map_properties.nested,
            "Data_version": data_version,
        }

    # TODO: Implement conversion to Rectangular map
    elif output_map_properties.sky_tiling_type is ena_maps.RectangularSkyMap:
        map_attrs = {
            "Spacing_degrees": output_map_properties.spacing_deg,
            "Data_version": data_version,
        }
        pass

    # Add the defined attributes to the map's global attrs
    map_dataset.attrs.update(map_attrs)
    return [map_dataset]
