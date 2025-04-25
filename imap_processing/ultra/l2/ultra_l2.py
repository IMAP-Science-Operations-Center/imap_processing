"""Calculate ULTRA Level 2 (L2) ENA Map Product."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils.coordinates import CoordNames

logger = logging.getLogger(__name__)
logger.info("Importing ultra_l2 module")

# Default properties for the Ultra L2 map
DEFAULT_ULTRA_L2_MAP_STRUCTURE: ena_maps.RectangularSkyMap | ena_maps.HealpixSkyMap = (
    ena_maps.AbstractSkyMap.from_dict(
        {
            "sky_tiling_type": "HEALPIX",
            "spice_reference_frame": "ECLIPJ2000",
            "values_to_push_project": [
                "counts",
                "exposure_factor",
                "sensitivity",
                "background_rates",
            ],
            "nside": 32,
            "nested": False,
        }
    )
)

# Set some default Healpix parameters - these must be defined, even if also
# present in the DEFAULT_ULTRA_L2_MAP_STRUCTURE, because we always make a Healpix map
# regardless of the output map type
DEFAULT_L2_HEALPIX_NSIDE = 32
DEFAULT_L2_HEALPIX_NESTED = False


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


def generate_ultra_healpix_skymap(
    ultra_l1c_psets: list[str | xr.Dataset],
    output_map_structure: (
        ena_maps.RectangularSkyMap | ena_maps.HealpixSkyMap
    ) = DEFAULT_ULTRA_L2_MAP_STRUCTURE,
) -> ena_maps.HealpixSkyMap:
    """
    Generate a Healpix skymap from ULTRA L1C pointing sets.

    This function combines IMAP Ultra L1C pointing sets into a single L2 HealpixSkyMap.
    It handles the projection of values from pointing sets to the map, applies necessary
    weighting and background subtraction, and calculates flux and flux uncertainty.

    Parameters
    ----------
    ultra_l1c_psets : list[str | xr.Dataset]
        List of paths to ULTRA L1C pointing set files or xarray Datasets containing
        pointing set data.
    output_map_structure : ena_maps.RectangularSkyMap | ena_maps.HealpixSkyMap, optional
        Empty SkyMap structure providing the properties of the map to be generated.
        Defaults to DEFAULT_ULTRA_L2_MAP_STRUCTURE defined in this module.

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
    if output_map_structure.tiling_type is ena_maps.SkyTilingType.HEALPIX:
        map_nside, map_nested = (
            output_map_structure.nside,
            output_map_structure.nested,
        )
    else:
        map_nside, map_nested = (DEFAULT_L2_HEALPIX_NSIDE, DEFAULT_L2_HEALPIX_NESTED)

    # Initialize the HealpixSkyMap object
    skymap = ena_maps.HealpixSkyMap(
        nside=map_nside,
        nested=map_nested,
        spice_frame=output_map_structure.spice_reference_frame,
    )

    # Add additional data variables to the map
    output_map_structure.values_to_push_project.extend(
        [
            "observation_time",
            "pointing_set_exposure_times_solid_angle",
            "num_pointing_set_pixel_members",
        ]
    )

    # Get full list of variables to push to the map: all requested variables plus
    # any which are required for L2 processing
    value_keys_to_push_project = list(
        set(output_map_structure.values_to_push_project + REQUIRED_L1C_VARIABLES)
    )

    for ultra_l1c_pset in ultra_l1c_psets:
        pointing_set = ena_maps.UltraPointingSet(ultra_l1c_pset)
        logger.info(
            f"Projecting a PointingSet with {pointing_set.num_points} pixels "
            f"at epoch:{pointing_set.epoch}\n"
            f"These values will be projected: {value_keys_to_push_project}"
        )

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
        pointing_set.data[
            VARIABLES_TO_WEIGHT_BY_POINTING_SET_EXPOSURE_TIMES_SOLID_ANGLE
        ] *= pointing_set.data["pointing_set_exposure_times_solid_angle"]

        skymap.project_pset_values_to_map(
            pointing_set=pointing_set,
            value_keys=value_keys_to_push_project,
            index_match_method=ena_maps.IndexMatchMethod.PUSH,
        )

    # Subsequent processing for weighted quantities at SkyMap level
    skymap.data_1d[VARIABLES_TO_WEIGHT_BY_POINTING_SET_EXPOSURE_TIMES_SOLID_ANGLE] /= (
        skymap.data_1d["pointing_set_exposure_times_solid_angle"]
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
    # Exposure time may contain 0s, producing NaNs in the corrected count rate and flux.
    # These NaNs are not incorrect, so we temporarily ignore numpy div by 0 warnings.
    with np.errstate(divide="ignore"):
        # Get corrected count rate with background subtraction applied
        skymap.data_1d["corrected_count_rate"] = (
            skymap.data_1d["counts"].astype(float) / skymap.data_1d["exposure_factor"]
        ) - skymap.data_1d["background_rates"]

        # Calculate flux = corrected_counts / (sensitivity * solid_angle * delta_energy)
        skymap.data_1d["flux"] = skymap.data_1d["corrected_count_rate"] / (
            skymap.data_1d["sensitivity"] * skymap.solid_angle * delta_energy
        )

        skymap.data_1d["flux_uncertainty"] = (
            skymap.data_1d["counts"].astype(float) ** 0.5
        ) / (
            skymap.data_1d["exposure_factor"]
            * skymap.data_1d["sensitivity"]
            * skymap.solid_angle
            * delta_energy
        )

    # Drop the variables that are no longer needed
    skymap.data_1d = skymap.data_1d.drop_vars(
        VARIABLES_TO_DROP_AFTER_FLUX_CALCULATION,
    )

    return skymap


def ultra_l2(
    data_dict: dict[str, xr.Dataset | str],
    data_version: str,
    output_map_structure: (
        ena_maps.RectangularSkyMap | ena_maps.HealpixSkyMap
    ) = DEFAULT_ULTRA_L2_MAP_STRUCTURE,
) -> list[xr.Dataset]:
    """
    Generate and format Ultra L2 ENA Map Product from L1C Products.

    Parameters
    ----------
    data_dict : dict[str, xr.Dataset]
        Dict mapping l1c product identifiers to paths/Datasets containing l1c psets.
    data_version : str
        Version of the data product being created.
    output_map_structure : ena_maps.RectangularSkyMap | ena_maps.HealpixSkyMap, optional
        Empty SkyMap structure providing the properties of the map to be generated.
        Defaults to DEFAULT_ULTRA_L2_MAP_STRUCTURE defined in this module.

    Returns
    -------
    list[xarray.Dataset,]
        L2 output dataset containing map of the counts on the sky.
        Wrapped in a list for consistency with other product levels.

    Raises
    ------
    NotImplementedError
        If asked to project to a rectangular map.
        # TODO: This is coming shortly
    """
    l1c_products = data_dict.values()
    num_l1c_products = len(l1c_products)
    logger.info(f"Running ultra_l2 processing on {num_l1c_products} L1C products")

    # Regardless of the output sky tiling type, we will directly
    # project the PSET values into a healpix map. However, if we are outputting
    # a Healpix map, we can go directly to map with desired nside, nested params
    healpix_skymap = generate_ultra_healpix_skymap(
        ultra_l1c_psets=list(l1c_products),
        output_map_structure=output_map_structure,
    )

    # Output formatting for HEALPIX tiling
    if output_map_structure.tiling_type is ena_maps.SkyTilingType.HEALPIX:
        map_dataset = healpix_skymap.to_dataset()
        # Add attributes related to the map
        map_attrs = {
            "HEALPix_nside": output_map_structure.nside,
            "HEALPix_nest": output_map_structure.nested,
            "Data_version": data_version,
        }

    # TODO: Implement conversion to Rectangular map
    elif output_map_structure.tiling_type is ena_maps.SkyTilingType.RECTANGULAR:
        map_attrs = {
            "Spacing_degrees": output_map_structure.spacing_deg,
            "Data_version": data_version,
        }
        raise NotImplementedError

    # Always add the following attributes to the map
    map_attrs.update(
        {
            "Sky_tiling_type": output_map_structure.tiling_type.value,
            "Spice_reference_frame": output_map_structure.spice_reference_frame,
        }
    )

    # Add the defined attributes to the map's global attrs
    map_dataset.attrs.update(map_attrs)
    return [map_dataset]
