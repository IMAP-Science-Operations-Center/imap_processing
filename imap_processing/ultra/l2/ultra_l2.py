"""Calculate ULTRA Level 2 (L2) ENA Map Product."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.naming import (
    INERTIAL_FRAME_LONG_NAMES,
    MapDescriptor,
    ns_to_duration_months,
)
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import get_energy_delta_minus_plus

logger = logging.getLogger(__name__)
logger.info("Importing ultra_l2 module")

# Default properties for the Ultra L2 map
DEFAULT_ULTRA_L2_MAP_STRUCTURE: ena_maps.RectangularSkyMap | ena_maps.HealpixSkyMap = (
    ena_maps.AbstractSkyMap.from_properties_dict(
        {
            "sky_tiling_type": "HEALPIX",
            "spice_reference_frame": "ECLIPJ2000",
            "values_to_push_project": [
                "counts",
            ],
            "values_to_pull_project": [
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
REQUIRED_L1C_VARIABLES_PUSH = [
    "counts",
]
REQUIRED_L1C_VARIABLES_PULL = [
    "exposure_factor",
    "sensitivity",
    "background_rates",
    "obs_date",
]

# These variables are projected to the map as the mean of pointing set pixels value,
# weighted by that pointing set pixel's exposure and solid angle
VARIABLES_TO_WEIGHT_BY_POINTING_SET_EXPOSURE_TIMES_SOLID_ANGLE = [
    "sensitivity",
    "background_rates",
    "obs_date",
]

# These variables are dropped after they are used to
# calculate ena_intensity and its statistical uncertainty
# They will not be present in the final map
VARIABLES_TO_DROP_AFTER_INTENSITY_CALCULATION = [
    "counts",
    "background_rates",
    "pointing_set_exposure_times_solid_angle",
    "num_pointing_set_pixel_members",
    "corrected_count_rate",
    "obs_date_for_std",
    "obs_date_squared_for_std",
]

# These variables may or may not be energy dependent, depending on the
# input data. They must be handled slightly differently when it comes to adding
# metadata to the map dataset.
INCONSISTENTLY_ENERGY_DEPENDENT_VARIABLES = [
    "obs_date",
    "exposure_factor",
    "obs_date_range",
]


def get_variable_attributes_optional_energy_dependence(
    cdf_attrs: ImapCdfAttributes,
    variable_array: xr.DataArray,
    *,
    check_schema: bool = True,
) -> dict:
    """
    Wrap `get_variable_attributes` to handle optionally energy-dependent vars.

    Several variables are only energy dependent in some cases (input PSET dependent).
    The metadata on those variables must be handled differently in such cases.

    Parameters
    ----------
    cdf_attrs : ImapCdfAttributes
        The CDF attributes object to use for getting variable attributes.
    variable_array : xr.DataArray
        The xarray DataArray containing the variable data and dims.
        Must have a name attribute.
    check_schema : bool
        Flag to bypass schema validation.

    Returns
    -------
    dict
        The attributes for the variable.
    """
    variable_name = variable_array.name
    variable_dims = variable_array.dims

    # These variables must get metadata with a different key if they are energy
    # dependent.
    if (variable_name in INCONSISTENTLY_ENERGY_DEPENDENT_VARIABLES) and (
        (CoordNames.ENERGY_L2.value not in variable_dims)
        and (CoordNames.ENERGY_ULTRA_L1C.value not in variable_dims)
    ):
        variable_name = f"{variable_name}_energy_independent"

    metadata = cdf_attrs.get_variable_attributes(
        variable_name=variable_name,
        check_schema=check_schema,
    )
    return metadata


def generate_ultra_healpix_skymap(
    ultra_l1c_psets: list[str | xr.Dataset],
    output_map_structure: (
        ena_maps.RectangularSkyMap | ena_maps.HealpixSkyMap
    ) = DEFAULT_ULTRA_L2_MAP_STRUCTURE,
) -> tuple[ena_maps.HealpixSkyMap, NDArray]:
    """
    Generate a Healpix skymap from ULTRA L1C pointing sets.

    This function combines IMAP Ultra L1C pointing sets into a single L2 HealpixSkyMap.
    It handles the projection of values from pointing sets to the map, applies necessary
    weighting and background subtraction, and calculates ena_intensity
    and ena_intensity_stat_unc.

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
        with calculated ena_intensity and its statistical uncertainty values.
    NDArray
        Array of epochs corresponding to the pointing sets used in the map.

    Raises
    ------
    ValueError
        If there are overlapping variable names in the push and pull projection lists.

    Notes
    -----
    The structure of this function goes as follows:
    1. Initialize the HealpixSkyMap object with the specified properties.
    2. Iterate over the input pointing sets and read them into UltraPointingSet objects.
    3. For each pointing set, weight certain variables by exposure and solid angle of
    the pointing set pixels.
    4. Project the pointing set values to the map using the push/pull methods.
    5. Perform subsequent processing for weighted quantities at the SkyMap level
    (e.g., divide weighted quantities by their summed weights to
    get their weighted mean)
    6. Calculate corrected count rate with background subtraction applied.
    7. Calculate ena_intensity and its statistical uncertainty.
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
            "num_pointing_set_pixel_members",
            "obs_date_for_std",
            "obs_date_squared_for_std",
        ]
    )
    output_map_structure.values_to_pull_project.extend(
        [
            "obs_date",
            "pointing_set_exposure_times_solid_angle",
        ]
    )

    # Get full list of variables to push to the map: all requested variables plus
    # any which are required for L2 processing
    output_map_structure.values_to_push_project = list(
        set(output_map_structure.values_to_push_project + REQUIRED_L1C_VARIABLES_PUSH)
    )
    output_map_structure.values_to_pull_project = list(
        set(output_map_structure.values_to_pull_project + REQUIRED_L1C_VARIABLES_PULL)
    )
    # If there are overlapping variable names, raise an error
    if set(output_map_structure.values_to_push_project).intersection(
        set(output_map_structure.values_to_pull_project)
    ):
        raise ValueError(
            "Some variables are present in both the PUSH and PULL projection lists. "
            "They will be projected in both ways (PUSH then PULL), which is likely "
            "not the intended behavior. Please check the projection lists."
            f"PUSH Variables: {output_map_structure.values_to_push_project} \n"
            f"PULL Variables: {output_map_structure.values_to_pull_project}"
        )

    all_pset_epochs = []
    for ultra_l1c_pset in ultra_l1c_psets:
        pointing_set = ena_maps.UltraPointingSet(ultra_l1c_pset)
        all_pset_epochs.append(pointing_set.epoch)
        logger.info(
            f"Projecting a PointingSet with {pointing_set.num_points} pixels "
            f"at epoch:{pointing_set.epoch}\n"
            "These values will be push projected: "
            f">> {output_map_structure.values_to_push_project}"
            "\nThese values will be pull projected: "
            f">> {output_map_structure.values_to_pull_project}",
        )

        pointing_set.data["num_pointing_set_pixel_members"] = xr.DataArray(
            np.ones(pointing_set.num_points, dtype=int),
            dims=(CoordNames.HEALPIX_INDEX.value),
        )

        # The obs_date is the same for all pixels in a pointing set, and the same
        # dimension as the exposure_factor.
        pointing_set.data["obs_date"] = xr.full_like(
            pointing_set.data["exposure_factor"],
            fill_value=pointing_set.epoch,
            dtype=np.int64,
        )
        pointing_set.data["obs_date_for_std"] = pointing_set.data["obs_date"].astype(
            np.float64
        )
        pointing_set.data["obs_date_squared_for_std"] = (
            pointing_set.data["obs_date_for_std"] ** 2
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

        # Project values such as counts via the PUSH method
        skymap.project_pset_values_to_map(
            pointing_set=pointing_set,
            value_keys=output_map_structure.values_to_push_project,
            index_match_method=ena_maps.IndexMatchMethod.PUSH,
        )

        # Project values such as exposure_factor via the PULL method
        skymap.project_pset_values_to_map(
            pointing_set=pointing_set,
            value_keys=output_map_structure.values_to_pull_project,
            index_match_method=ena_maps.IndexMatchMethod.PULL,
        )

    # Subsequent processing for weighted quantities at SkyMap level
    skymap.data_1d[VARIABLES_TO_WEIGHT_BY_POINTING_SET_EXPOSURE_TIMES_SOLID_ANGLE] /= (
        skymap.data_1d["pointing_set_exposure_times_solid_angle"]
    )

    # Background rates must be scaled by the ratio of the solid angles of the
    # map pixel / pointing set pixel
    skymap.data_1d["background_rates"] *= skymap.solid_angle / pointing_set.solid_angle

    # Get the energy bin widths from a PointingSet (they will all be the same)
    delta_energy = pointing_set.data["energy_bin_delta"]
    if CoordNames.TIME.value in delta_energy.dims:
        delta_energy = delta_energy.mean(
            dim=CoordNames.TIME.value,
        )

    # Core calculations of ena_intensity and its statistical uncertainty for L2
    # Exposure time may contain 0s, producing NaNs in the corrected count rate
    # and ena_intensity.
    # These NaNs are not incorrect, so we temporarily ignore numpy div by 0 warnings.
    with np.errstate(divide="ignore"):
        # Get corrected count rate with background subtraction applied
        skymap.data_1d["corrected_count_rate"] = (
            skymap.data_1d["counts"].astype(float) / skymap.data_1d["exposure_factor"]
        ) - skymap.data_1d["background_rates"]

        # Calculate ena_intensity = corrected_counts / (
        # sensitivity * solid_angle * delta_energy)
        skymap.data_1d["ena_intensity"] = skymap.data_1d["corrected_count_rate"] / (
            skymap.data_1d["sensitivity"] * skymap.solid_angle * delta_energy
        )

        skymap.data_1d["ena_intensity_stat_unc"] = (
            skymap.data_1d["counts"].astype(float) ** 0.5
        ) / (
            skymap.data_1d["exposure_factor"]
            * skymap.data_1d["sensitivity"]
            * skymap.solid_angle
            * delta_energy
        )

        # Calculate the standard deviation of the observation date as:
        # sqrt((sum(obs_date^2) / N) - (sum(obs_date) / N)^2)
        # where sum here refers to the projection process
        # summing over N pset pixels across different psets
        skymap.data_1d["obs_date_range"] = (
            (
                (
                    skymap.data_1d["obs_date_squared_for_std"]
                    / (skymap.data_1d["num_pointing_set_pixel_members"])
                )
                - (
                    (
                        skymap.data_1d["obs_date_for_std"]
                        / (skymap.data_1d["num_pointing_set_pixel_members"])
                    )
                    ** 2
                )
            )
            ** 0.5
        ).astype(np.int64)

    # Drop the variables that are no longer needed
    skymap.data_1d = skymap.data_1d.drop_vars(
        VARIABLES_TO_DROP_AFTER_INTENSITY_CALCULATION,
    )

    return skymap, np.array(all_pset_epochs)


def ultra_l2(
    data_dict: dict[str, xr.Dataset | str | Path],
    output_map_structure: (
        ena_maps.RectangularSkyMap | ena_maps.HealpixSkyMap
    ) = DEFAULT_ULTRA_L2_MAP_STRUCTURE,
    *,
    descriptor: str | None = None,
    store_subdivision_depth: bool = False,
) -> list[xr.Dataset]:
    """
    Generate and format Ultra L2 ENA Map Product from L1C Products.

    Parameters
    ----------
    data_dict : dict[str, xr.Dataset]
        Dict mapping l1c product identifiers to paths/Datasets containing l1c psets.
    output_map_structure : ena_maps.RectangularSkyMap | ena_maps.HealpixSkyMap, optional
        Empty SkyMap structure providing the properties of the map to be generated.
        If a descriptor is provided, this will be ignored.
        Defaults to DEFAULT_ULTRA_L2_MAP_STRUCTURE defined in this module.
    descriptor : str | None, optional
        A descriptor to set the output map structure
        If provided, this overrides the default output_map_structure parameter.
    store_subdivision_depth : bool, optional
        If True, the subdivision depth required to calculate each rectangular pixel
        value will be added to the map dataset.
        E.g. a "ena_intensity_subdivision_depth" DataArray will be
        added to the map dataset.
        Defaults to False.

    Returns
    -------
    list[xarray.Dataset,]
        L2 output dataset containing map of the counts on the sky.
        Wrapped in a list for consistency with other product levels.
    """
    inertial_frame = "unknown"
    if descriptor is not None:
        logger.info(
            f"Using the provided descriptor '{descriptor}' to set the map structure."
            "\nThis will override any input map structure."
        )
        map_descriptor = MapDescriptor.from_string(descriptor)
        output_map_structure = map_descriptor.to_empty_map()
        inertial_frame = map_descriptor.frame_descriptor
    inertial_frame_long_name = INERTIAL_FRAME_LONG_NAMES.get(inertial_frame, "unknown")

    # Object which holds CDF attributes for the map
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs(instrument="ultra")

    l1c_products: list[xr.Dataset] = list(data_dict.values())
    num_l1c_products = len(l1c_products)
    logger.info(f"Running ultra_l2 processing on {num_l1c_products} L1C products")

    ultra_sensor_number = 45 if "45sensor" in next(iter(data_dict.keys())) else 90
    logger.info(f"Assuming all products are from sensor {ultra_sensor_number}")

    # Regardless of the output sky tiling type, we will directly
    # project the PSET values into a healpix map. However, if we are outputting
    # a Healpix map, we can go directly to map with desired nside, nested params
    healpix_skymap, pset_epochs = generate_ultra_healpix_skymap(
        ultra_l1c_psets=l1c_products,
        output_map_structure=output_map_structure,
    )
    # Ensure that the epoch of the map is the earliest epoch of the input PSETs
    healpix_skymap.data_1d.assign_coords(
        epoch=(
            (CoordNames.TIME.value,),
            [
                pset_epochs.min(),
            ],
        ),
    )

    # TODO: replace 1 day in ns below with the actual end time of the last PSET.
    # Currently assumes the end time of the last PSET is 1 day after its start.
    map_duration_ns = (pset_epochs.max() + (86400 * 1e9)) - pset_epochs.min()
    map_duration_months_int = ns_to_duration_months(map_duration_ns)
    map_duration = f"{map_duration_months_int}mo"

    # Always add the common (non-tiling specific) attributes to the attr handler.
    # These can be updated/overwritten by the tiling specific attributes.
    cdf_attrs.add_instrument_variable_attrs(instrument="enamaps", level="l2-common")

    # Output formatting for HEALPIX tiling
    if output_map_structure.tiling_type is ena_maps.SkyTilingType.HEALPIX:
        # Add the tiling specific attributes to the attr handler.
        cdf_attrs.add_instrument_variable_attrs(
            instrument="enamaps", level="l2-healpix"
        )

        # Add the longitude and latitude coordinate-like data_vars to the map dataset
        # These are not xarray coordinates, but the lon/lat corresponding to the
        # Healpix pixel centers.
        for i, angle_name in enumerate(["longitude", "latitude"]):
            healpix_skymap.data_1d[angle_name] = xr.DataArray(
                data=healpix_skymap.az_el_points[:, i],
                dims=(CoordNames.GENERIC_PIXEL.value,),
            )
        map_dataset = healpix_skymap.to_dataset()
        # Add attributes related to the map
        map_attrs = {
            "HEALPix_nside": str(output_map_structure.nside),
            "HEALPix_nest": str(output_map_structure.nested),
        }

    elif output_map_structure.tiling_type is ena_maps.SkyTilingType.RECTANGULAR:
        # Add the tiling specific attributes to the attr handler.
        cdf_attrs.add_instrument_variable_attrs(
            instrument="enamaps", level="l2-rectangular"
        )
        rectangular_skymap, subdiv_depth_dict = healpix_skymap.to_rectangular_skymap(
            rect_spacing_deg=output_map_structure.spacing_deg,
            value_keys=healpix_skymap.data_1d.data_vars,
        )

        # Add the subdiv_depth_by_pixel of each key to the map dataset if requested
        if store_subdivision_depth:
            logger.info(
                "For debugging purposes, adding the subdivision depth "
                "required to calculate each rectangular pixel value to the map dataset."
            )
            for key, depth_by_pixel in subdiv_depth_dict.items():
                subdiv_depth_key = f"{key}_subdivision_depth"
                logger.info(f"Adding {subdiv_depth_key} to the map dataset.")
                rectangular_skymap.data_1d[subdiv_depth_key] = xr.DataArray(
                    data=depth_by_pixel,
                    dims=(CoordNames.GENERIC_PIXEL.value,),
                    attrs={
                        "long_name": f"Subdiv_depth of {key}",
                    },
                )

        map_dataset = rectangular_skymap.to_dataset()

        # Add longitude_delta, latitude_delta to the map dataset
        map_dataset["longitude_delta"] = rectangular_skymap.spacing_deg / 2
        map_dataset["latitude_delta"] = rectangular_skymap.spacing_deg / 2

        map_attrs = {
            "Spacing_degrees": str(output_map_structure.spacing_deg),
        }

    # Get the global attributes, and then fill the sensor, tiling, etc. in the
    # format-able strings.
    map_attrs.update(cdf_attrs.get_global_attributes("imap_ultra_l2_enamap"))
    for key in ["Data_type", "Logical_source", "Logical_source_description"]:
        map_attrs[key] = map_attrs[key].format(
            sensor=ultra_sensor_number,
            tiling=output_map_structure.tiling_type.value.lower(),
            duration=map_duration,
            resolution_string=(
                f"{output_map_structure.spacing_deg:.0f}deg"
                if (
                    output_map_structure.tiling_type
                    is ena_maps.SkyTilingType.RECTANGULAR
                )
                else f"nside{output_map_structure.nside}"
            ),
            inertial_frame_short_name=inertial_frame,
            inertial_frame_long_name=inertial_frame_long_name,
        )

    # Always add the following attributes to the map
    map_attrs.update(
        {
            "Sky_tiling_type": output_map_structure.tiling_type.value,
            "Spice_reference_frame": output_map_structure.spice_reference_frame.name,
        }
    )

    # Rename any variables as necessary for L2 Map schema compliance
    # Energy at L1C is named "energy_bin_geometric_mean", but at L2 it is standardized
    # to "energy" for all instruments.
    map_dataset = map_dataset.rename({"energy_bin_geometric_mean": "energy"})

    # Add the defined attributes to the map's global attrs
    map_dataset.attrs.update(map_attrs)

    # Add the "label" coordinates to the map dataset
    for coord_var, coord_data in map_dataset.coords.items():
        if coord_var != "epoch":
            map_dataset.coords[f"{coord_var}_label"] = xr.DataArray(
                coord_data.values.astype(str),
                dims=[
                    coord_var,
                ],
                name=f"{coord_var}_label",
            )

    # Add systematic error as all zeros with shape matching statistical unc
    # TODO: update once we have information from the instrument team
    map_dataset["ena_intensity_sys_err"] = xr.zeros_like(
        map_dataset["ena_intensity_stat_unc"],
    )

    # Add epoch_delta
    map_dataset.coords["epoch_delta"] = xr.DataArray(
        [
            map_duration_ns.astype(np.int64),
        ],
        dims=(CoordNames.TIME.value,),
    )
    map_dataset.coords["epoch"].attrs["DELTA_PLUS_VAR"] = "epoch_delta"

    # Add the energy delta plus/minus to the map dataset
    energy_delta_minus, energy_delta_plus = get_energy_delta_minus_plus()
    map_dataset.coords["energy_delta_minus"] = xr.DataArray(
        energy_delta_minus,
        dims=(CoordNames.ENERGY_L2.value,),
    )
    map_dataset.coords["energy_delta_plus"] = xr.DataArray(
        energy_delta_plus,
        dims=(CoordNames.ENERGY_L2.value,),
    )

    # Add variable specific attributes to the map's data_vars and coords
    for variable in map_dataset.data_vars:
        # Skip the subdivision depth variables, as these will only be
        # present for debugging purposes
        if "subdivision_depth" in variable:
            continue

        # The longitude and latitude variables will be present only in Healpix tiled
        # map, and, as support_data, should not have schema validation
        map_dataset[variable].attrs.update(
            get_variable_attributes_optional_energy_dependence(
                cdf_attrs=cdf_attrs,
                variable_array=map_dataset[variable],
                check_schema=variable
                not in ["longitude", "latitude", "longitude_delta", "latitude_delta"],
            )
        )
    for coord_variable in map_dataset.coords:
        map_dataset[coord_variable].attrs.update(
            cdf_attrs.get_variable_attributes(
                variable_name=coord_variable,
                check_schema=False,
            )
        )

    # Adjust the dtype of obs_date to be int64
    map_dataset["obs_date"] = map_dataset["obs_date"].astype(np.int64)
    return [map_dataset]
