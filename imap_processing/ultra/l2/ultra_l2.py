"""Calculate ULTRA Level 2 (L2) ENA Map Product."""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils.map_properties import MapProperties
from imap_processing.spice import geometry

logger = logging.getLogger(__name__)
logger.info("Importing ultra_l2 module")


PSET_SPICE_FRAME = geometry.SpiceFrame.IMAP_DPS
VALUES_TO_PUSH = ["counts", "exposure_time", "sensitivity"]

DEFAULT_L2_NSIDE = 128
DEFAULT_L2_NESTED = False


def read_into_pointing_set(
    input_data: xr.Dataset | str | Path,
) -> ena_maps.UltraPointingSet:
    # Allow for passing in either xarray Datasets, or paths to the CDF files
    if isinstance(input_data, xr.Dataset):
        return ena_maps.UltraPointingSet(
            l1c_dataset=input_data, spice_reference_frame=PSET_SPICE_FRAME
        )
    elif isinstance(input_data, str | Path):
        return ena_maps.UltraPointingSet(
            l1c_dataset=load_cdf(input_data), spice_reference_frame=PSET_SPICE_FRAME
        )
    else:
        raise ValueError(
            f"Input data must be either an xarray Dataset or a path to a CDF file "
            "containing the L1C dataset.\n"
            f"Found {type(input_data)} instead."
        )


def generate_healpix_skymap(
    ultra_l1c_psets: list[str],
    map_spice_reference_frame: geometry.SpiceFrame = geometry.SpiceFrame.ECLIPJ2000,
    map_nside: int = DEFAULT_L2_NSIDE,
    map_nested: bool = DEFAULT_L2_NESTED,
):
    skymap = ena_maps.HealpixSkyMap(
        nside=map_nside, nested=map_nested, spice_frame=map_spice_reference_frame
    )

    for ultra_l1c_pset in ultra_l1c_psets:
        pointing_set = read_into_pointing_set(ultra_l1c_pset)
        skymap.project_pset_values_to_map(
            pointing_set=pointing_set,
            value_keys=VALUES_TO_PUSH,
            index_match_method=ena_maps.IndexMatchMethod.PUSH,
        )
    return skymap


def ultra_l2(
    data_dict: dict[str, xr.Dataset | str],
    data_version: str,
    output_map_properties: MapProperties,
) -> list[xr.Dataset]:
    """
    Generate Ultra L2 Product from L1C Products.

    Parameters
    ----------
    data_dict : dict[str, xr.Dataset]
        Dict mapping l1c product identifiers to paths/Datasets containing l1c psets.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    list[xarray.Dataset,]
        L2 output dataset containing map of the counts on the sky.
        Wrapped in a list for consistency with other product levels.
    """
    # regardless of the output sky tiling type, we will directly
    # project the PSET values into a healpix map
    healpix_map = generate_healpix_skymap(
        ultra_l1c_psets=list(data_dict.values()),
    )

    return healpix_map.to_dataset()


#     l1c_product_names, l1c_products = zip(*data_dict.items())
#     num_l1c_products = len(l1c_products)
#     logger.info(
#         f"Running ultra_l2 processing on the following {num_l1c_products} L1C products:"
#         f"\n{l1c_product_names}"
#     )
#     frame_epochs = np.unique([l1c_product.epoch for l1c_product in l1c_products])
#     if len(frame_epochs) != num_l1c_products:
#         raise ValueError(
#             "All L1C products must have unique epochs. "
#             f"Found {num_l1c_products} products with {len(frame_epochs)} unique epochs."
#         )

#     rect_sky_map = ena_maps.RectangularSkyMap(
#         spacing_deg=l2_spacing_deg,
#         spice_frame=geometry.SpiceFrame.ECLIPJ2000,
#     )

#     for prod_num, l1c_prod in enumerate(l1c_products):
#         # TODO: Determine if the ultra45 / ultra90 distinction is necessary
#         head = "45" if is_ultra45(l1c_prod) else "90"

#         time = float(l1c_prod.epoch.values)
#         ultra_pointing_set = ena_maps.UltraPointingSet(
#             l1c_dataset=l1c_prod, spice_reference_frame=geometry.SpiceFrame.IMAP_DPS
#         )


#         # Some quantities need to be exposure time weighted means:
#         for quantity_to_weight, quantity_to_weight_by in [
#             ("sensitivity", "exposure_time"),
#             ("counts", "solid_angle"),
#         ]:
#             ultra_pointing_set.data[
#                 f"{quantity_to_weight_by}_times_{quantity_to_weight}"
#             ] = (
#                 ultra_pointing_set.data[quantity_to_weight]
#                 * ultra_pointing_set.data[quantity_to_weight_by]
#             )
#         logger.info(
#             f"Pushing/Pulling values for ultra{head} product at time index {prod_num} "
#             f"with epoch {time}.\n\t{ultra_pointing_set!r}"
#         )
#         # Apply any "push"ing of values from the PSET --> SkyMap
#         rect_sky_map.project_pset_values_to_map(
#             pointing_set=ultra_pointing_set,
#             value_keys=["exposure_time", "counts", "exposure_time_times_sensitivity"],
#             skymap_value_keys=[
#                 "exposure_time_pushed",
#                 "counts_pushed",
#                 "exposure_time_times_sensitivity_pushed",
#             ],
#             index_match_method=ena_maps.IndexMatchMethod.PUSH,
#         )
#         # Apply any "pull"ing of values into the SkyMap <-- PSET
#         rect_sky_map.project_pset_values_to_map(
#             pointing_set=ultra_pointing_set,
#             value_keys=[
#                 "counts",
#                 "exposure_time",
#                 "exposure_time_times_sensitivity",
#                 "solid_angle_times_counts",
#             ],
#             skymap_value_keys=[
#                 "counts_pulled",
#                 "exposure_time_pulled",
#                 "exposure_time_times_sensitivity_pulled",
#                 "solid_angle_times_counts_pulled",
#             ],
#             index_match_method=ena_maps.IndexMatchMethod.PULL,
#         )
#     # TODO: remove this debug logging
#     logger.warning("here are all the keys in rect_sky_map.data_dict:")
#     logger.warning(rect_sky_map.data_dict.keys())

#     rect_sky_map.data_dict["count_rate_pushed"] = (
#         rect_sky_map.data_dict["counts_pushed"]
#         / rect_sky_map.data_dict["exposure_time_pushed"]
#     )
#     rect_sky_map.data_dict["count_rate_pulled"] = (
#         rect_sky_map.data_dict["counts_pulled"]
#         / rect_sky_map.data_dict["exposure_time_pulled"]
#     )

#     rect_sky_map.data_dict["mean_sensitivity_pushed"] = (
#         rect_sky_map.data_dict["exposure_time_times_sensitivity_pushed"]
#         / rect_sky_map.data_dict["exposure_time_pushed"]
#     )
#     rect_sky_map.data_dict["mean_sensitivity_pulled"] = (
#         rect_sky_map.data_dict["exposure_time_times_sensitivity_pulled"]
#         / rect_sky_map.data_dict["exposure_time_pulled"]
#     )
#     rect_sky_map.data_dict["counts_pulled_solid_angle_weighted_mean"] = (
#         rect_sky_map.data_dict["solid_angle_times_counts_pulled"] / (4 * np.pi)
#     )

#     dim_names = {
#         "az": "azimuth_bin_center",
#         "el": "elevation_bin_center",
#         "energy": "energy_bin_center",
#     }
#     rect_sky_map_ds = rect_sky_map.to_dataset(
#         output_value_keys_dims={
#             "counts_pushed": [dim_names["az"], dim_names["el"], dim_names["energy"]],
#             "counts_pulled": [dim_names["az"], dim_names["el"], dim_names["energy"]],
#             "count_rate_pushed": [
#                 dim_names["az"],
#                 dim_names["el"],
#                 dim_names["energy"],
#             ],
#             "count_rate_pulled": [
#                 dim_names["az"],
#                 dim_names["el"],
#                 dim_names["energy"],
#             ],
#             "exposure_time_pushed": [dim_names["az"], dim_names["el"]],
#             "exposure_time_pulled": [dim_names["az"], dim_names["el"]],
#             "mean_sensitivity_pushed": [
#                 dim_names["az"],
#                 dim_names["el"],
#                 dim_names["energy"],
#             ],
#             "mean_sensitivity_pulled": [
#                 dim_names["az"],
#                 dim_names["el"],
#                 dim_names["energy"],
#             ],
#             "counts_pulled_solid_angle_weighted_mean": [
#                 dim_names["az"],
#                 dim_names["el"],
#                 dim_names["energy"],
#             ],
#         },
#         global_attrs={
#             "data_version": data_version,
#             "Logical_source": "imap_ultra_l2_skymap",
#         },
#     )

#     # Add an energy coordinate to the dataset
#     rect_sky_map_ds["energy_bin_center"] = l1c_products[0].energy_bin_center

#     # Wrap the output in a list for consistency with other data product levels
#     output_datasets = [rect_sky_map_ds]
#     return output_datasets
