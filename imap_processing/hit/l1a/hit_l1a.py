"""Decommutate HIT CCSDS data and create L1a data products."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.hit_utils import (
    HitAPID,
    add_energy_variables,
    get_attribute_manager,
    get_datasets_by_apid,
    process_housekeeping_data,
)
from imap_processing.hit.l0.constants import (
    AZIMUTH_ANGLES,
    MOD_10_MAPPING,
    ZENITH_ANGLES,
)
from imap_processing.hit.l0.decom_hit import decom_hit
from imap_processing.spice.time import (
    et_to_datetime64,
    met_to_datetime64,
    ttj2000ns_to_et,
)

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)

# Fill value for missing data
fillval = -9223372036854775808

# Processing dates [start, end) for which the sectored counts product is not
# created. The end date is exclusive. Before 2026-05-19 the HIT team supplied
# custom L0 files with no valid hdr_minute_cnt, so sectored counts cannot be
# produced from them. The start is the day before launch rather than an open
# bound so that tests, which use data from 2010, still produce sectored counts.
SECTORED_UNAVAILABLE_DATE_RANGE = (datetime(2025, 9, 23), datetime(2026, 5, 19))


def hit_l1a(packet_file: Path, packet_date: str | None) -> list[xr.Dataset]:
    """
    Will process HIT L0 data into L1A data products.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    packet_date : str
        The date of the packet data in 'YYYYMMDD' format. This is used to filter
        data to the correct processing day since L0 will have a 20-minute
        buffer before and after the processing day.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of Datasets of L1A processed data.
    """
    if not packet_date:
        raise ValueError("Packet date is required for processing L1A data.")

    # Unpack ccsds file to xarray datasets
    datasets_by_apid = get_datasets_by_apid(str(packet_file))

    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager("l1a")

    # Process l1a data products
    l1a_datasets = []
    if HitAPID.HIT_HSKP in datasets_by_apid:
        logger.info("Creating HIT L1A housekeeping dataset")
        hk_dataset = process_housekeeping_data(
            datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr, "imap_hit_l1a_hk"
        )
        # filter the housekeeping dataset to the processing day
        hk_dataset = filter_dataset_to_processing_day(
            hk_dataset, str(packet_date), epoch_vals=hk_dataset["epoch"].values
        )
        l1a_datasets.append(hk_dataset)
    if HitAPID.HIT_SCIENCE in datasets_by_apid:
        l1a_datasets.extend(
            process_science(
                datasets_by_apid[HitAPID.HIT_SCIENCE], attr_mgr, str(packet_date)
            )
        )
    return l1a_datasets


def subcom_sectorates(sci_dataset: xr.Dataset) -> xr.Dataset:
    """
    Subcommutate sectorates data.

    Sectored rates data contains raw counts for 5 species and 10
    energy ranges. This function subcommutates the sectored
    rates data by organizing the counts by species. Which
    species and energy range the data belongs to is determined
    by taking the mod 10 value of the corresponding header
    minute count value in the dataset. A mapping of mod 10
    values to species and energy ranges is provided in constants.py.

    MOD_10_MAPPING = {
        0: {"species": "h", "energy_min": 1.8, "energy_max": 3.6},
        1: {"species": "h", "energy_min": 4, "energy_max": 6},
        2: {"species": "h", "energy_min": 6, "energy_max": 10},
        3: {"species": "he4", "energy_min": 4, "energy_max": 6},
        ...
        9: {"species": "fe", "energy_min": 4, "energy_max": 12}}

    The data is added to the dataset as new data fields named
    according to their species. They have 4 dimensions: epoch
    energy mean, azimuth, and zenith. The energy mean
    dimension is used to distinguish between the different energy
    ranges the data belongs to. The energy deltas for each species
    are also added to the dataset as new data fields.

    Parameters
    ----------
    sci_dataset : xarray.Dataset
        Xarray dataset containing parsed HIT science data.

    Returns
    -------
    sci_dataset : xarray.Dataset
        Xarray dataset with sectored rates data organized by species.
    """
    # Initialize the dataset with the required variables
    updated_dataset = sci_dataset[
        [
            "sectorates",
            "hdr_minute_cnt",
            "livetime_counter",
            "hdr_dynamic_threshold_state",
        ]
    ].copy(deep=True)

    # Calculate mod 10 values
    hdr_min_count_mod_10 = updated_dataset.hdr_minute_cnt.values % 10

    # Reference mod 10 mapping to initialize data structure for species and
    # energy ranges and add arrays with fill values for each science frame.
    num_frames = len(hdr_min_count_mod_10)
    data_by_species_and_energy_range = {
        key: {
            **value,
            "counts": np.full(
                (num_frames, len(AZIMUTH_ANGLES), len(ZENITH_ANGLES)),
                fill_value=fillval,
                dtype=np.int64,
            ),
        }
        for key, value in MOD_10_MAPPING.items()
    }

    # Update counts for science frames where data is available
    for i, mod_10 in enumerate(hdr_min_count_mod_10):
        # NOTE: this ignore is needed to avoid a mypy error in Git
        # tests.
        data_by_species_and_energy_range[mod_10]["counts"][i] = updated_dataset[  # type: ignore[index, unused-ignore]
            "sectorates"
        ].values[i]

    # H has 3 energy ranges, 4He, CNO, NeMgSi have 2, and Fe has 1.
    # Aggregate sectored rates and energy min/max values for each species.
    # First, initialize dictionaries to store rates and min/max energy values by species
    data_by_species: dict = {
        value["species"]: {"counts": [], "energy_min": [], "energy_max": []}
        for value in data_by_species_and_energy_range.values()
    }

    for value in data_by_species_and_energy_range.values():
        species = value["species"]
        data_by_species[species]["counts"].append(value["counts"])
        data_by_species[species]["energy_min"].append(value["energy_min"])
        data_by_species[species]["energy_max"].append(value["energy_max"])

    # Add sectored rates by species to the dataset
    for species, data in data_by_species.items():
        # Rates data has shape: energy_mean, epoch, azimuth, zenith
        # Convert rates to numpy array and transpose axes to get
        # shape: epoch, energy_mean, azimuth, zenith
        rates_data = np.transpose(np.array(data["counts"]), axes=(1, 0, 2, 3))

        updated_dataset[f"{species}_sectored_counts"] = xr.DataArray(
            data=rates_data,
            dims=["epoch", f"{species}_energy_mean", "azimuth", "zenith"],
            name=f"{species}_counts_sectored",
        )

        # Add energy mean and deltas for each species
        updated_dataset = add_energy_variables(
            updated_dataset,
            species,
            np.array(data["energy_min"]),
            np.array(data["energy_max"]),
        )

    return updated_dataset


def calculate_uncertainties(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate statistical uncertainties.

    Calculate the upper and lower uncertainties. The uncertainty for
    the raw Lev1A HIT data will be calculated as asymmetric Poisson
    uncertainty as prescribed in Gehrels 1986 (DOI: 10.1086/164079).
    See section 5.5 in the algorithm document for details.

    The upper uncertainty will be calculated as
        uncert_plus = sqrt(counts + 1) + 1

    The lower uncertainty will be calculated as
        uncert_minus = sqrt(counts)

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing counts data.

    Returns
    -------
    dataset : xarray.Dataset
        The dataset with added uncertainties for each counts data variable.
    """
    # Variables that aren't counts data and should be skipped in the calculation
    ignore_vars = [
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "hdr_unit_num",
        "hdr_frame_version",
        "hdr_dynamic_threshold_state",
        "hdr_leak_conv",
        "hdr_heater_duty_cycle",
        "hdr_code_ok",
        "hdr_minute_cnt",
        "livetime_counter",
        "h_energy_delta_minus",
        "h_energy_delta_plus",
        "he4_energy_delta_minus",
        "he4_energy_delta_plus",
        "cno_energy_delta_minus",
        "cno_energy_delta_plus",
        "nemgsi_energy_delta_minus",
        "nemgsi_energy_delta_plus",
        "fe_energy_delta_minus",
        "fe_energy_delta_plus",
    ]

    # Counts data that need uncertainties calculated
    count_vars = set(dataset.data_vars) - set(ignore_vars)

    # Calculate uncertainties for counts data variables.
    # Arrays with fill values (i.e. missing data) are skipped in this calculation
    # but are kept in the new data arrays to retain shape and dimensions.
    for var in count_vars:
        mask = dataset[var] != fillval  # Mask for valid values
        # Ensure that the values are positive before taking the square root
        safe_values_plus = np.maximum(dataset[var] + 1, 0).astype(np.float32)
        safe_values_minus = np.maximum(dataset[var], 0).astype(np.float32)

        dataset[f"{var}_stat_uncert_plus"] = xr.DataArray(
            np.where(
                mask, np.sqrt(safe_values_plus) + 1, dataset[var].astype(np.float32)
            ),
            dims=dataset[var].dims,
        )
        dataset[f"{var}_stat_uncert_minus"] = xr.DataArray(
            np.where(mask, np.sqrt(safe_values_minus), dataset[var].astype(np.float32)),
            dims=dataset[var].dims,
        )
    return dataset


def add_cdf_attributes(
    dataset: xr.Dataset, logical_source: str, attr_mgr: ImapCdfAttributes
) -> xr.Dataset:
    """
    Add attributes to the dataset.

    This function adds attributes to the dataset variables and dimensions.
    It also adds dimension labels as coordinates to the dataset.The attributes
    are defined in a YAML file and retrieved by the attribute manager.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to update.
    logical_source : str
        The logical source of the dataset.
    attr_mgr : ImapCdfAttributes
        The attribute manager to retrieve attributes.

    Returns
    -------
    xarray.Dataset
        The updated dataset with attributes and dimension labels.
    """
    dataset.attrs = attr_mgr.get_global_attributes(logical_source)

    # Assign attributes and dimensions to each data array in the Dataset
    for var in dataset.data_vars.keys():
        try:
            if "energy_delta" in var or var in {
                "pkt_len",
                "version",
                "type",
                "src_seq_ctr",
                "seq_flgs",
                "pkt_apid",
                "sec_hdr_flg",
            }:
                # skip schema check to avoid DEPEND_0 being added unnecessarily
                dataset[var].attrs = attr_mgr.get_variable_attributes(
                    var, check_schema=False
                )
            else:
                dataset[var].attrs = attr_mgr.get_variable_attributes(var)
        except KeyError:
            logger.warning(f"Field {var} not found in attribute manager.")

    # check_schema=False to avoid attr_mgr adding stuff dimensions don't need
    for dim in dataset.dims:
        dataset[dim].attrs = attr_mgr.get_variable_attributes(dim, check_schema=False)
        if dim != "epoch":
            label_array = xr.DataArray(
                dataset[dim].values.astype(str),
                name=f"{dim}_label",
                dims=[dim],
                attrs=attr_mgr.get_variable_attributes(
                    f"{dim}_label", check_schema=False
                ),
            )
            dataset.coords[f"{dim}_label"] = label_array
    return dataset


def find_mod10_set_ids(hdr_minute_cnt: np.ndarray) -> np.ndarray:
    """
    Assign a set id to each frame of sectored counts.

    A set is nominally 10 frames (10 minutes) with mod 10 values 0-9. A new set
    starts wherever the mod 10 value decreases. Sets can be incomplete, e.g. 9
    frames, if a frame was dropped.

    Parameters
    ----------
    hdr_minute_cnt : numpy.ndarray
        The header minute count data.

    Returns
    -------
    np.ndarray
        Set id for each frame in the dataset, starting at 0.
    """
    # Identify 10-minute intervals of sectored counts
    # by using the mod 10 values of the header minute counts.
    # Mod 10 determines the species and energy bin the data belongs
    # to. A mapping of mod 10 values to species and energy bins is
    # provided in l0/constants.py for reference.
    mod_10: np.ndarray = hdr_minute_cnt % 10

    # Assign frame a set ID
    # E.g. if we have 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9
    # The set_ids would be [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
    # A set can be incomplete if a frame was dropped so we can not just look for
    # complete sets [0-9].
    ids: np.ndarray = np.cumsum(np.diff(mod_10, prepend=0) < 0)
    return ids


def subset_sectored_counts(
    sectored_counts_dataset: xr.Dataset, packet_date: str
) -> xr.Dataset:
    """
    Subset data for sets of sectored counts and corresponding livetime values.

    A set of sectored data starts with hydrogen and ends with iron and correspond to
    the mod 10 values 0-9. The livetime values from the previous set are used
    to calculate the rates for each set since those counts are transmitted 10 minutes
    after they were collected. Sets with a dropped frame are kept. The first set is
    only used for the livetime values of the next set.

    Parameters
    ----------
    sectored_counts_dataset : xarray.Dataset
        The sectored counts dataset.

    packet_date : str
        The date of the packet data in 'YYYYMMDD' format, used for filtering.

    Returns
    -------
    xarray.Dataset
        A dataset of sectored counts and corresponding livetime values
        for the processing day.
    """
    # Modify livetime_counter to use a new epoch coordinate
    # that is aligned with the original epoch dimension. This
    # ensures that livetime doesn't get filtered when the original
    # epoch dimension is filtered for sets.
    sectored_counts_dataset = update_livetime_coord(sectored_counts_dataset)

    # Get set id for each frame
    # The first set is set -1 and is only used for livetime values.
    frame_set_ids: np.ndarray = (
        find_mod10_set_ids(sectored_counts_dataset.hdr_minute_cnt.values) - 1
    )
    start_indices = np.flatnonzero(np.diff(frame_set_ids) > 0) + 1
    if start_indices.size == 0:
        raise ValueError(
            "No data to process - valid start indices not found for sectored counts."
        )

    # Subset data for sets of sectored counts.
    # Each set of sectored counts is generally expected to be 10 minutes long,
    # but we can have incomplete sets where a frame was dropped so we take the indices
    # starting from the first start index to the end.
    # This creates a 1D array of indices that correspond to the
    # sets of sectored counts which is used to filter the L1A dataset and
    # create the L1B sectored rates dataset.
    complete_sectored_counts_dataset = sectored_counts_dataset.isel(
        epoch=slice(start_indices[0], None)
    )

    filtered_frame_epochs = complete_sectored_counts_dataset.epoch.values
    filtered_frame_set_ids = frame_set_ids[start_indices[0] :]

    # Get the mean epoch of each set
    set_counts = np.unique(filtered_frame_set_ids, return_counts=True)[1]
    set_sums = np.bincount(filtered_frame_set_ids, weights=filtered_frame_epochs)
    mean_epoch_per_set: np.ndarray = np.repeat(set_sums / set_counts, set_counts)

    # Filter dataset for data in the processing day
    # Trim the sectored data to epoch_per_set values in the processing day
    filtered_dataset = filter_dataset_to_processing_day(
        complete_sectored_counts_dataset, packet_date, epoch_vals=mean_epoch_per_set
    )

    # Keep the livetime values from the set before each set of sectored counts
    filtered_dataset = subset_livetime(
        filtered_dataset, sectored_counts_dataset.epoch.values, frame_set_ids
    )

    return filtered_dataset


def update_livetime_coord(sectored_dataset: xr.Dataset) -> xr.Dataset:
    """
    Update livetime_counter to use a new epoch coordinate.

    Assign a new epoch coordinate to the `livetime_counter` variable.
    This new coordinate is aligned with the original `epoch` dimension,
    ensuring that `livetime_counter` remains unaffected when the original
    `epoch` dimension is filtered for sets in `subset_sectored_counts`
    function.

    Parameters
    ----------
    sectored_dataset : xarray.Dataset
        The dataset containing sectored counts and livetime_counter data.

    Returns
    -------
    xarray.Dataset
        The updated dataset with modified livetime_counter.
    """
    epoch_values = sectored_dataset.epoch.values
    sectored_dataset = sectored_dataset.assign_coords(
        {
            "epoch_livetime": ("epoch", epoch_values),
        }
    )
    da = sectored_dataset["livetime_counter"]
    da = da.assign_coords(epoch_livetime=("epoch", epoch_values))

    # Swap the dimension from 'epoch' to 'epoch_livetime'
    da = da.swap_dims({"epoch": "epoch_livetime"})

    # Update the dataset with the modified variable
    sectored_dataset["livetime_counter"] = da

    return sectored_dataset


def subset_livetime(
    dataset: xr.Dataset, all_frame_epochs: np.ndarray, frame_set_ids: np.ndarray
) -> xr.Dataset:
    """
    Trim livetime to the values from the set before each set of sectored counts.

    Sectored counts are collected over 10 minutes and transmitted over the next
    10 minutes, so the counts in set k use the livetime values from the frames
    in set k-1. Livetime is matched by set number rather than by position so
    that sets with missing frames are paired with the correct livetime values.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing sectored counts and livetime data.
    all_frame_epochs : np.ndarray
        Epoch values for all frames, including the set before the first set of
        sectored counts.
    frame_set_ids : np.ndarray
        Set number for each value in `frame_epochs`.

    Returns
    -------
    xarray.Dataset
        The updated dataset with trimmed livetime data.
    """
    if not dataset["epoch"].size:
        raise ValueError(
            "Epoch values are empty. Cannot proceed with livetime subsetting."
        )
    # Determine for each frame whether the epoch was kept (in the processing day)
    kept_epochs = np.isin(all_frame_epochs, dataset["epoch"].values)
    # get all the kept set ids
    kept_sets = np.unique(frame_set_ids[kept_epochs])
    # Livetime values for the set comes from the frames in the previous set
    livetime_sets = np.isin(frame_set_ids, kept_sets - 1)
    livetime_ids = np.flatnonzero(
        np.isin(dataset["epoch_livetime"].values, all_frame_epochs[livetime_sets])
    )
    return dataset.isel(epoch_livetime=livetime_ids)


def filter_dataset_to_processing_day(
    dataset: xr.Dataset,
    packet_date: str,
    epoch_vals: np.ndarray,
    sc_tick: bool = False,
) -> xr.Dataset:
    """
    Filter the dataset for data within the processing day.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to filter.
    packet_date : str
        The date of the packet data in 'YYYYMMDD' format.
    epoch_vals : np.ndarray
        An array of epoch values. Used to identify indices of data that
        belong in the processing day. For sectored counts data, an
        array of mean epoch values for major frames (10 min. intervals)
        is used to filter the dataset to ensure that major frames that span
        midnight, but belong to the processing day, are included. For other
        datasets, the dataset's epoch coordinate values will be used.
    sc_tick : bool
        If true, the dataset's sc_tick will be used to filter data as well.
        This ensures that the ccsds headers that use sc_tick as a coordinate,
        instead of epoch, also corresponds to the processing day.

    Returns
    -------
    xarray.Dataset
        The filtered dataset containing data within the processing day.
    """
    processing_day = datetime.strptime(packet_date, "%Y%m%d").strftime("%Y-%m-%d")

    # Filter dataset by epoch indices in processing day
    epoch_dt = et_to_datetime64(ttj2000ns_to_et(epoch_vals))
    epoch_indices_in_processing_day = np.where(
        epoch_dt.astype("datetime64[D]") == np.datetime64(processing_day)
    )[0]
    dataset = dataset.isel(epoch=epoch_indices_in_processing_day)

    # If sc_tick is provided (coord for ccsds headers), filter by sc_tick too
    if sc_tick:
        sc_tick_dt = met_to_datetime64(dataset["sc_tick"].values)
        indices_in_processing_day = np.where(
            sc_tick_dt.astype("datetime64[D]") == np.datetime64(processing_day)
        )[0]
        dataset = dataset.isel(sc_tick=indices_in_processing_day)
    return dataset


def process_science(
    dataset: xr.Dataset, attr_mgr: ImapCdfAttributes, packet_date: str
) -> list[xr.Dataset]:
    """
    Will process science datasets for CDF products.

    The function processes binary science data for CDF creation.
    The data is decommutated, decompressed, grouped into science frames,
    and split into count rates, sectored count rates, and event datasets.
    It also updates the dataset attributes according to specifications
    in a cdf yaml file.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset containing HIT science data.
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.
    packet_date : str
        The date of the packet data, used for processing.

    Returns
    -------
    dataset : list
        A list of science datasets ready for CDF conversion.
    """
    logger.info("Creating HIT L1A science datasets")

    # Decommutate and decompress the science data
    sci_dataset = decom_hit(dataset)

    # Skip the sectored counts product for dates without sectored data
    sectored_start, sectored_end = SECTORED_UNAVAILABLE_DATE_RANGE
    processing_day = datetime.strptime(packet_date, "%Y%m%d")
    create_sectored = not (sectored_start <= processing_day < sectored_end)

    sectored_count_rates_dataset = None
    if create_sectored:
        # Create dataset for sectored data organized by species type.
        # This uses the unfiltered science data since sectored counts need the
        # buffer of frames before the processing day for livetime.
        sectored_dataset = subcom_sectorates(sci_dataset)

        # Subset sectored data for complete sets (10 min intervals covering all
        # species)
        sectored_dataset = subset_sectored_counts(sectored_dataset, packet_date)
        sectored_count_rates_dataset = calculate_uncertainties(sectored_dataset)
    else:
        logger.info(
            f"Skipping HIT L1A sectored counts for {packet_date}: sectored data is "
            f"not available from {sectored_start.date()} to {sectored_end.date()} "
            "(end exclusive)"
        )

    # TODO:
    #  - headers are values per packet rather than per frame. Do these need to align
    #    with the science frames?
    #    For instance, the mean epoch for a frame that spans midnight might contain
    #    packets from the previous day but filtering sc_tick by processing day will
    #    exclude those packets. Is this an issue?

    # Filter the science dataset to only include data from the processing day
    sci_dataset = filter_dataset_to_processing_day(
        sci_dataset, packet_date, epoch_vals=sci_dataset["epoch"].values, sc_tick=True
    )

    # Split the science data into count rates and event datasets
    pha_raw_dataset = xr.Dataset(
        {"pha_raw": sci_dataset["pha_raw"]}, coords={"epoch": sci_dataset["epoch"]}
    )
    count_rates_dataset = sci_dataset.drop_vars(["pha_raw", "sectorates"])

    # Calculate uncertainties for count rates
    count_rates_dataset = calculate_uncertainties(count_rates_dataset)

    l1a_datasets: dict = {
        "imap_hit_l1a_counts-standard": count_rates_dataset,
        "imap_hit_l1a_direct-events": pha_raw_dataset,
    }
    if sectored_count_rates_dataset is not None:
        l1a_datasets["imap_hit_l1a_counts-sectored"] = sectored_count_rates_dataset

    # Update attributes and dimensions
    for logical_source, ds in l1a_datasets.items():
        l1a_datasets[logical_source] = add_cdf_attributes(ds, logical_source, attr_mgr)

        logger.info(f"HIT L1A dataset created for {logical_source}")

    return list(l1a_datasets.values())
