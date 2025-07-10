"""
Classes and functions used in HIT processing.

This module contains utility classes and functions that are used by
HIT processing modules.
"""

from enum import IntEnum

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.utils import packet_file_to_datasets


class HitAPID(IntEnum):
    """
    HIT APID Mappings.

    Attributes
    ----------
    HIT_HSKP: int
        Housekeeping
    HIT_SCIENCE : int
        Science
    HIT_IALRT : int
        I-ALiRT
    """

    HIT_HSKP = 1251
    HIT_SCIENCE = 1252
    HIT_IALRT = 1253


def get_datasets_by_apid(
    packet_file: str, derived: bool = False
) -> dict[int, xr.Dataset]:
    """
    Get datasets by APID from a CCSDS packet file.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    derived : bool, optional
        Flag to use derived values, by default False.
        Only set to True to get engineering units for L1B
        housekeeping data product.

    Returns
    -------
    datasets_by_apid : dict[int, xr.Dataset]
        Dictionary of xarray datasets by APID.
    """
    # Unpack ccsds file
    packet_definition = (
        imap_module_directory / "hit/packet_definitions/hit_packet_definitions.xml"
    )
    datasets_by_apid: dict[int, xr.Dataset] = packet_file_to_datasets(
        packet_file=packet_file,
        xtce_packet_definition=packet_definition,
        use_derived_value=derived,
    )
    return datasets_by_apid


def get_attribute_manager(level: str) -> ImapCdfAttributes:
    """
    Create an attribute manager for the HIT data products.

    Parameters
    ----------
    level : str
        Data level of the product being created.

    Returns
    -------
    attr_mgr : ImapCdfAttributes
        Attribute manager to set CDF attributes.
    """
    # Create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hit")
    attr_mgr.add_instrument_variable_attrs(instrument="hit", level=level)
    return attr_mgr


def concatenate_leak_variables(
    dataset: xr.Dataset, adc_channels: xr.DataArray
) -> xr.Dataset:
    """
    Concatenate leak variables in the dataset.

    Updates the housekeeping dataset to replace the individual
    leak_i_00, leak_i_01, ..., leak_i_63 variables with a single
    leak_i variable as a 2D array. "i" here represents current
    in the leakage current [Voltage] data.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing 64 leak variables.
    adc_channels : xarray.DataArray
        DataArray to be used as a dimension for the concatenated leak variables.

    Returns
    -------
    dataset : xarray.Dataset
        Updated dataset with concatenated leak variables.
    """
    # Stack 64 leak variables (leak_00, leak_01, ..., leak_63)
    leak_vars = [dataset[f"leak_i_{i:02d}"] for i in range(64)]

    # Concatenate along 'adc_channels' and reorder dimensions
    stacked_leaks = xr.concat(leak_vars, dim=adc_channels).transpose(
        "epoch", "adc_channels"
    )
    dataset["leak_i"] = stacked_leaks

    # Drop the individual leak variables
    updated_dataset = dataset.drop_vars([f"leak_i_{i:02d}" for i in range(64)])

    return updated_dataset


def process_housekeeping_data(
    dataset: xr.Dataset, attr_mgr: ImapCdfAttributes, logical_source: str
) -> xr.Dataset:
    """
    Will process housekeeping dataset for CDF product.

    Updates the housekeeping dataset with a single 2D leak_i
    variable. Also updates the dataset attributes, coordinates
    and data variable dimensions according to specifications in
    a cdf yaml file. This function is used for both L1A and L1B
    housekeeping data products.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing HIT housekeeping data.

    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.

    logical_source : str
        Logical source of the data -> imap_hit_l1a_hk or imap_hit_l1b_hk.

    Returns
    -------
    dataset : xarray.Dataset
        An updated dataset ready for CDF conversion.
    """
    # Drop keys that are not CDF data variables
    drop_keys = [
        "pkt_apid",
        "version",
        "type",
        "sec_hdr_flg",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "hskp_spare1",
        "hskp_spare2",
        "hskp_spare3",
        "hskp_spare4",
        "hskp_spare5",
    ]

    # Drop variables not needed for CDF
    dataset = dataset.drop_vars(drop_keys)

    # Create data arrays for dependencies
    adc_channels = xr.DataArray(
        np.arange(64, dtype=np.uint8),
        name="adc_channels",
        dims=["adc_channels"],
        attrs=attr_mgr.get_variable_attributes("adc_channels"),
    )

    # NOTE: LABL_PTR_1 should be CDF_CHAR.
    adc_channels_label = xr.DataArray(
        adc_channels.values.astype(str),
        name="adc_channels_label",
        dims=["adc_channels_label"],
        attrs=attr_mgr.get_variable_attributes("adc_channels_label"),
    )

    # Update dataset coordinates and attributes
    dataset = dataset.assign_coords(
        {
            "adc_channels": adc_channels,
            "adc_channels_label": adc_channels_label,
        }
    )
    dataset.attrs = attr_mgr.get_global_attributes(logical_source)

    # Stack 64 leak variables (leak_00, leak_01, ..., leak_63)
    dataset = concatenate_leak_variables(dataset, adc_channels)

    # Assign attributes and dimensions to each data array in the Dataset
    for field in dataset.data_vars.keys():
        # Create a dict of dimensions using the DEPEND_I keys in the
        # attributes
        dims = {
            key: value
            for key, value in attr_mgr.get_variable_attributes(field).items()
            if "DEPEND" in key
        }
        dataset[field].attrs = attr_mgr.get_variable_attributes(field)
        dataset[field].assign_coords(dims)

    dataset.epoch.attrs = attr_mgr.get_variable_attributes("epoch")

    return dataset


def initialize_particle_data_arrays(
    dataset: xr.Dataset,
    particle: str,
    num_energy_ranges: int,
    epoch_size: int,
) -> xr.Dataset:
    """
    Add empty data arrays for a given particle.

    Valid particle names:
        h
        he3
        he4
        he
        c
        n
        o
        ne
        na
        mg
        al
        si
        s
        ar
        ca
        fe
        ni

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to add the data arrays to.
    particle : str
        The abbreviated particle name.
    num_energy_ranges : int
        Number of energy ranges for the particle.
        Used to define the shape of the data arrays.
    epoch_size : int
        Used to define the shape of the data arrays.

    Returns
    -------
    updated_ds : xr.Dataset
        The updated dataset with the particle data arrays added.
    """
    updated_ds = dataset.copy()

    updated_ds[f"{particle}"] = xr.DataArray(
        data=np.zeros((epoch_size, num_energy_ranges), dtype=np.float32),
        dims=["epoch", f"{particle}_energy_mean"],
        name=f"{particle}",
    )
    updated_ds[f"{particle}_stat_uncert_minus"] = xr.DataArray(
        data=np.zeros((epoch_size, num_energy_ranges), dtype=np.float32),
        dims=["epoch", f"{particle}_energy_mean"],
        name=f"{particle}_stat_uncert_minus",
    )
    updated_ds[f"{particle}_stat_uncert_plus"] = xr.DataArray(
        data=np.zeros((epoch_size, num_energy_ranges), dtype=np.float32),
        dims=["epoch", f"{particle}_energy_mean"],
        name=f"{particle}_stat_uncert_plus",
    )
    updated_ds.coords[f"{particle}_energy_mean"] = xr.DataArray(
        np.zeros(num_energy_ranges, dtype=np.int8),
        dims=[f"{particle}_energy_mean"],
        name=f"{particle}_energy_mean",
    )

    return updated_ds


def sum_particle_data(
    dataset: xr.Dataset, indices: dict
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Sum particle data for a given energy range.

    Parameters
    ----------
    dataset : xr.Dataset
        A dataset containing particle data to sum in the l2fgrates, l3fgrates,
        penfgrates data variables. If it's an L1A dataset, these variables
        contain particle counts. If it's an L1B dataset, these variables
        contain particle rates.

    indices : dict
        A dictionary containing the indices for particle data to sum for a given
        energy range. The dictionary should have the following keys:
            R2 = indices for l2fgrates
            R3 = indices for l3fgrates
            R4 = indices for penfgrates

    Returns
    -------
    summed_data : xr.DataArray
        The summed data for the given energy range.

    summed_uncertainty_minus : xr.DataArray
        The summed data for delta minus statistical uncertainty.

    summed_uncertainty_plus : xr.DataArray
        The summed data for delta plus statistical uncertainty.
    """
    summed_data = (
        dataset["l2fgrates"][:, indices["R2"]].sum(axis=1)
        + dataset["l3fgrates"][:, indices["R3"]].sum(axis=1)
        + dataset["penfgrates"][:, indices["R4"]].sum(axis=1)
    )

    summed_uncertainty_minus = (
        dataset["l2fgrates_stat_uncert_minus"][:, indices["R2"]].sum(axis=1)
        + dataset["l3fgrates_stat_uncert_minus"][:, indices["R3"]].sum(axis=1)
        + dataset["penfgrates_stat_uncert_minus"][:, indices["R4"]].sum(axis=1)
    )

    summed_uncertainty_plus = (
        dataset["l2fgrates_stat_uncert_plus"][:, indices["R2"]].sum(axis=1)
        + dataset["l3fgrates_stat_uncert_plus"][:, indices["R3"]].sum(axis=1)
        + dataset["penfgrates_stat_uncert_plus"][:, indices["R4"]].sum(axis=1)
    )

    return summed_data, summed_uncertainty_minus, summed_uncertainty_plus


def add_energy_variables(
    dataset: xr.Dataset,
    particle: str,
    energy_min_values: np.ndarray,
    energy_max_values: np.ndarray,
) -> xr.Dataset:
    """
    Add energy min and max variables to the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to add the energy variables to.
    particle : str
        The particle name.
    energy_min_values : np.ndarray
        The minimum energy values for each energy range.
    energy_max_values : np.ndarray
        The maximum energy values for each energy range.

    Returns
    -------
    updated_ds : xr.Dataset
        The updated dataset with the energy variables added.
    """
    updated_ds = dataset.copy()

    energy_mean = np.round(
        np.mean(np.array([energy_min_values, energy_max_values]), axis=0), 3
    ).astype(np.float32)

    updated_ds[f"{particle}_energy_mean"] = xr.DataArray(
        data=energy_mean,
        dims=[f"{particle}_energy_mean"],
        name=f"{particle}_energy_mean",
    )
    updated_ds[f"{particle}_energy_delta_minus"] = xr.DataArray(
        data=np.array(energy_mean - np.array(energy_min_values), dtype=np.float32),
        dims=[f"{particle}_energy_mean"],
        name=f"{particle}_energy_delta_minus",
    )
    updated_ds[f"{particle}_energy_delta_plus"] = xr.DataArray(
        data=np.array(energy_max_values - energy_mean, dtype=np.float32),
        dims=[f"{particle}_energy_mean"],
        name=f"{particle}_energy_delta_plus",
    )
    return updated_ds


def add_summed_particle_data_to_dataset(
    dataset: xr.Dataset,
    source_dataset: xr.Dataset,
    particle: str,
    energy_ranges: list,
) -> xr.Dataset:
    """
    Add summed particle data to the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to add the rates to (not modified in-place).
    source_dataset : xr.Dataset
        The dataset containing data to sum (counts or rates).
    particle : str
        The particle name.
    energy_ranges : list
        A list of energy range dictionaries for the particle.

    Returns
    -------
    xr.Dataset
        A new dataset with summed particle data added.
    """
    # Make a copy of the dataset to update
    ds = dataset.copy()

    # Initialize particle data arrays
    ds = initialize_particle_data_arrays(
        ds, particle, len(energy_ranges), source_dataset.sizes["epoch"]
    )

    # Initialize arrays for energy values
    energy_min = np.zeros(len(energy_ranges), dtype=np.float32)
    energy_max = np.zeros(len(energy_ranges), dtype=np.float32)

    # Compute summed data and update the dataset
    for i, energy_range_dict in enumerate(energy_ranges):
        summed_data, summed_data_uncert_minus, summed_data_uncert_plus = (
            sum_particle_data(source_dataset, energy_range_dict)
        )

        ds[f"{particle}"][:, i] = summed_data.astype(np.float32)
        ds[f"{particle}_stat_uncert_minus"][:, i] = summed_data_uncert_minus.astype(
            np.float32
        )
        ds[f"{particle}_stat_uncert_plus"][:, i] = summed_data_uncert_plus.astype(
            np.float32
        )

        # Store energy range values
        energy_min[i] = energy_range_dict["energy_min"]
        energy_max[i] = energy_range_dict["energy_max"]

    # Add energy variables
    ds = add_energy_variables(ds, particle, energy_min, energy_max)

    return ds
