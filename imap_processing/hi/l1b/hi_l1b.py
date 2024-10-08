"""IMAP-HI L1B processing module."""

import logging
from enum import IntEnum

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hi.utils import HIAPID, HiConstants, create_dataset_variables
from imap_processing.utils import convert_raw_to_eu

logger = logging.getLogger(__name__)
ATTR_MGR = ImapCdfAttributes()
ATTR_MGR.add_instrument_global_attrs("hi")
ATTR_MGR.load_variable_attributes("imap_hi_variable_attrs.yaml")


class TriggerId(IntEnum):
    """Int Enum class for trigger id values."""

    A = 1
    B = 2
    C = 3


class CoincidenceBitmap(IntEnum):
    """Int Enum class for coincidence type bitmap values."""

    A = 2**3
    B = 2**2
    C1 = 2**1
    C2 = 2**0


def hi_l1b(l1a_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    High level IMAP-HI L1B processing function.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        L1A dataset to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        Processed xarray dataset.
    """
    logger.info(
        f"Running Hi L1B processing on dataset: {l1a_dataset.attrs['Logical_source']}"
    )
    logical_source_parts = l1a_dataset.attrs["Logical_source"].split("_")
    # TODO: apid is not currently stored in all L1A data but should be.
    #    Use apid to determine what L1B processing function to call

    # Housekeeping processing
    if logical_source_parts[-1].endswith("hk"):
        # if packet_enum in (HIAPID.H45_APP_NHK, HIAPID.H90_APP_NHK):
        packet_enum = HIAPID(l1a_dataset["pkt_apid"].data[0])
        conversion_table_path = str(
            imap_module_directory / "hi" / "l1b" / "hi_eng_unit_convert_table.csv"
        )
        l1b_dataset = convert_raw_to_eu(
            l1a_dataset,
            conversion_table_path=conversion_table_path,
            packet_name=packet_enum.name,
            comment="#",
            converters={"mnemonic": str.lower},
        )

        l1b_dataset.attrs.update(ATTR_MGR.get_global_attributes("imap_hi_l1b_hk_attrs"))
    elif logical_source_parts[-1].endswith("de"):
        l1b_dataset = annotate_direct_events(l1a_dataset)
    else:
        raise NotImplementedError(
            f"No Hi L1B processing defined for file type: "
            f"{l1a_dataset.attrs['Logical_source']}"
        )
    # Update global attributes
    # TODO: write a function that extracts the sensor from Logical_source
    #    some functionality can be found in imap_data_access.file_validation but
    #    only works on full file names
    sensor_str = logical_source_parts[-1].split("-")[0]
    l1b_dataset.attrs["Logical_source"] = l1b_dataset.attrs["Logical_source"].format(
        sensor=sensor_str
    )
    # TODO: revisit this
    l1b_dataset.attrs["Data_version"] = data_version
    return l1b_dataset


def annotate_direct_events(l1a_dataset: xr.Dataset) -> xr.Dataset:
    """
    Perform Hi L1B processing on direct event data.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        L1A direct event data.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        L1B direct event data.
    """
    l1b_dataset = compute_coincidence_type_and_time_deltas(l1a_dataset)
    l1b_de_var_names = [
        "esa_energy_step",
        "spin_phase",
        "hae_latitude",
        "hae_longitude",
        "quality_flag",
        "nominal_bin",
    ]
    new_data_vars = create_dataset_variables(
        l1b_de_var_names, l1a_dataset["epoch"].size, att_manager_lookup_str="hi_de_{0}"
    )
    l1b_dataset = l1b_dataset.assign(new_data_vars)
    l1b_dataset = l1b_dataset.drop_vars(
        ["tof_1", "tof_2", "tof_3", "de_tag", "ccsds_met", "meta_event_met"]
    )

    de_global_attrs = ATTR_MGR.get_global_attributes("imap_hi_l1b_de_attrs")
    l1b_dataset.attrs.update(**de_global_attrs)
    return l1b_dataset


def compute_coincidence_type_and_time_deltas(dataset: xr.Dataset) -> xr.Dataset:
    """
    Compute coincidence type and time deltas.

    Adds the new variables "coincidence_type", "delta_t_ab", "delta_t_ac1",
    "delta_t_bc1", and "delta_t_c1c2" to the input xarray.Dataset and returns
    the result.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L1A/B dataset that results from reading in the L1A CDF and
        allocating the new L1B DataArrays.

    Returns
    -------
    xr.Dataset
        Input `dataset` is modified in-place.
    """
    new_data_vars = create_dataset_variables(
        [
            "coincidence_type",
            "delta_t_ab",
            "delta_t_ac1",
            "delta_t_bc1",
            "delta_t_c1c2",
        ],
        len(dataset.epoch),
        "hi_de_{0}",
    )
    out_ds = dataset.assign(new_data_vars)

    # compute masks needed for coincidence type and delta t calculations
    a_mask = out_ds.trigger_id.values == TriggerId.A
    b_mask = out_ds.trigger_id.values == TriggerId.B
    c_mask = out_ds.trigger_id.values == TriggerId.C

    tof_1_valid_mask = np.isin(
        out_ds.tof_1.values, HiConstants.TOF1_BAD_VALUES, invert=True
    )
    tof_2_valid_mask = np.isin(
        out_ds.tof_2.values, HiConstants.TOF2_BAD_VALUES, invert=True
    )
    tof_1and2_valid_mask = tof_1_valid_mask & tof_2_valid_mask
    tof_3_valid_mask = np.isin(
        out_ds.tof_3.values, HiConstants.TOF3_BAD_VALUES, invert=True
    )

    # Table denoting how hit-first mask and valid TOF masks are used to set
    # coincidence type bitmask
    # -----------------------------------------------------------------------
    # | Trigger ID  |  Hit First  | TOF 1 Valid | TOF 2 Valid | TOF 3 Valid |
    # -----------------------------------------------------------------------
    # |      1      |      A      |      B      |      C1     |      C2     |
    # |      2      |      B      |      A      |      C1     |      C2     |
    # |      3      |      C1     |      A      |      B      |      C2     |
    # Set coincidence type bitmask
    out_ds.coincidence_type[a_mask | tof_1_valid_mask] |= CoincidenceBitmap.A.value
    out_ds.coincidence_type[
        b_mask | (a_mask & tof_1_valid_mask) | (c_mask & tof_2_valid_mask)
    ] |= CoincidenceBitmap.B.value
    out_ds.coincidence_type[
        c_mask | (a_mask & tof_2_valid_mask) | (b_mask & tof_2_valid_mask)
    ] |= CoincidenceBitmap.C1.value
    out_ds.coincidence_type[tof_3_valid_mask] |= CoincidenceBitmap.C2.value

    # Table denoting how TOF is interpreted for each Trigger ID
    # -----------------------------------------------------------------------
    # | Trigger ID  |  Hit First  |    TOF 1    |    TOF 2    |    TOF 3    |
    # -----------------------------------------------------------------------
    # |      1      |      A      |  t_b - t_a  | t_c1 - t_a  | t_c2 - t_c1 |
    # |      2      |      B      |  t_a - t_b  | t_c1 - t_b  | t_c2 - t_c1 |
    # |      3      |      C      |  t_a - t_c1 | t_b  - t_c1 | t_c2 - t_c1 |
    # delta_t_ab
    out_ds.delta_t_ab.values[a_mask & tof_1_valid_mask] = (
        out_ds.tof_1.values[a_mask & tof_1_valid_mask].astype(np.float32)
        * HiConstants.TOF1_TICK_PER_NS
    )
    out_ds.delta_t_ab.values[b_mask & tof_1_valid_mask] = (
        -out_ds.tof_1.values[b_mask & tof_1_valid_mask].astype(np.float32)
        * HiConstants.TOF1_TICK_PER_NS
    )
    out_ds.delta_t_ab.values[c_mask & tof_1and2_valid_mask] = (
        out_ds.tof_2.values[c_mask & tof_1and2_valid_mask].astype(np.float32)
        * HiConstants.TOF2_TICK_PER_NS
        - out_ds.tof_1.values[c_mask & tof_1and2_valid_mask].astype(np.float32)
        * HiConstants.TOF1_TICK_PER_NS
    )

    # delta_t_ac1
    out_ds.delta_t_ac1.values[a_mask & tof_2_valid_mask] = (
        out_ds.tof_2.values[a_mask & tof_2_valid_mask].astype(np.float32)
        * HiConstants.TOF2_TICK_PER_NS
    )
    out_ds.delta_t_ac1.values[b_mask & tof_1and2_valid_mask] = (
        out_ds.tof_2.values[b_mask & tof_1and2_valid_mask]
        * HiConstants.TOF2_TICK_PER_NS
        - out_ds.tof_1.values[b_mask & tof_1and2_valid_mask]
        * HiConstants.TOF1_TICK_PER_NS
    )
    out_ds.delta_t_ac1.values[c_mask & tof_1_valid_mask] = (
        -out_ds.tof_1.values[c_mask & tof_1_valid_mask] * HiConstants.TOF1_TICK_PER_NS
    )

    # delta_t_bc1
    out_ds.delta_t_bc1.values[a_mask & tof_1_valid_mask & tof_2_valid_mask] = (
        out_ds.tof_2.values[a_mask & tof_1and2_valid_mask]
        * HiConstants.TOF2_TICK_PER_NS
        - out_ds.tof_1.values[a_mask & tof_1and2_valid_mask]
        * HiConstants.TOF1_TICK_PER_NS
    )
    out_ds.delta_t_bc1.values[b_mask & tof_2_valid_mask] = (
        out_ds.tof_2.values[b_mask & tof_2_valid_mask] * HiConstants.TOF2_TICK_PER_NS
    )
    out_ds.delta_t_bc1.values[c_mask & tof_2_valid_mask] = (
        -out_ds.tof_2.values[c_mask & tof_2_valid_mask] * HiConstants.TOF2_TICK_PER_NS
    )

    # delta_t_c1c2
    out_ds.delta_t_c1c2.values[tof_3_valid_mask] = (
        out_ds.tof_3.values[tof_3_valid_mask] * HiConstants.TOF3_TICK_PER_NS
    )

    return out_ds
