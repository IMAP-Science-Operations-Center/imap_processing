"""IMAP-HI L1B processing module."""

import logging
from enum import IntEnum

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import parse_filename_like
from imap_processing.hi.utils import (
    HIAPID,
    HiConstants,
    create_dataset_variables,
    parse_sensor_number,
)
from imap_processing.spice.geometry import (
    SpiceFrame,
    get_instrument_spin_phase,
    get_spacecraft_spin_phase,
    instrument_pointing,
)
from imap_processing.spice.time import j2000ns_to_j2000s
from imap_processing.utils import convert_raw_to_eu


class TriggerId(IntEnum):
    """IntEnum class for trigger id values."""

    A = 1
    B = 2
    C = 3


class CoincidenceBitmap(IntEnum):
    """IntEnum class for coincidence type bitmap values."""

    A = 2**3
    B = 2**2
    C1 = 2**1
    C2 = 2**0


logger = logging.getLogger(__name__)
ATTR_MGR = ImapCdfAttributes()
ATTR_MGR.add_instrument_global_attrs("hi")
ATTR_MGR.add_instrument_variable_attrs(instrument="hi", level=None)


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
    logical_source_parts = parse_filename_like(l1a_dataset.attrs["Logical_source"])
    # TODO: apid is not currently stored in all L1A data but should be.
    #    Use apid to determine what L1B processing function to call

    # Housekeeping processing
    if logical_source_parts["descriptor"].endswith("hk"):
        # if packet_enum in (HIAPID.H45_APP_NHK, HIAPID.H90_APP_NHK):
        packet_enum = HIAPID(l1a_dataset["pkt_apid"].data[0])
        conversion_table_path = str(
            imap_module_directory / "hi" / "l1b" / "hi_eng_unit_convert_table.csv"
        )
        l1b_dataset = convert_raw_to_eu(
            l1a_dataset,
            conversion_table_path=conversion_table_path,
            packet_name=packet_enum.name,
            comment="#",  # type: ignore[arg-type]
            # Todo error, Argument "comment" to "convert_raw_to_eu" has incompatible
            # type "str"; expected "dict[Any, Any]"
            converters={"mnemonic": str.lower},
        )

        l1b_dataset.attrs.update(ATTR_MGR.get_global_attributes("imap_hi_l1b_hk_attrs"))
    elif logical_source_parts["descriptor"].endswith("de"):
        l1b_dataset = annotate_direct_events(l1a_dataset)
    else:
        raise NotImplementedError(
            f"No Hi L1B processing defined for file type: "
            f"{l1a_dataset.attrs['Logical_source']}"
        )
    # Update global attributes
    l1b_dataset.attrs["Logical_source"] = l1b_dataset.attrs["Logical_source"].format(
        sensor=logical_source_parts["sensor"]
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
    l1b_dataset = l1a_dataset.copy()
    l1b_dataset.update(compute_coincidence_type_and_time_deltas(l1b_dataset))
    l1b_dataset.update(de_nominal_bin_and_spin_phase(l1b_dataset))
    l1b_dataset.update(compute_hae_coordinates(l1b_dataset))
    l1b_dataset.update(de_esa_energy_step(l1b_dataset))
    l1b_dataset.update(
        create_dataset_variables(
            ["quality_flag"],
            l1b_dataset["epoch"].size,
            att_manager_lookup_str="hi_de_{0}",
        )
    )
    l1b_dataset = l1b_dataset.drop_vars(
        ["tof_1", "tof_2", "tof_3", "de_tag", "ccsds_met", "meta_event_met"]
    )

    de_global_attrs = ATTR_MGR.get_global_attributes("imap_hi_l1b_de_attrs")
    l1b_dataset.attrs.update(**de_global_attrs)
    return l1b_dataset


def compute_coincidence_type_and_time_deltas(
    dataset: xr.Dataset,
) -> dict[str, xr.DataArray]:
    """
    Compute coincidence type and time deltas.

    Generates the new variables "coincidence_type", "delta_t_ab", "delta_t_ac1",
    "delta_t_bc1", and "delta_t_c1c2" and returns a dictionary with the new
    variables that can be added to the input dataset by calling the
    xarray.Dataset.update method.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L1A/B dataset that results from reading in a Hi L1A DE CDF.

    Returns
    -------
    new_vars : dict[str, xarray.DataArray]
        Keys are variable names and values are new populated `xarray.DataArray`.
    """
    new_vars = create_dataset_variables(
        [
            "coincidence_type",
            "delta_t_ab",
            "delta_t_ac1",
            "delta_t_bc1",
            "delta_t_c1c2",
        ],
        len(dataset.epoch),
        att_manager_lookup_str="hi_de_{0}",
    )
    out_ds = dataset.assign(new_vars)

    # compute masks needed for coincidence type and delta t calculations
    a_first = out_ds.trigger_id.values == TriggerId.A
    b_first = out_ds.trigger_id.values == TriggerId.B
    c_first = out_ds.trigger_id.values == TriggerId.C

    tof1_valid = np.isin(out_ds.tof_1.values, HiConstants.TOF1_BAD_VALUES, invert=True)
    tof2_valid = np.isin(out_ds.tof_2.values, HiConstants.TOF2_BAD_VALUES, invert=True)
    tof1and2_valid = tof1_valid & tof2_valid
    tof3_valid = np.isin(out_ds.tof_3.values, HiConstants.TOF3_BAD_VALUES, invert=True)

    # Table denoting how hit-first mask and valid TOF masks are used to set
    # coincidence type bitmask
    # -----------------------------------------------------------------------
    # | Trigger ID  |  Hit First  | TOF 1 Valid | TOF 2 Valid | TOF 3 Valid |
    # -----------------------------------------------------------------------
    # |      1      |      A      |     A,B     |     A,C1    |    C1,C2    |
    # |      2      |      B      |     A,B     |     B,C1    |    C1,C2    |
    # |      3      |      C1     |     A,C1    |     B,C1    |    C1,C2    |
    # Set coincidence type bitmask
    new_vars["coincidence_type"][a_first | tof1_valid] |= CoincidenceBitmap.A
    new_vars["coincidence_type"][
        b_first | (a_first & tof1_valid) | (c_first & tof2_valid)
    ] |= CoincidenceBitmap.B
    new_vars["coincidence_type"][c_first | tof2_valid] |= CoincidenceBitmap.C1
    new_vars["coincidence_type"][tof3_valid] |= CoincidenceBitmap.C2

    # Table denoting how TOF is interpreted for each Trigger ID
    # -----------------------------------------------------------------------
    # | Trigger ID  |  Hit First  |    TOF 1    |    TOF 2    |    TOF 3    |
    # -----------------------------------------------------------------------
    # |      1      |      A      |  t_b - t_a  | t_c1 - t_a  | t_c2 - t_c1 |
    # |      2      |      B      |  t_a - t_b  | t_c1 - t_b  | t_c2 - t_c1 |
    # |      3      |      C      |  t_a - t_c1 | t_b  - t_c1 | t_c2 - t_c1 |

    # Prepare for delta_t calculations by converting TOF values to nanoseconds
    tof_1_ns = (out_ds.tof_1.values * HiConstants.TOF1_TICK_DUR).astype(np.int32)
    tof_2_ns = (out_ds.tof_2.values * HiConstants.TOF2_TICK_DUR).astype(np.int32)
    tof_3_ns = (out_ds.tof_3.values * HiConstants.TOF3_TICK_DUR).astype(np.int32)

    # # ********** delta_t_ab = (t_b - t_a) **********
    # Table: row 1, column 1
    a_and_tof1 = a_first & tof1_valid
    new_vars["delta_t_ab"].values[a_and_tof1] = tof_1_ns[a_and_tof1]
    # Table: row 2, column 1
    b_and_tof1 = b_first & tof1_valid
    new_vars["delta_t_ab"].values[b_and_tof1] = -1 * tof_1_ns[b_and_tof1]
    # Table: row 3, column 1 and 2
    # delta_t_ab = (t_b - t_c1) - (t_a - t_c1) = (t_b - t_a)
    c_and_tof1and2 = c_first & tof1and2_valid
    new_vars["delta_t_ab"].values[c_and_tof1and2] = (
        tof_2_ns[c_and_tof1and2] - tof_1_ns[c_and_tof1and2]
    )

    # ********** delta_t_ac1 = (t_c1 - t_a) **********
    # Table: row 1, column 2
    a_and_tof2 = a_first & tof2_valid
    new_vars["delta_t_ac1"].values[a_and_tof2] = tof_2_ns[a_and_tof2]
    # Table: row 2, column 1 and 2
    # delta_t_ac1 = (t_c1 - t_b) - (t_a - t_b) = (t_c1 - t_a)
    b_and_tof1and2 = b_first & tof1and2_valid
    new_vars["delta_t_ac1"].values[b_and_tof1and2] = (
        tof_2_ns[b_and_tof1and2] - tof_1_ns[b_and_tof1and2]
    )
    # Table: row 3, column 1
    c_and_tof1 = c_first & tof1_valid
    new_vars["delta_t_ac1"].values[c_and_tof1] = -1 * tof_1_ns[c_and_tof1]

    # ********** delta_t_bc1 = (t_c1 - t_b) **********
    # Table: row 1, column 1 and 2
    # delta_t_bc1 = (t_c1 - t_a) - (t_b - t_a) => (t_c1 - t_b)
    a_and_tof1and2 = a_first & tof1and2_valid
    new_vars["delta_t_bc1"].values[a_and_tof1and2] = (
        tof_2_ns[a_and_tof1and2] - tof_1_ns[a_and_tof1and2]
    )
    # Table: row 2, column 2
    b_and_tof2 = b_first & tof2_valid
    new_vars["delta_t_bc1"].values[b_and_tof2] = tof_2_ns[b_and_tof2]
    # Table: row 3, column 2
    c_and_tof2 = c_first & tof2_valid
    new_vars["delta_t_bc1"].values[c_and_tof2] = -1 * tof_2_ns[c_and_tof2]

    # ********** delta_t_c1c2 = (t_c2 - t_c1) **********
    # Table: all rows, column 3
    new_vars["delta_t_c1c2"].values[tof3_valid] = tof_3_ns[tof3_valid]

    return new_vars


def de_nominal_bin_and_spin_phase(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """
    Compute nominal bin and instrument spin-phase for each direct event.

    Parameters
    ----------
    dataset : xarray.Dataset
        Direct event data to compute instrument spin-phase for.

    Returns
    -------
    new_vars : dict[str, xarray.DataArray]
        Dictionary containing new "spin_phase" variable.
    """
    new_vars = create_dataset_variables(
        [
            "spin_phase",
            "nominal_bin",
        ],
        len(dataset.epoch),
        att_manager_lookup_str="hi_de_{0}",
    )

    # nominal_bin is the index number of the 90 4-degree bins that each DE would
    # be binned into in the histogram packet. The Hi histogram data is binned by
    # spacecraft spin-phase, not instrument spin-phase, so the same is done here.
    met_query_times = j2000ns_to_j2000s(dataset.event_met.values)
    imap_spin_phase = get_spacecraft_spin_phase(met_query_times)
    new_vars["nominal_bin"].values = np.asarray(imap_spin_phase * 360 / 4).astype(
        np.uint8
    )

    sensor_number = parse_sensor_number(dataset.attrs["Logical_source"])
    new_vars["spin_phase"].values = np.asarray(
        get_instrument_spin_phase(
            met_query_times, SpiceFrame[f"IMAP_HI_{sensor_number}"]
        )
    ).astype(np.float32)
    return new_vars


def compute_hae_coordinates(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """
    Compute HAE latitude and longitude.

    The HAE coordinates are returned in a dictionary that can be added to the
    input dataset using the `.update()` method.

    Parameters
    ----------
    dataset : xarray.Dataset
        The partial L1B dataset that has had coincidence type, time deltas, and
        spin phase computed and added to the L1A data.

    Returns
    -------
    new_vars : dict[str, xarray.DataArray]
        Keys are variable names and values are `xarray.DataArray`.
    """
    new_vars = create_dataset_variables(
        [
            "hae_latitude",
            "hae_longitude",
        ],
        len(dataset.epoch),
        att_manager_lookup_str="hi_de_{0}",
    )
    et = j2000ns_to_j2000s(dataset.epoch.values)
    sensor_number = parse_sensor_number(dataset.attrs["Logical_source"])
    # TODO: For now, we are using SPICE to compute the look direction for each
    #   direct event. This will eventually be replaced by the algorithm Paul
    #   Janzen provided in the Hi Algorithm Document which should be faster
    pointing_coordinates = instrument_pointing(
        et, SpiceFrame[f"IMAP_HI_{sensor_number}"], SpiceFrame.ECLIPJ2000
    )
    new_vars["hae_latitude"].values = pointing_coordinates[:, 0]
    new_vars["hae_longitude"].values = pointing_coordinates[:, 1]

    return new_vars


def de_esa_energy_step(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """
    Compute esa_energy_step for each direct event.

    TODO: For now this function just returns the esa_step from the input dataset.
        Eventually, it will take L1B housekeeping data and determine the esa
        energy steps from that data.

    Parameters
    ----------
    dataset : xarray.Dataset
        The partial L1B dataset.

    Returns
    -------
    new_vars : dict[str, xarray.DataArray]
        Keys are variable names and values are `xarray.DataArray`.
    """
    new_vars = create_dataset_variables(
        ["esa_energy_step"],
        len(dataset.epoch),
        att_manager_lookup_str="hi_de_{0}",
    )
    # TODO: Implement this algorithm
    new_vars["esa_energy_step"].values = dataset.esa_step.values

    return new_vars
