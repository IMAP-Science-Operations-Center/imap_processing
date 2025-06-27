"""IMAP-HI L1B processing module."""

import logging
from enum import IntEnum
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import parse_filename_like
from imap_processing.hi.hi_l1a import HALF_CLOCK_TICK_S
from imap_processing.hi.utils import (
    HIAPID,
    CoincidenceBitmap,
    HiConstants,
    create_dataset_variables,
    parse_sensor_number,
)
from imap_processing.spice.geometry import (
    SpiceFrame,
    instrument_pointing,
)
from imap_processing.spice.spin import (
    get_instrument_spin_phase,
    get_spacecraft_spin_phase,
)
from imap_processing.spice.time import met_to_sclkticks, sct_to_et
from imap_processing.utils import packet_file_to_datasets


class TriggerId(IntEnum):
    """IntEnum class for trigger id values."""

    A = 1
    B = 2
    C = 3


logger = logging.getLogger(__name__)
ATTR_MGR = ImapCdfAttributes()
ATTR_MGR.add_instrument_global_attrs("hi")
ATTR_MGR.add_instrument_variable_attrs(instrument="hi", level=None)


def hi_l1b(dependency: Union[str, Path, xr.Dataset]) -> list[xr.Dataset]:
    """
    High level IMAP-HI L1B processing function.

    Parameters
    ----------
    dependency : str or xarray.Dataset
        Path to L0 file or L1A dataset to process.

    Returns
    -------
    l1b_dataset : list[xarray.Dataset]
        Processed xarray datasets.
    """
    # Housekeeping processing
    if isinstance(dependency, (Path, str)):
        logger.info(f"Running Hi L1B processing on file: {dependency}")
        l1b_datasets = housekeeping(dependency)
    elif isinstance(dependency, xr.Dataset):
        l1a_dataset = dependency
        logger.info(
            f"Running Hi L1B processing on dataset: "
            f"{l1a_dataset.attrs['Logical_source']}"
        )
        logical_source_parts = parse_filename_like(l1a_dataset.attrs["Logical_source"])
        # TODO: apid is not currently stored in all L1A data but should be.
        #    Use apid to determine what L1B processing function to call

        # DE processing
        if logical_source_parts["descriptor"].endswith("de"):
            l1b_datasets = [annotate_direct_events(l1a_dataset)]
            l1b_datasets[0].attrs["Logical_source"] = (
                l1b_datasets[0]
                .attrs["Logical_source"]
                .format(sensor=logical_source_parts["sensor"])
            )
        else:
            raise NotImplementedError(
                f"No Hi L1B processing defined for file type: "
                f"{l1a_dataset.attrs['Logical_source']}"
            )

    return l1b_datasets


def housekeeping(packet_file_path: Union[str, Path]) -> list[xr.Dataset]:
    """
    Will process IMAP raw data to l1b housekeeping dataset.

    In order to use `space_packet_parser` and the xtce which contains the
    DN to EU conversion factors, the L0 packet file is used to go straight to
    L1B.

    Parameters
    ----------
    packet_file_path : str
        Packet file path.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        Housekeeping datasets with engineering units.
    """
    packet_def_file = (
        imap_module_directory / "hi/packet_definitions/TLM_HI_COMBINED_SCI.xml"
    )
    # TODO: If raw and derived values can be gotten from one call to
    #    packet_file_to_datasets, the L1A and L1B could be generated
    #    in a single L1A/B function.
    datasets_by_apid = packet_file_to_datasets(
        packet_file=packet_file_path,
        xtce_packet_definition=packet_def_file,
        use_derived_value=True,
    )

    # Extract only the HK datasets
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    datasets = list()
    for apid in [HIAPID.H45_APP_NHK, HIAPID.H90_APP_NHK]:
        if apid in datasets_by_apid:
            datasets.append(datasets_by_apid[apid])
            # Update the dataset global attributes
            datasets[-1].attrs.update(
                ATTR_MGR.get_global_attributes("imap_hi_l1b_hk_attrs")
            )
            datasets[-1].attrs["Logical_source"] = (
                datasets[-1].attrs["Logical_source"].format(sensor=apid.sensor)
            )
    return datasets


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
    l1b_dataset.update(de_esa_energy_step(l1b_dataset))
    l1b_dataset.update(compute_coincidence_type_and_tofs(l1b_dataset))
    l1b_dataset.update(de_nominal_bin_and_spin_phase(l1b_dataset))
    l1b_dataset.update(compute_hae_coordinates(l1b_dataset))
    l1b_dataset.update(
        create_dataset_variables(
            ["quality_flag"],
            l1b_dataset["event_met"].size,
            att_manager_lookup_str="hi_de_{0}",
        )
    )
    l1b_dataset = l1b_dataset.drop_vars(
        [
            "src_seq_ctr",
            "pkt_len",
            "last_spin_num",
            "spin_invalids",
            "esa_step_seconds",
            "esa_step_milliseconds",
            "tof_1",
            "tof_2",
            "tof_3",
            "de_tag",
        ]
    )

    de_global_attrs = ATTR_MGR.get_global_attributes("imap_hi_l1b_de_attrs")
    l1b_dataset.attrs.update(**de_global_attrs)
    return l1b_dataset


def compute_coincidence_type_and_tofs(
    dataset: xr.Dataset,
) -> dict[str, xr.DataArray]:
    """
    Compute coincidence type and time of flights.

    Generates the new variables "coincidence_type", "tof_ab", "tof_ac1",
    "tof_bc1", and "tof_c1c2" and returns a dictionary with the new
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
            "tof_ab",
            "tof_ac1",
            "tof_bc1",
            "tof_c1c2",
        ],
        len(dataset.event_met),
        att_manager_lookup_str="hi_de_{0}",
    )

    # compute masks needed for coincidence type and ToF calculations
    a_first = dataset.trigger_id.values == TriggerId.A
    b_first = dataset.trigger_id.values == TriggerId.B
    c_first = dataset.trigger_id.values == TriggerId.C

    tof1_valid = np.isin(dataset.tof_1.values, HiConstants.TOF1_BAD_VALUES, invert=True)
    tof2_valid = np.isin(dataset.tof_2.values, HiConstants.TOF2_BAD_VALUES, invert=True)
    tof1and2_valid = tof1_valid & tof2_valid
    tof3_valid = np.isin(dataset.tof_3.values, HiConstants.TOF3_BAD_VALUES, invert=True)

    # Table denoting how hit-first mask and valid TOF masks are used to set
    # coincidence type bitmask
    # -----------------------------------------------------------------------
    # | Trigger ID  |  Hit First  | TOF 1 Valid | TOF 2 Valid | TOF 3 Valid |
    # -----------------------------------------------------------------------
    # |      1      |      A      |     A,B     |     A,C1    |    C1,C2    |
    # |      2      |      B      |     A,B     |     B,C1    |    C1,C2    |
    # |      3      |      C1     |     A,C1    |     B,C1    |    C1,C2    |
    # Set coincidence type bitmask
    new_vars["coincidence_type"][a_first | tof1_valid] |= np.uint8(CoincidenceBitmap.A)
    new_vars["coincidence_type"][
        b_first | (a_first & tof1_valid) | (c_first & tof2_valid)
    ] |= np.uint8(CoincidenceBitmap.B)
    new_vars["coincidence_type"][c_first | tof2_valid] |= np.uint8(CoincidenceBitmap.C1)
    new_vars["coincidence_type"][tof3_valid] |= np.uint8(CoincidenceBitmap.C2)

    # Table denoting how TOF is interpreted for each Trigger ID
    # -----------------------------------------------------------------------
    # | Trigger ID  |  Hit First  |    TOF 1    |    TOF 2    |    TOF 3    |
    # -----------------------------------------------------------------------
    # |      1      |      A      |  t_b - t_a  | t_c1 - t_a  | t_c2 - t_c1 |
    # |      2      |      B      |  t_a - t_b  | t_c1 - t_b  | t_c2 - t_c1 |
    # |      3      |      C      |  t_a - t_c1 | t_b  - t_c1 | t_c2 - t_c1 |

    # Prepare for L1B ToF calculations by converting L1A TOF values to nanoseconds
    tof_1_ns = (dataset.tof_1.values * HiConstants.TOF1_TICK_DUR).astype(np.int32)
    tof_2_ns = (dataset.tof_2.values * HiConstants.TOF2_TICK_DUR).astype(np.int32)
    tof_3_ns = (dataset.tof_3.values * HiConstants.TOF3_TICK_DUR).astype(np.int32)

    # # ********** tof_ab = (t_b - t_a) **********
    # Table: row 1, column 1
    a_and_tof1 = a_first & tof1_valid
    new_vars["tof_ab"].values[a_and_tof1] = tof_1_ns[a_and_tof1]
    # Table: row 2, column 1
    b_and_tof1 = b_first & tof1_valid
    new_vars["tof_ab"].values[b_and_tof1] = -1 * tof_1_ns[b_and_tof1]
    # Table: row 3, column 1 and 2
    # tof_ab = (t_b - t_c1) - (t_a - t_c1) = (t_b - t_a)
    c_and_tof1and2 = c_first & tof1and2_valid
    new_vars["tof_ab"].values[c_and_tof1and2] = (
        tof_2_ns[c_and_tof1and2] - tof_1_ns[c_and_tof1and2]
    )

    # ********** tof_ac1 = (t_c1 - t_a) **********
    # Table: row 1, column 2
    a_and_tof2 = a_first & tof2_valid
    new_vars["tof_ac1"].values[a_and_tof2] = tof_2_ns[a_and_tof2]
    # Table: row 2, column 1 and 2
    # tof_ac1 = (t_c1 - t_b) - (t_a - t_b) = (t_c1 - t_a)
    b_and_tof1and2 = b_first & tof1and2_valid
    new_vars["tof_ac1"].values[b_and_tof1and2] = (
        tof_2_ns[b_and_tof1and2] - tof_1_ns[b_and_tof1and2]
    )
    # Table: row 3, column 1
    c_and_tof1 = c_first & tof1_valid
    new_vars["tof_ac1"].values[c_and_tof1] = -1 * tof_1_ns[c_and_tof1]

    # ********** tof_bc1 = (t_c1 - t_b) **********
    # Table: row 1, column 1 and 2
    # tof_bc1 = (t_c1 - t_a) - (t_b - t_a) => (t_c1 - t_b)
    a_and_tof1and2 = a_first & tof1and2_valid
    new_vars["tof_bc1"].values[a_and_tof1and2] = (
        tof_2_ns[a_and_tof1and2] - tof_1_ns[a_and_tof1and2]
    )
    # Table: row 2, column 2
    b_and_tof2 = b_first & tof2_valid
    new_vars["tof_bc1"].values[b_and_tof2] = tof_2_ns[b_and_tof2]
    # Table: row 3, column 2
    c_and_tof2 = c_first & tof2_valid
    new_vars["tof_bc1"].values[c_and_tof2] = -1 * tof_2_ns[c_and_tof2]

    # ********** tof_c1c2 = (t_c2 - t_c1) **********
    # Table: all rows, column 3
    new_vars["tof_c1c2"].values[tof3_valid] = tof_3_ns[tof3_valid]

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
        len(dataset.event_met),
        att_manager_lookup_str="hi_de_{0}",
    )

    # nominal_bin is the index number of the 90 4-degree bins that each DE would
    # be binned into in the histogram packet. The Hi histogram data is binned by
    # spacecraft spin-phase, not instrument spin-phase, so the same is done here.
    # We have to add 1/2 clock tick to MET time before getting spin phase
    met_seconds = dataset.event_met.values + HALF_CLOCK_TICK_S
    imap_spin_phase = get_spacecraft_spin_phase(met_seconds)
    new_vars["nominal_bin"].values = np.asarray(imap_spin_phase * 360 / 4).astype(
        np.uint8
    )

    sensor_number = parse_sensor_number(dataset.attrs["Logical_source"])
    new_vars["spin_phase"].values = np.asarray(
        get_instrument_spin_phase(met_seconds, SpiceFrame[f"IMAP_HI_{sensor_number}"])
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
        The partial L1B dataset that has had coincidence type, times of flight,
        and spin phase computed and added to the L1A data.

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
        len(dataset.event_met),
        att_manager_lookup_str="hi_de_{0}",
    )
    # Per Section 2.2.5 of Algorithm Document, add 1/2 of tick duration
    # to MET before computing pointing.
    sclk_ticks = met_to_sclkticks(dataset.event_met.values + HALF_CLOCK_TICK_S)
    et = sct_to_et(sclk_ticks)
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
