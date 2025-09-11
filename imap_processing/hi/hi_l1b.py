"""IMAP-HI L1B processing module."""

import logging
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import parse_filename_like
from imap_processing.hi.hi_l1a import HALF_CLOCK_TICK_S
from imap_processing.hi.utils import (
    HIAPID,
    CoincidenceBitmap,
    EsaEnergyStepLookupTable,
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
from imap_processing.spice.time import met_to_sclkticks, met_to_utc, sct_to_et
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


def housekeeping(packet_file_path: str | Path) -> list[xr.Dataset]:
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
    logger.info(f"Running Hi L1B processing on file: {packet_file_path}")
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


def annotate_direct_events(
    l1a_de_dataset: xr.Dataset, l1b_hk_dataset: xr.Dataset, esa_energies_anc: Path
) -> list[xr.Dataset]:
    """
    Perform Hi L1B processing on direct event data.

    Parameters
    ----------
    l1a_de_dataset : xarray.Dataset
        L1A direct event data.
    l1b_hk_dataset : xarray.Dataset
        L1B housekeeping data coincident with the L1A DE data.
    esa_energies_anc : pathlib.Path
        Location of the esa-energies ancillary csv file.

    Returns
    -------
    l1b_datasets : list[xarray.Dataset]
        List containing exactly one L1B direct event dataset.
    """
    logger.info(
        f"Running Hi L1B processing on dataset: "
        f"{l1a_de_dataset.attrs['Logical_source']}"
    )

    l1b_de_dataset = l1a_de_dataset.copy()
    l1b_de_dataset.update(
        de_esa_energy_step(l1b_de_dataset, l1b_hk_dataset, esa_energies_anc)
    )
    l1b_de_dataset.update(compute_coincidence_type_and_tofs(l1b_de_dataset))
    l1b_de_dataset.update(de_nominal_bin_and_spin_phase(l1b_de_dataset))
    l1b_de_dataset.update(compute_hae_coordinates(l1b_de_dataset))
    l1b_de_dataset.update(
        create_dataset_variables(
            ["quality_flag"],
            l1b_de_dataset["event_met"].size,
            att_manager_lookup_str="hi_de_{0}",
        )
    )
    l1b_de_dataset = l1b_de_dataset.drop_vars(
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
    l1b_de_dataset.attrs.update(**de_global_attrs)

    logical_source_parts = parse_filename_like(l1a_de_dataset.attrs["Logical_source"])
    l1b_de_dataset.attrs["Logical_source"] = l1b_de_dataset.attrs[
        "Logical_source"
    ].format(sensor=logical_source_parts["sensor"])
    return [l1b_de_dataset]


def any_good_direct_events(dataset: xr.Dataset) -> bool:
    """
    Test dataset to see if there are any good direct events.

    Datasets can have no good direct events when there were no DEs in a pointing.
    In this case, due to restrictions with cdflib, we have to write a single
    bad DE in the CDF.

    Parameters
    ----------
    dataset : xarray.Dataset
        Run the check on this dataset.

    Returns
    -------
    any_good_events : bool
        True if there is at least one good direct event. False otherwise.
    """
    return bool(np.any(dataset["trigger_id"] != dataset["trigger_id"].attrs["FILLVAL"]))


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
    # Check for no valid direct events.
    if not any_good_direct_events(dataset):
        return new_vars

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
    # Check for no valid direct events.
    if not any_good_direct_events(dataset):
        return new_vars

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
    # Check for no valid direct events.
    if not any_good_direct_events(dataset):
        return new_vars

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


def de_esa_energy_step(
    l1b_de_ds: xr.Dataset, l1b_hk_ds: xr.Dataset, esa_energies_anc: Path
) -> dict[str, xr.DataArray]:
    """
    Compute esa_energy_step for each direct event.

    Parameters
    ----------
    l1b_de_ds : xarray.Dataset
        The partial L1B dataset.
    l1b_hk_ds : xarray.Dataset
        L1B housekeeping data coincident with the L1A DE data.
    esa_energies_anc : pathlib.Path
        Location of the esa-energies ancillary csv file.

    Returns
    -------
    new_vars : dict[str, xarray.DataArray]
        Keys are variable names and values are `xarray.DataArray`.
    """
    new_vars = create_dataset_variables(
        ["esa_energy_step"],
        len(l1b_de_ds.epoch),
        att_manager_lookup_str="hi_de_{0}",
    )
    # Check for no valid direct events.
    if not any_good_direct_events(l1b_de_ds):
        return new_vars

    # Get the LUT object using the HK data and esa-energies ancillary csv
    esa_energies_lut = pd.read_csv(esa_energies_anc, comment="#")
    esa_to_esa_energy_step_lut = get_esa_to_esa_energy_step_lut(
        l1b_hk_ds, esa_energies_lut
    )
    new_vars["esa_energy_step"].values = esa_to_esa_energy_step_lut.query(
        l1b_de_ds["ccsds_met"].data, l1b_de_ds["esa_step"].data
    )

    return new_vars


def get_esa_to_esa_energy_step_lut(
    l1b_hk_ds: xr.Dataset, esa_energies_lut: pd.DataFrame
) -> EsaEnergyStepLookupTable:
    """
    Generate a lookup table that associates an esa_step to an esa_energy_step.

    Parameters
    ----------
    l1b_hk_ds : xarray.Dataset
        L1B housekeeping dataset.
    esa_energies_lut : pandas.DataFrame
        Esa energies lookup table derived from ancillary file.

    Returns
    -------
    esa_energy_step_lut : EsaEnergyStepLookupTable
        A lookup table object that can be used to query by MET time and esa_step
        for the associated esa_energy_step values.

    Notes
    -----
    Algorithm definition in section 2.1.2 of IMAP Hi Algorithm Document.
    """
    # Instantiate a lookup table object
    esa_energy_step_lut = EsaEnergyStepLookupTable()
    # Get the set of esa_steps visited
    esa_steps = list(sorted(set(l1b_hk_ds["sci_esa_step"].data)))
    # Break into contiguous segments where op_mode == "HVSCI"
    # Pad the boolean array `op_mode == HVSCI` with False values on each end.
    # This treats starting or ending in HVSCI mode as a transition in the next
    # step where np.diff is used to find op_mode transitions into and out of
    # HVSCI
    padded_mask = np.pad(
        l1b_hk_ds["op_mode"].data == "HVSCI", (1, 1), constant_values=False
    )
    mode_changes = np.diff(padded_mask.astype(int))
    hsvsci_starts = np.nonzero(mode_changes == 1)[0]
    hsvsci_ends = np.nonzero(mode_changes == -1)[0]
    for i_start, i_end in zip(hsvsci_starts, hsvsci_ends, strict=False):
        contiguous_hvsci_ds = l1b_hk_ds.isel(dict(epoch=slice(i_start, i_end)))
        # Find median inner and outer ESA voltages for each ESA step
        for esa_step in esa_steps:
            single_esa_ds = contiguous_hvsci_ds.where(
                contiguous_hvsci_ds["sci_esa_step"] == esa_step, drop=True
            )
            if len(single_esa_ds["epoch"].data) == 0:
                logger.debug(
                    f"No instances of sci_esa_step == {esa_step} "
                    f"present in contiguous HVSCI block with interval: "
                    f"({met_to_utc(contiguous_hvsci_ds['shcoarse'].data[[0, -1]])})"
                )
                continue
            inner_esa_voltage = np.where(
                single_esa_ds["inner_esa_state"].data == "LO",
                single_esa_ds["inner_esa_lo"].data,
                single_esa_ds["inner_esa_hi"].data,
            )
            median_inner_esa = np.median(inner_esa_voltage)
            median_outer_esa = np.median(single_esa_ds["outer_esa"].data)
            # Match median voltages to ESA Energies LUT
            inner_voltage_match = (
                np.abs(median_inner_esa - esa_energies_lut["inner_esa_voltage"])
                <= esa_energies_lut["inner_esa_delta_v"]
            )
            outer_voltage_match = (
                np.abs(median_outer_esa - esa_energies_lut["outer_esa_voltage"])
                <= esa_energies_lut["outer_esa_delta_v"]
            )
            matching_esa_energy = esa_energies_lut[
                np.logical_and(inner_voltage_match, outer_voltage_match)
            ]
            if len(matching_esa_energy) != 1:
                if len(matching_esa_energy) == 0:
                    logger.critical(
                        f"No esa_energy_step matches found for esa_step "
                        f"{esa_step} during interval: "
                        f"({met_to_utc(single_esa_ds['shcoarse'].data[[0, -1]])}) "
                        f"with median esa voltages: "
                        f"{median_inner_esa}, {median_outer_esa}."
                    )
                if len(matching_esa_energy) > 1:
                    logger.critical(
                        f"Multiple esa_energy_step matches found for esa_step "
                        f"{esa_step} during interval: "
                        f"({met_to_utc(single_esa_ds['shcoarse'].data[[0, -1]])}) "
                        f"with median esa voltages: "
                        f"{median_inner_esa}, {median_outer_esa}."
                    )
                continue
            # Set LUT to matching esa_energy_step for time range
            esa_energy_step_lut.add_entry(
                contiguous_hvsci_ds["shcoarse"].data[0],
                contiguous_hvsci_ds["shcoarse"].data[-1],
                esa_step,
                matching_esa_energy["esa_energy_step"].values[0],
            )
    return esa_energy_step_lut
