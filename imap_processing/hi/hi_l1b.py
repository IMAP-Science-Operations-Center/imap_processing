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
from imap_processing.hi.hi_l1a import MILLISECOND_TO_S
from imap_processing.hi.utils import (
    HIAPID,
    CoincidenceBitmap,
    EsaEnergyStepLookupTable,
    GoodMetRangeLookupTable,
    HiConstants,
    create_dataset_variables,
    parse_sensor_number,
)
from imap_processing.quality_flags import ImapHiL1bDeFlags
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
    l1a_de_dataset: xr.Dataset,
    l1b_hk_dataset: xr.Dataset,
    esa_energies_anc: Path,
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
        List containing exactly one L1B direct event dataset. Its
        "gain_match_{field}" global attributes (see
        `CalibrationProductConfig.GAIN_MATCH_FIELDS`) record the pointing's
        reference detector voltage deltas (see `de_gain_test_filter`); these
        are NaN if they could not be determined.
    """
    logger.info(
        f"Running Hi L1B processing on dataset: "
        f"{l1a_de_dataset.attrs['Logical_source']}"
    )

    l1b_de_dataset = l1a_de_dataset.copy()
    # Creates the baseline "ccsds_qf" (PACKET_FULL/BADSPIN bits) that
    # de_esa_energy_step() and de_gain_test_filter() build on.
    l1b_de_dataset.update(de_ccsds_qf(l1b_de_dataset))
    l1b_de_dataset.update(
        de_esa_energy_step(l1b_de_dataset, l1b_hk_dataset, esa_energies_anc)
    )
    # esa_step_met (the MET when the ESA was stepped -- i.e. when data
    # collection for this packet's ESA step actually began) is needed by
    # de_gain_test_filter() below, since "ccsds_met" (packet creation time)
    # lags real data collection by tens to over a hundred seconds on real
    # flight data and can spill a packet across a good/bad segment boundary.
    l1b_de_dataset.update(de_esa_step_met(l1b_de_dataset))
    # Modifies "esa_energy_step" and "ccsds_qf" in place, and sets the
    # "gain_match_{field}" global attributes.
    l1b_de_dataset = de_gain_test_filter(l1b_de_dataset, l1b_hk_dataset)
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
    met_seconds = dataset.event_met.values + HiConstants.HALF_CLOCK_TICK_S
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
    sclk_ticks = met_to_sclkticks(
        dataset.event_met.values + HiConstants.HALF_CLOCK_TICK_S
    )
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
    l1b_de_ds: xr.Dataset,
    l1b_hk_ds: xr.Dataset,
    esa_energies_anc: Path,
) -> dict[str, xr.DataArray]:
    """
    Compute esa_energy_step for each direct event from ESA voltage measurements.

    Must be called after de_ccsds_qf(), which creates the "ccsds_qf" variable
    this function modifies in place.

    Parameters
    ----------
    l1b_de_ds : xarray.Dataset
        The partial L1B dataset. Must already contain "ccsds_qf" (see
        de_ccsds_qf()). Modified in place: ImapHiL1bDeFlags.BAD_ESA_VOLTAGE is
        set in "ccsds_qf" for packets whose measured ESA voltage didn't match
        any esa_energy_step.
    l1b_hk_ds : xarray.Dataset
        L1B housekeeping data coincident with the L1A DE data.
    esa_energies_anc : pathlib.Path
        Location of the esa-energies ancillary csv file.

    Returns
    -------
    new_vars : dict[str, xarray.DataArray]
        Dictionary with the new "esa_energy_step" DataArray.
        de_gain_test_filter() must be called after this function to force
        FILLVAL into "esa_energy_step" and set its own "ccsds_qf" bit for
        events whose detector voltages don't match the pointing's gain
        configuration.
    """
    new_vars = create_dataset_variables(
        ["esa_energy_step"],
        len(l1b_de_ds.epoch),
        att_manager_lookup_str="hi_de_{0}",
    )
    # Check for no valid direct events.
    if not any_good_direct_events(l1b_de_ds):
        return new_vars

    esa_energies_lut = pd.read_csv(esa_energies_anc, comment="#")
    esa_to_esa_energy_step_lut = get_esa_to_esa_energy_step_lut(
        l1b_hk_ds, esa_energies_lut
    )
    new_vars["esa_energy_step"].values = esa_to_esa_energy_step_lut.query(
        l1b_de_ds["ccsds_met"].data, l1b_de_ds["esa_step"].data
    )
    # Set the ccsds_qf quality flag bit for packets whose measured ESA voltage
    # didn't match any esa_energy_step.
    esa_energy_step_fillval = new_vars["esa_energy_step"].attrs["FILLVAL"]
    l1b_de_ds["ccsds_qf"].values[
        new_vars["esa_energy_step"].values == esa_energy_step_fillval
    ] |= np.uint8(ImapHiL1bDeFlags.BAD_ESA_VOLTAGE)

    return new_vars


def compute_reference_hv_values(hk_segment_ds: xr.Dataset) -> dict[str, float]:
    """
    Compute median detector high voltage values from a housekeeping segment.

    Used both to detect gain test intervals (comparing every HVSCI segment's
    values against the pointing's own first-segment reference, see
    de_gain_test_filter()) and to derive the pointing's reference values for
    geometric factor lookup (see
    CalibrationProductConfig.match_gain_config_id()).

    Parameters
    ----------
    hk_segment_ds : xarray.Dataset
        A slice of L1B housekeeping data (e.g. the first few packets of a
        contiguous HVSCI segment, or a full segment) containing every field
        in HiConstants.GAIN_TEST_HV_DELTA_V.

    Returns
    -------
    dict[str, float]
        Median value for each field in HiConstants.GAIN_TEST_HV_DELTA_V.
    """
    return {
        field: float(np.median(hk_segment_ds[field].data))
        for field in HiConstants.GAIN_TEST_HV_DELTA_V
    }


def compute_gain_match_values(raw_hv_values: dict[str, float]) -> dict[str, float]:
    """
    Derive the back/front voltage differences used for geometric factor lookup.

    Computed as back minus front (rather than front minus back) so that the
    resulting deltas are positive, consistent with real flight detector
    voltages (front voltages are more negative than back voltages -- see
    imap_processing/hi/gain_test_analysis.ipynb).

    Parameters
    ----------
    raw_hv_values : dict[str, float]
        Raw detector high voltage values keyed by field name, e.g. as
        returned by compute_reference_hv_values() (must contain "mcp_f",
        "mcp_b", "cem_f", "cem_bk_a", "cem_bk_b", and "tof").

    Returns
    -------
    dict[str, float]
        Dictionary with keys "mcp_delta_v", "cem_a_delta_v", "cem_b_delta_v",
        and "tof_v", matching CalibrationProductConfig.GAIN_MATCH_FIELDS, for
        use with CalibrationProductConfig.match_gain_config_id().
    """
    return {
        "mcp_delta_v": raw_hv_values["mcp_b"] - raw_hv_values["mcp_f"],
        "cem_a_delta_v": raw_hv_values["cem_bk_a"] - raw_hv_values["cem_f"],
        "cem_b_delta_v": raw_hv_values["cem_bk_b"] - raw_hv_values["cem_f"],
        "tof_v": raw_hv_values["tof"],
    }


def de_gain_test_filter(
    l1b_de_ds: xr.Dataset,
    l1b_hk_ds: xr.Dataset,
) -> xr.Dataset:
    """
    Exclude gain test intervals and force FILLVAL for non-matching events.

    Must be called after de_esa_energy_step() (which sets the "esa_energy_step"
    and "ccsds_qf" variables this function modifies in place) and
    de_esa_step_met() (which sets the "esa_step_met" variable this function
    reads).

    A pointing's own first HVSCI segment (its first ~3 housekeeping packets)
    defines that pointing's reference detector voltages. Every contiguous
    HVSCI segment in the pointing is then compared against this reference;
    a segment whose median voltages drift from the reference by more than
    HiConstants.GAIN_TEST_HV_DELTA_V (per field) is treated as a gain test
    and excluded. No ancillary file is involved -- the reference is derived
    entirely from the pointing's own data and the tolerances are hardcoded
    constants.

    Parameters
    ----------
    l1b_de_ds : xarray.Dataset
        The partial L1B dataset. Must already contain "esa_energy_step" and
        "ccsds_qf" (see de_esa_energy_step()) and "esa_step_met" (see
        de_esa_step_met()). Modified in place: FILLVAL is forced into
        "esa_energy_step" for events falling outside a matching
        HVSCI segment, ImapHiL1bDeFlags.BAD_DETECTOR_VOLTAGE is set in
        "ccsds_qf" for the same events, and new "gain_match_{field}" global
        attributes (one per CalibrationProductConfig.GAIN_MATCH_FIELDS) are
        set to the pointing's reference voltage deltas (NaN if they could
        not be determined). The geometric factor itself is not computed
        here -- downstream processing (L1C) looks up the geometric factor
        per esa_energy_step from the cal-prod ancillary file's matching
        gain_config_id, using these recorded "gain_match_{field}" attributes
        (see hi_l1c.pset_geometric_factor()).
    l1b_hk_ds : xarray.Dataset
        L1B housekeeping data coincident with the L1A DE data.

    Returns
    -------
    l1b_de_ds : xarray.Dataset
        The same dataset passed in, modified in place as described above.
    """
    nan_gain_match_attrs = {
        f"gain_match_{field}": value
        for field, value in compute_gain_match_values(
            {field: np.nan for field in HiConstants.GAIN_TEST_HV_DELTA_V}
        ).items()
    }

    # Check for no valid direct events.
    if not any_good_direct_events(l1b_de_ds):
        logger.critical(
            "No good direct events in dataset; skipping gain test filtering."
        )
        l1b_de_ds.attrs.update(nan_gain_match_attrs)
        return l1b_de_ds

    segments = _get_hvsci_segments(l1b_hk_ds)
    if not segments:
        logger.critical(
            "No HVSCI segments found; cannot determine reference voltages "
            "for gain test filtering. All direct events will be flagged as "
            "BAD_DETECTOR_VOLTAGE."
        )
        esa_energy_step = l1b_de_ds["esa_energy_step"]
        esa_energy_step.values[:] = esa_energy_step.attrs["FILLVAL"]
        l1b_de_ds["ccsds_qf"].values[:] |= np.uint8(
            ImapHiL1bDeFlags.BAD_DETECTOR_VOLTAGE
        )
        l1b_de_ds.attrs.update(nan_gain_match_attrs)
        return l1b_de_ds

    # Use the first ~3 housekeeping packets of the first HVSCI segment as
    # this pointing's reference voltages.
    first_start, first_end = segments[0]
    n_reference_packets = min(3, first_end - first_start)
    reference_ds = l1b_hk_ds.isel(
        epoch=slice(first_start, first_start + n_reference_packets)
    )
    reference_hv = compute_reference_hv_values(reference_ds)

    good_met_ranges = GoodMetRangeLookupTable()
    n_excluded = 0
    for i_start, i_end in segments:
        segment_ds = l1b_hk_ds.isel(epoch=slice(i_start, i_end))
        segment_hv = compute_reference_hv_values(segment_ds)
        matches_reference = all(
            abs(segment_hv[field] - reference_hv[field]) <= delta_v
            for field, delta_v in HiConstants.GAIN_TEST_HV_DELTA_V.items()
        )
        segment_start = segment_ds["shcoarse"].data[0]
        segment_end = segment_ds["shcoarse"].data[-1]
        if matches_reference:
            good_met_ranges.add_entry(segment_start, segment_end)
        else:
            n_excluded += 1
            interval = met_to_utc(np.array([segment_start, segment_end]))
            logger.info(
                f"HVSCI segment during interval ({interval}) does not match "
                f"the pointing's reference voltages and is likely a gain "
                f"test; excluding it. Direct events in this segment will be "
                f"flagged as BAD_DETECTOR_VOLTAGE."
            )
    logger.info(
        f"Gain test filtering: {len(segments) - n_excluded} of "
        f"{len(segments)} HVSCI segments matched the pointing's reference "
        f"voltages ({n_excluded} segment(s) excluded as likely gain tests)."
    )

    # Use esa_step_met (the MET when the ESA was stepped, i.e. the start of
    # this packet's 8-spin data collection) rather than ccsds_met (packet
    # creation time, logged by flight software only after the data was
    # collected) so that a packet is attributed to the segment its data was
    # actually collected in, not the segment active when the packet happened
    # to be created.
    esa_step_met = l1b_de_ds["esa_step_met"].data
    detector_voltage_bad_mask = ~good_met_ranges.query(esa_step_met)

    n_bad = int(np.sum(detector_voltage_bad_mask))
    if n_bad > 0:
        logger.info(
            f"Flagging {n_bad} of {detector_voltage_bad_mask.size} direct "
            f"events as BAD_DETECTOR_VOLTAGE (likely during a gain test); "
            f"their esa_energy_step is forced to FILLVAL."
        )

    esa_energy_step = l1b_de_ds["esa_energy_step"]
    esa_energy_step.values = np.where(
        detector_voltage_bad_mask,
        esa_energy_step.attrs["FILLVAL"],
        esa_energy_step.values,
    )
    l1b_de_ds["ccsds_qf"].values[detector_voltage_bad_mask] |= np.uint8(
        ImapHiL1bDeFlags.BAD_DETECTOR_VOLTAGE
    )

    gain_match_values = compute_gain_match_values(reference_hv)
    l1b_de_ds.attrs.update(
        {f"gain_match_{field}": value for field, value in gain_match_values.items()}
    )
    logger.info(f"Pointing reference gain match values set: {gain_match_values}.")
    return l1b_de_ds


def _get_hvsci_segments(l1b_hk_ds: xr.Dataset) -> list[tuple[int, int]]:
    """
    Find contiguous segments where op_mode == "HVSCI" in housekeeping data.

    Parameters
    ----------
    l1b_hk_ds : xarray.Dataset
        L1B housekeeping dataset.

    Returns
    -------
    segments : list[tuple[int, int]]
        List of (start_index, end_index) tuples. `end_index` is exclusive,
        suitable for use with `Dataset.isel(epoch=slice(start, end))`.
    """
    # Pad the boolean array `op_mode == HVSCI` with False values on each end.
    # This treats starting or ending in HVSCI mode as a transition in the next
    # step where np.diff is used to find op_mode transitions into and out of
    # HVSCI
    padded_mask = np.pad(
        l1b_hk_ds["op_mode"].data == "HVSCI", (1, 1), constant_values=False
    )
    mode_changes = np.diff(padded_mask.astype(int))
    starts = np.nonzero(mode_changes == 1)[0]
    ends = np.nonzero(mode_changes == -1)[0]
    return list(zip(starts, ends, strict=False))


def get_esa_to_esa_energy_step_lut(
    l1b_hk_ds: xr.Dataset,
    esa_energies_lut: pd.DataFrame,
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
        A lookup table object that can be used to query by MET time and
        esa_step for the associated esa_energy_step values. Segments/esa_steps
        where the measured ESA voltage did not match any esa_energy_step are
        left out of the LUT entirely, so querying them returns FILLVAL.

    Notes
    -----
    Algorithm definition in section 2.1.2 of IMAP Hi Algorithm Document.
    """
    # Instantiate a lookup table object
    esa_energy_step_lut = EsaEnergyStepLookupTable()
    # Get the set of esa_steps visited
    esa_steps = list(sorted(set(l1b_hk_ds["sci_esa_step"].data)))

    for i_start, i_end in _get_hvsci_segments(l1b_hk_ds):
        contiguous_hvsci_ds = l1b_hk_ds.isel(dict(epoch=slice(i_start, i_end)))
        segment_start = contiguous_hvsci_ds["shcoarse"].data[0]
        segment_end = contiguous_hvsci_ds["shcoarse"].data[-1]

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
                segment_start,
                segment_end,
                esa_step,
                matching_esa_energy["esa_energy_step"].values[0],
            )
    return esa_energy_step_lut


def de_esa_step_met(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """
    Compute esa_step_met for each CCSDS packet.

    The esa_step_met is the MET time when the ESA was stepped, computed from
    esa_step_seconds and esa_step_milliseconds.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L1A/B dataset containing esa_step_seconds and esa_step_milliseconds.

    Returns
    -------
    new_vars : dict[str, xarray.DataArray]
        Dictionary with "esa_step_met" key and float64 DataArray value.
    """
    new_vars = create_dataset_variables(
        ["esa_step_met"],
        len(dataset.epoch),
        att_manager_lookup_str="hi_de_{0}",
    )

    # Compute esa_step_met from esa_step_seconds and esa_step_milliseconds
    new_vars["esa_step_met"].values = (
        dataset["esa_step_seconds"].values.astype(np.float64)
        + dataset["esa_step_milliseconds"].values * MILLISECOND_TO_S
    )

    return new_vars


def de_ccsds_qf(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """
    Compute the baseline ccsds_qf quality flag for each CCSDS packet.

    Sets the PACKET_FULL and BADSPIN bits. Must be called first, before
    de_esa_energy_step() and de_gain_test_filter(), which add their own
    bits (BAD_ESA_VOLTAGE, BAD_DETECTOR_VOLTAGE) to this same "ccsds_qf"
    variable.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L1A/B dataset containing "ccsds_index" and "spin_invalids" for
        mapping events to packets.

    Returns
    -------
    new_vars : dict[str, xarray.DataArray]
        Dictionary with the new "ccsds_qf" DataArray.
    """
    max_events_per_packet = 664

    new_vars = create_dataset_variables(
        ["ccsds_qf"],
        len(dataset.epoch),
        att_manager_lookup_str="hi_de_{0}",
    )

    # Initialize all values to 0 (no flags set)
    new_vars["ccsds_qf"].values[:] = 0

    # Count events per CCSDS packet
    # ccsds_index maps each event to its originating packet
    ccsds_indices = dataset["ccsds_index"].values
    n_packets = len(dataset.epoch)

    # Filter out fill/out-of-range indices (e.g., uint16 FILLVAL 65535)
    valid_mask = (ccsds_indices >= 0) & (ccsds_indices < n_packets)

    # Set BADSPIN flag for packets with nonzero spin_invalids
    spin_invalid_mask = dataset["spin_invalids"].values != 0
    new_vars["ccsds_qf"].values[spin_invalid_mask] |= np.uint8(ImapHiL1bDeFlags.BADSPIN)

    # If there are no valid events, skip the PACKET_FULL check
    if not np.any(valid_mask):
        return new_vars

    # Compute event counts per valid CCSDS packet
    event_counts = np.bincount(
        ccsds_indices[valid_mask].astype(np.int64),
        minlength=n_packets,
    )
    # Set PACKET_FULL flag for packets with 664 events
    full_packet_mask = event_counts == max_events_per_packet
    new_vars["ccsds_qf"].values[full_packet_mask] |= np.uint8(
        ImapHiL1bDeFlags.PACKET_FULL
    )

    return new_vars
