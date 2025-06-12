"""IMAP-HI l1c processing module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from numpy import typing as npt
from numpy._typing import NDArray

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import parse_filename_like
from imap_processing.hi.hi_l1a import (
    DE_CLOCK_TICK_S,
    HALF_CLOCK_TICK_S,
)
from imap_processing.hi.utils import (
    CoincidenceBitmap,
    create_dataset_variables,
    full_dataarray,
    parse_sensor_number,
)
from imap_processing.spice.geometry import (
    SpiceFrame,
    frame_transform,
    frame_transform_az_el,
)
from imap_processing.spice.spin import (
    get_instrument_spin_phase,
    get_spin_data,
)
from imap_processing.spice.time import ttj2000ns_to_et

N_SPIN_BINS = 3600
SPIN_PHASE_BIN_EDGES = np.linspace(0, 1, N_SPIN_BINS + 1)
SPIN_PHASE_BIN_CENTERS = (SPIN_PHASE_BIN_EDGES[:-1] + SPIN_PHASE_BIN_EDGES[1:]) / 2

logger = logging.getLogger(__name__)


def hi_l1c(
    de_dataset: xr.Dataset, calibration_prod_config_path: Path
) -> list[xr.Dataset]:
    """
    High level IMAP-Hi l1c processing function.

    This function will be expanded once the l1c processing is better defined. It
    will need to add inputs such as Ephemerides, Goodtimes inputs, and
    instrument status summary and will output a Pointing Set CDF as well as a
    Goodtimes list (CDF?).

    Parameters
    ----------
    de_dataset : xarray.Dataset
        IMAP-Hi l1b de product.
    calibration_prod_config_path : pathlib.Path
        Calibration product configuration file.

    Returns
    -------
    l1c_dataset : xarray.Dataset
        Processed xarray dataset.
    """
    logger.info("Running Hi l1c processing")

    # TODO: I am not sure what the input for Goodtimes will be so for now,
    #    If the input is an xarray Dataset, do pset processing
    l1c_dataset = generate_pset_dataset(de_dataset, calibration_prod_config_path)

    return [l1c_dataset]


def generate_pset_dataset(
    de_dataset: xr.Dataset, calibration_prod_config_path: Path
) -> xr.Dataset:
    """
    Generate IMAP-Hi l1c pset xarray dataset from l1b product.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        IMAP-Hi l1b de product.
    calibration_prod_config_path : pathlib.Path
        Calibration product configuration file.

    Returns
    -------
    pset_dataset : xarray.Dataset
        Ready to be written to CDF.
    """
    logger.info(
        f"Generating IMAP-Hi l1c pset dataset for product "
        f"{de_dataset.attrs['Logical_file_id']}"
    )
    logical_source_parts = parse_filename_like(de_dataset.attrs["Logical_source"])
    # read calibration product configuration file
    config_df = CalibrationProductConfig.from_csv(calibration_prod_config_path)

    pset_dataset = empty_pset_dataset(
        de_dataset.epoch.data[0],
        de_dataset.esa_energy_step.data,
        config_df.cal_prod_config.number_of_products,
        logical_source_parts["sensor"],
    )
    pset_et = ttj2000ns_to_et(pset_dataset.epoch.data[0])
    # Calculate and add despun_z, hae_latitude, and hae_longitude variables to
    # the pset_dataset
    pset_dataset.update(pset_geometry(pset_et, logical_source_parts["sensor"]))
    # Bin the counts into the spin-bins
    pset_dataset.update(pset_counts(pset_dataset.coords, config_df, de_dataset))
    # Calculate and add the exposure time to the pset_dataset
    pset_dataset.update(pset_exposure(pset_dataset.coords, de_dataset))
    # Get the backgrounds
    pset_dataset.update(pset_backgrounds(pset_dataset.coords))

    return pset_dataset


def empty_pset_dataset(
    epoch_val: int, l1b_energy_steps: np.ndarray, n_cal_prods: int, sensor_str: str
) -> xr.Dataset:
    """
    Allocate an empty xarray.Dataset with appropriate pset coordinates.

    Parameters
    ----------
    epoch_val : int
        The starting epoch in J2000 TT nanoseconds for data in the PSET.
    l1b_energy_steps : np.ndarray
        The array of esa_energy_step data from the L1B DE product.
    n_cal_prods : int
        Number of calibration products to allocate.
    sensor_str : str
        '45sensor' or '90sensor'.

    Returns
    -------
    dataset : xarray.Dataset
        Empty xarray.Dataset ready to be filled with data.
    """
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)

    # preallocate coordinates xr.DataArrays
    coords = dict()
    # epoch coordinate has only 1 entry for pointing set
    epoch_attrs = attr_mgr.get_variable_attributes("epoch", check_schema=False)
    epoch_attrs.update(
        attr_mgr.get_variable_attributes("hi_pset_epoch", check_schema=False)
    )
    coords["epoch"] = xr.DataArray(
        np.array([epoch_val], dtype=np.int64),  # TODO: get dtype from cdf attrs?
        name="epoch",
        dims=["epoch"],
        attrs=epoch_attrs,
    )
    # Create the esa_energy_step coordinate
    attrs = attr_mgr.get_variable_attributes(
        "hi_pset_esa_energy_step", check_schema=False
    ).copy()
    dtype = attrs.pop("dtype")
    # Find the unique, non-zero esa_energy_steps from the L1B data
    esa_energy_steps = np.array(sorted(set(l1b_energy_steps) - {0}), dtype=dtype)
    coords["esa_energy_step"] = xr.DataArray(
        esa_energy_steps,
        name="esa_energy_step",
        dims=["esa_energy_step"],
        attrs=attrs,
    )

    attrs = attr_mgr.get_variable_attributes(
        "hi_pset_calibration_prod", check_schema=False
    ).copy()
    dtype = attrs.pop("dtype")
    coords["calibration_prod"] = xr.DataArray(
        np.arange(n_cal_prods, dtype=dtype),
        name="calibration_prod",
        dims=["calibration_prod"],
        attrs=attrs,
    )
    # spin angle bins are 0.1 degree bins for full 360 degree spin
    attrs = attr_mgr.get_variable_attributes(
        "hi_pset_spin_angle_bin", check_schema=False
    ).copy()
    dtype = attrs.pop("dtype")
    coords["spin_angle_bin"] = xr.DataArray(
        np.arange(N_SPIN_BINS, dtype=dtype),
        name="spin_angle_bin",
        dims=["spin_angle_bin"],
        attrs=attrs,
    )

    # Allocate the coordinate label variables
    data_vars = dict()
    # Generate label variables
    data_vars["esa_energy_step_label"] = xr.DataArray(
        coords["esa_energy_step"].values.astype(str),
        name="esa_energy_step_label",
        dims=["esa_energy_step"],
        attrs=attr_mgr.get_variable_attributes(
            "hi_pset_esa_energy_step_label", check_schema=False
        ),
    )
    data_vars["calibration_prod_label"] = xr.DataArray(
        coords["calibration_prod"].values.astype(str),
        name="calibration_prod_label",
        dims=["calibration_prod"],
        attrs=attr_mgr.get_variable_attributes(
            "hi_pset_calibration_prod_label", check_schema=False
        ),
    )
    data_vars["spin_bin_label"] = xr.DataArray(
        coords["spin_angle_bin"].values.astype(str),
        name="spin_bin_label",
        dims=["spin_angle_bin"],
        attrs=attr_mgr.get_variable_attributes(
            "hi_pset_spin_bin_label", check_schema=False
        ),
    )
    data_vars["label_vector_HAE"] = xr.DataArray(
        np.array(["x HAE", "y HAE", "z HAE"], dtype=str),
        name="label_vector_HAE",
        dims=[" "],
        attrs=attr_mgr.get_variable_attributes(
            "hi_pset_label_vector_HAE", check_schema=False
        ),
    )

    pset_global_attrs = attr_mgr.get_global_attributes("imap_hi_l1c_pset_attrs").copy()
    pset_global_attrs["Logical_source"] = pset_global_attrs["Logical_source"].format(
        sensor=sensor_str
    )
    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=pset_global_attrs)
    return dataset


def pset_geometry(pset_et: float, sensor_str: str) -> dict[str, xr.DataArray]:
    """
    Calculate PSET geometry variables.

    Parameters
    ----------
    pset_et : float
        Pointing set ephemeris time for which to calculate PSET geometry.
    sensor_str : str
        '45sensor' or '90sensor'.

    Returns
    -------
    geometry_vars : dict[str, xarray.DataArray]
        Keys are variable names and values are data arrays.
    """
    geometry_vars = create_dataset_variables(
        ["despun_z"], (1, 3), att_manager_lookup_str="hi_pset_{0}"
    )
    despun_z = frame_transform(
        pset_et,
        np.array([0, 0, 1]),
        SpiceFrame.IMAP_DPS,
        SpiceFrame.ECLIPJ2000,
    )
    geometry_vars["despun_z"].values = despun_z[np.newaxis, :].astype(np.float32)

    # Calculate hae_latitude and hae_longitude of the spin bins
    # define the azimuth/elevation coordinates in the Pointing Frame (DPS)
    # TODO: get the sensor's true elevation using SPICE?
    el = 0 if "90" in sensor_str else -45
    dps_az_el = np.array(
        [
            SPIN_PHASE_BIN_CENTERS * 360,
            np.full(N_SPIN_BINS, el),
        ]
    ).T
    hae_az_el = frame_transform_az_el(
        pset_et, dps_az_el, SpiceFrame.IMAP_DPS, SpiceFrame.ECLIPJ2000, degrees=True
    )

    geometry_vars.update(
        create_dataset_variables(
            ["hae_latitude", "hae_longitude"],
            (1, N_SPIN_BINS),
            att_manager_lookup_str="hi_pset_{0}",
        )
    )
    geometry_vars["hae_longitude"].values = hae_az_el[:, 0].astype(np.float32)[
        np.newaxis, :
    ]
    geometry_vars["hae_latitude"].values = hae_az_el[:, 1].astype(np.float32)[
        np.newaxis, :
    ]
    return geometry_vars


def pset_counts(
    pset_coords: dict[str, xr.DataArray],
    config_df: pd.DataFrame,
    l1b_de_dataset: xr.Dataset,
) -> dict[str, xr.DataArray]:
    """
    Bin direct events into PSET spin-bins.

    Parameters
    ----------
    pset_coords : dict[str, xarray.DataArray]
        The PSET coordinates from the xarray.Dataset.
    config_df : pandas.DataFrame
        The calibration product configuration dataframe.
    l1b_de_dataset : xarray.Dataset
        The L1B dataset for the pointing being processed.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dictionary containing new exposure_times DataArray to be added to the PSET
        dataset.
    """
    # Generate exposure time variable filled with zeros
    counts_var = create_dataset_variables(
        ["counts"],
        coords=pset_coords,
        att_manager_lookup_str="hi_pset_{0}",
        fill_value=0,
    )

    # Convert list of DEs to pandas dataframe for ease indexing/filtering
    de_df = l1b_de_dataset.drop_dims("epoch").to_pandas()

    # Remove DEs not in Goodtimes/angles
    good_mask = good_time_and_phase_mask(
        l1b_de_dataset.event_met.values, l1b_de_dataset.spin_phase.values
    )
    de_df = de_df[good_mask]

    # The calibration product configuration potentially has different coincidence
    # types for each ESA and different TOF windows for each calibration product,
    # esa energy step combination. Because of this we need to filter DEs that
    # belong to each combo individually.
    # Loop over the esa_energy_step values first
    for esa_energy, esa_df in config_df.groupby(level="esa_energy_step"):
        # Create a mask for all DEs at the current esa_energy_step.
        # esa_energy_step is recorded for each packet rather than for each DE,
        # so we use ccsds_index to get the esa_energy_step for each DE
        esa_mask = (
            l1b_de_dataset["esa_energy_step"].data[de_df["ccsds_index"].to_numpy()]
            == esa_energy
        )
        # Now loop over the calibration products for the current ESA energy
        for config_row in esa_df.itertuples():
            # Remove DEs that are not at the current ESA energy and in the list
            # of coincidence types for the current calibration product
            type_mask = de_df["coincidence_type"].isin(
                config_row.coincidence_type_values
            )
            filtered_de_df = de_df[(esa_mask & type_mask)]

            # Use the TOF window mask to remove DEs with TOFs outside the allowed range
            tof_fill_vals = {
                f"tof_{detector_pair}": l1b_de_dataset[f"tof_{detector_pair}"].attrs[
                    "FILLVAL"
                ]
                for detector_pair in CalibrationProductConfig.tof_detector_pairs
            }
            tof_in_window_mask = get_tof_window_mask(
                filtered_de_df, config_row, tof_fill_vals
            )
            filtered_de_df = filtered_de_df[tof_in_window_mask]

            # Bin remaining DEs into spin-bins
            i_esa = np.flatnonzero(pset_coords["esa_energy_step"].data == esa_energy)[0]
            # spin_phase is in the range [0, 1). Multiplying by N_SPIN_BINS and
            # truncating to an integer gives the correct bin index
            spin_bin_indices = (
                filtered_de_df["spin_phase"].to_numpy() * N_SPIN_BINS
            ).astype(int)
            # When iterating over rows of a dataframe, the names of the multi-index
            # are not preserved. Below, `config_row.Index[0]` gets the cal_prod_num
            # value from the namedtuple representing the dataframe row.
            np.add.at(
                counts_var["counts"].data[0, i_esa, config_row.Index[0]],
                spin_bin_indices,
                1,
            )
    return counts_var


def get_tof_window_mask(
    de_df: pd.DataFrame, prod_config_row: NamedTuple, fill_vals: dict
) -> NDArray[bool]:
    """
    Generate a mask indicating which DEs to keep based on TOF windows.

    Parameters
    ----------
    de_df : pandas.DataFrame
        The Direct Event dataframe for the DEs to filter based on the TOF
        windows.
    prod_config_row : NamedTuple
        A single row of the prod config dataframe represented as a named tuple.
    fill_vals : dict
        A dictionary containing the fill values used in the input DE TOF
        dataframe values. This value should be derived from the L1B DE CDF
        TOF variable attributes.

    Returns
    -------
    window_mask : np.ndarray
        A mask with one entry per DE in the input `de_df` indicating which DEs
        contain TOF values within the windows specified by `prod_config_row`.
        The mask is intended to directly filter the DE dataframe.
    """
    detector_pairs = CalibrationProductConfig.tof_detector_pairs
    tof_in_window_mask = np.empty((len(detector_pairs), len(de_df)), dtype=bool)
    for i_pair, detector_pair in enumerate(detector_pairs):
        low_limit = getattr(prod_config_row, f"tof_{detector_pair}_low")
        high_limit = getattr(prod_config_row, f"tof_{detector_pair}_high")
        tof_array = de_df[f"tof_{detector_pair}"].to_numpy()
        # The TOF in window mask contains True wherever the TOF is within
        # the configuration low/high bounds OR the FILLVAL is present. The
        # FILLVAL indicates that the detector pair was not hit. DEs with
        # the incorrect coincidence_type are already filtered out and this
        # implementation simplifies combining the tof_in_window_masks in
        # the next step.
        tof_in_window_mask[i_pair] = np.logical_or(
            np.logical_and(low_limit <= tof_array, tof_array <= high_limit),
            tof_array == fill_vals[f"tof_{detector_pair}"],
        )
    return np.all(tof_in_window_mask, axis=0)


def pset_backgrounds(pset_coords: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """
    Calculate pointing set backgrounds and background uncertainties.

    Parameters
    ----------
    pset_coords : dict[str, xarray.DataArray]
        The PSET coordinates from the xarray.Dataset.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dictionary containing background_rates and background_rates_unc DataArrays
        to be added to the PSET dataset.
    """
    # TODO: This is just a placeholder setting backgrounds to zero. The background
    #    algorithm will be determined in flight.
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)

    return {
        var_name: full_dataarray(
            var_name,
            attr_mgr.get_variable_attributes(f"hi_pset_{var_name}", check_schema=False),
            pset_coords,
            fill_value=fill_val,
        )
        for var_name, fill_val in [
            ("background_rates", 0),
            ("background_rates_uncertainty", 1),
        ]
    }


def pset_exposure(
    pset_coords: dict[str, xr.DataArray], l1b_de_dataset: xr.Dataset
) -> dict[str, xr.DataArray]:
    """
    Calculate PSET exposure time.

    Parameters
    ----------
    pset_coords : dict[str, xarray.DataArray]
        The PSET coordinates from the xarray.Dataset.
    l1b_de_dataset : xarray.Dataset
        The L1B dataset for the pointing being processed.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dictionary containing new exposure_times DataArray to be added to the PSET
        dataset.
    """
    # Extract the sensor number (45 or 90) for computing spin phase
    sensor_number = parse_sensor_number(l1b_de_dataset.attrs["Logical_source"])

    # Generate exposure time variable filled with zeros
    exposure_var = create_dataset_variables(
        ["exposure_times"],
        coords=pset_coords,
        att_manager_lookup_str="hi_pset_{0}",
        fill_value=0,
    )

    # Get a subset of the l1b_de_dataset that contains only the second
    # of each pair of packets at an ESA step.
    data_subset = find_second_de_packet_data(l1b_de_dataset)

    # Get the pandas dataframe with spin data
    spin_df = get_spin_data()

    # Loop over each of the CCSDS data rows that have been identified as the second
    # packet at an ESA step.
    # When implementing this, the memory needed to avoid this for loop was computed
    # and determined to be so large that the for loop is warranted.
    for _, packet_row in data_subset.groupby("epoch"):
        clock_tick_mets, clock_tick_weights = get_de_clock_ticks_for_esa_step(
            packet_row["ccsds_met"].values, spin_df
        )

        # Clock tick MET times are accumulation "edges". To get the mean spin-phase
        # for a given clock tick, add 1/2 clock tick and compute spin-phase.
        spin_phases = np.atleast_1d(
            get_instrument_spin_phase(
                clock_tick_mets + HALF_CLOCK_TICK_S,
                SpiceFrame[f"IMAP_HI_{sensor_number}"],
            )
        )

        # Remove ticks not in good times/angles
        good_mask = good_time_and_phase_mask(clock_tick_mets, spin_phases)
        spin_phases = spin_phases[good_mask]
        clock_tick_weights = clock_tick_weights[good_mask]

        # TODO: Account for flyback time. See alg doc section 2.3.5

        # Bin exposure times into spin-phase bins
        new_exposure_times, _ = np.histogram(
            spin_phases, bins=SPIN_PHASE_BIN_EDGES, weights=clock_tick_weights
        )
        # Accumulate the new exposure times for current esa_step
        i_esa = np.flatnonzero(
            pset_coords["esa_energy_step"].values
            == packet_row["esa_energy_step"].values
        )[0]
        exposure_var["exposure_times"].values[:, i_esa] += new_exposure_times

    # Convert exposure clock ticks to seconds
    exposure_var["exposure_times"].values *= DE_CLOCK_TICK_S

    return exposure_var


def find_second_de_packet_data(l1b_dataset: xr.Dataset) -> xr.Dataset:
    """
    Find the telemetry entries for the second packet at an ESA step.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The L1B Direct Event Dataset for the current pointing.

    Returns
    -------
    reduced_dataset : xarray.Dataset
        A dataset containing only the entries for the second packet at an ESA step.
    """
    epoch_dataset = l1b_dataset.drop_dims("event_met")
    # We should get two CCSDS packets per 8-spin ESA step.
    # Get the indices of the packet before each ESA change.
    esa_step = epoch_dataset["esa_step"].values
    second_esa_packet_idx = np.append(
        np.flatnonzero(np.diff(esa_step) != 0), len(esa_step) - 1
    )
    # Remove esa steps at 0 - these are calibrations
    second_esa_packet_idx = second_esa_packet_idx[esa_step[second_esa_packet_idx] != 0]
    # Remove indices where we don't have two consecutive packets at the same ESA
    if second_esa_packet_idx[0] == 0:
        logger.warning(
            f"Removing packet 0 with ESA step: {esa_step[0]} from"
            f"calculation of exposure time due to missing matched pair."
        )
        second_esa_packet_idx = second_esa_packet_idx[1:]
    missing_esa_pair_mask = (
        esa_step[second_esa_packet_idx - 1] != esa_step[second_esa_packet_idx]
    )
    if missing_esa_pair_mask.any():
        logger.warning(
            f"Removing {missing_esa_pair_mask.sum()} packets from exposure "
            f"time calculation due to missing ESA step DE packet pairs."
        )
    second_esa_packet_idx = second_esa_packet_idx[~missing_esa_pair_mask]
    # Reduce the dataset to just the second packet entries
    data_subset = epoch_dataset.isel(epoch=second_esa_packet_idx)
    return data_subset


def get_de_clock_ticks_for_esa_step(
    ccsds_met: float, spin_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an array of clock tick MET times for an 8-spin ESA step.

    Find the closest spin start time in the input spin dataframe to the packet
    creation time (`ccsds_met`) and generate an array of clock tick MET times
    for the period covered by the previous 8-spin group and an array of weights
    that represent the fraction of each clock tick that occurred in the 8-spin
    group.

    Parameters
    ----------
    ccsds_met : float
        The CCSDS MET of the second packet in a DE packet pair.
    spin_df : pandas.DataFrame
        Universal spin table dataframe.

    Returns
    -------
    clock_tick_mets : np.ndarray
        Array of MET times that a clock tick occurred in an 8-spin group of spins
        during which the ESA step was constant.
    clock_tick_weights : np.ndarray
        Array of weights to use when binning the clock tick MET times into spin-bins.
    """
    # Find the last spin_table entry with the start less than the CCSDS MET.
    # The CCSDS packet gets created just AFTER the final spin in the 8-spin
    # ESA step group so this match is the end time. The start time is
    # 8-spins earlier.
    spin_start_mets = spin_df.spin_start_met.to_numpy()
    # CCSDS MET has one second resolution, add one to it to make sure it is
    # greater than the spin start time it ended on.
    end_time_ind = np.flatnonzero(ccsds_met + 1 >= spin_start_mets).max()

    # If the minimum absolute difference is greater than 1/2 the spin-phase
    # we have a problem.
    if (
        ccsds_met - spin_start_mets[end_time_ind]
        > spin_df.iloc[end_time_ind].spin_period_sec / 2
    ):
        raise ValueError(
            "The difference between ccsds_met and spin_start_met, "
            f"{ccsds_met - spin_start_mets[end_time_ind]} seconds, "
            f"is too large. Check the spin table loaded for this pointing."
        )
    # If the end time index less than 8, we don't have enough spins in the
    # spin table to get a start time, so raise an error.
    if end_time_ind < 8:
        raise ValueError(
            "Error determining start/end time for exposure time. "
            f"The CCSDS MET time {ccsds_met} "
            "is less than 8 spins from the loaded spin table data."
        )
    clock_tick_mets = np.arange(
        spin_start_mets[end_time_ind - 8],
        spin_start_mets[end_time_ind],
        DE_CLOCK_TICK_S,
        dtype=float,
    )
    # The final clock-tick bin has less exposure time because the next spin
    # will trigger FSW to change ESA steps part way through that time. To
    # account for this in exposure time calculation, assign an array of
    # weights to use when binnig the clock-ticks to spin-bins. Weights are
    # fractional clock ticks. All weights are 1 except for the last one in
    # the array.
    clock_tick_weights = np.ones_like(clock_tick_mets, dtype=float)
    clock_tick_weights[-1] = (
        spin_start_mets[end_time_ind] - clock_tick_mets[-1]
    ) / DE_CLOCK_TICK_S
    return clock_tick_mets, clock_tick_weights


def good_time_and_phase_mask(
    tick_mets: np.ndarray, spin_phases: np.ndarray
) -> npt.NDArray:
    """
    Filter out the clock tick times that are not in good times and angles.

    Parameters
    ----------
    tick_mets : np.ndarray
        Clock-tick MET times.
    spin_phases : np.ndarray
        Spin phases for each clock tick.

    Returns
    -------
    keep_mask : np.ndarray
        Boolean mask indicating which clock ticks are in good times/phases.
    """
    # TODO: Implement this once we have Goodtimes data product defined.
    return np.full_like(tick_mets, True, dtype=bool)


@pd.api.extensions.register_dataframe_accessor("cal_prod_config")
class CalibrationProductConfig:
    """
    Register custom accessor for calibration product configuration DataFrames.

    Parameters
    ----------
    pandas_obj : pandas.DataFrame
        Object to run validation and use accessor functions on.
    """

    index_columns = (
        "cal_prod_num",
        "esa_energy_step",
    )
    tof_detector_pairs = ("ab", "ac1", "bc1", "c1c2")
    required_columns = (
        "coincidence_type_list",
        *[
            f"tof_{det_pair}_{limit}"
            for det_pair in tof_detector_pairs
            for limit in ["low", "high"]
        ],
    )

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._add_coincidence_values_column()

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Validate the current configuration.

        Parameters
        ----------
        df : pandas.DataFrame
            Object to validate.

        Raises
        ------
        AttributeError : If the dataframe does not pass validation.
        """
        for index_name in self.index_columns:
            if index_name in df.index:
                raise AttributeError(
                    f"Required index {index_name} not present in dataframe."
                )
        # Verify that the Dataframe has all the required columns
        for col in self.required_columns:
            if col not in df.columns:
                raise AttributeError(f"Required column {col} not present in dataframe.")
        # TODO: Verify that the same ESA energy steps exist in all unique calibration
        #   product numbers

    def _add_coincidence_values_column(self) -> None:
        """Generate and add the coincidence_type_values column to the dataframe."""
        # Add a column that consists of the coincidence type strings converted
        # to integer values
        self._obj["coincidence_type_values"] = self._obj.apply(
            lambda row: tuple(
                CoincidenceBitmap.detector_hit_str_to_int(entry)
                for entry in row["coincidence_type_list"]
            ),
            axis=1,
        )

    @classmethod
    def from_csv(cls, path: Path) -> pd.DataFrame:
        """
        Read configuration CSV file into a pandas.DataFrame.

        Parameters
        ----------
        path : pathlib.Path
            Location of the Calibration Product configuration CSV file.

        Returns
        -------
        dataframe : pandas.DataFrame
            Validated calibration product configuration data frame.
        """
        df = pd.read_csv(
            path,
            index_col=cls.index_columns,
            converters={"coincidence_type_list": lambda s: tuple(s.split("|"))},
            comment="#",
        )
        # Force the _init_ method to run by using the namespace
        _ = df.cal_prod_config.number_of_products
        return df

    @property
    def number_of_products(self) -> int:
        """
        Get the number of calibration products in the current configuration.

        Returns
        -------
        number_of_products : int
            The maximum number of calibration products defined in the list of
            calibration product definitions.
        """
        return len(self._obj.index.unique(level="cal_prod_num"))
