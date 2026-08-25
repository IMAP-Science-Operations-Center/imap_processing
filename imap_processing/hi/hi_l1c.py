"""IMAP-HI l1c processing module."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy import typing as npt

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import parse_filename_like
from imap_processing.hi.utils import (
    BackgroundConfig,
    CalibrationProductConfig,
    HiConstants,
    create_dataset_variables,
    full_dataarray,
    iter_background_events_by_config,
    iter_qualified_events_by_config,
    parse_sensor_number,
)
from imap_processing.spice.geometry import (
    SpiceFrame,
    frame_transform,
    frame_transform_az_el,
    get_spacecraft_to_instrument_spin_phase_offset,
)
from imap_processing.spice.repoint import get_pointing_times
from imap_processing.spice.spin import (
    get_spacecraft_spin_phase,
    get_spin_data,
)
from imap_processing.spice.time import met_to_ttj2000ns, ttj2000ns_to_et

N_SPIN_BINS = 3600
SPIN_PHASE_BIN_EDGES = np.linspace(0, 1, N_SPIN_BINS + 1)
SPIN_PHASE_BIN_CENTERS = (SPIN_PHASE_BIN_EDGES[:-1] + SPIN_PHASE_BIN_EDGES[1:]) / 2

logger = logging.getLogger(__name__)


def hi_l1c(
    de_dataset: xr.Dataset,
    calibration_prod_config_path: Path,
    goodtimes_ds: xr.Dataset,
    background_config_path: Path,
) -> list[xr.Dataset]:
    """
    High level IMAP-Hi l1c processing function.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        IMAP-Hi l1b de product.
    calibration_prod_config_path : pathlib.Path
        Calibration product configuration file.
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset with cull_flags.
    background_config_path : pathlib.Path
        Background configuration file.

    Returns
    -------
    l1c_dataset : xarray.Dataset
        Processed xarray dataset.
    """
    logger.info("Running Hi l1c processing")

    l1c_dataset = generate_pset_dataset(
        de_dataset,
        calibration_prod_config_path,
        goodtimes_ds,
        background_config_path,
    )

    return [l1c_dataset]


def generate_pset_dataset(
    de_dataset: xr.Dataset,
    calibration_prod_config_path: Path,
    goodtimes_ds: xr.Dataset,
    background_config_path: Path,
) -> xr.Dataset:
    """
    Generate IMAP-Hi l1c pset xarray dataset from l1b product.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        IMAP-Hi l1b de product.
    calibration_prod_config_path : pathlib.Path
        Calibration product configuration file.
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset with cull_flags.
    background_config_path : pathlib.Path
        Background configuration file.

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
    # Select this pointing's matched gain state up front
    hv_deltas = {
        field: de_dataset.attrs[field]
        for field in CalibrationProductConfig.GAIN_MATCH_FIELDS
    }
    gain_config_df = config_df.cal_prod_config.select_gain_config(hv_deltas)
    # read background configuration file
    background_df = BackgroundConfig.from_csv(background_config_path)

    pset_dataset = empty_pset_dataset(
        de_dataset.ccsds_met.data.mean(),
        de_dataset.esa_energy_step,
        config_df.cal_prod_config.calibration_product_numbers,
        logical_source_parts["sensor"],
    )
    # Calculate and add despun_z, hae_latitude, and hae_longitude variables to
    # the pset_dataset
    pset_midpoint_et = ttj2000ns_to_et(
        pset_dataset.epoch.data[0] + pset_dataset.epoch_delta.data[0] / 2
    )
    pset_dataset.update(pset_geometry(pset_midpoint_et, logical_source_parts["sensor"]))
    # Look up the per-esa_energy_step geometric factor for this pointing's
    # gain state.
    pset_dataset = add_pset_geometric_factor(pset_dataset, gain_config_df)
    # Bin the counts into the spin-bins
    pset_dataset.update(
        pset_counts(pset_dataset.coords, gain_config_df, de_dataset, goodtimes_ds)
    )
    # Calculate and add the exposure time to the pset_dataset
    pset_dataset.update(pset_exposure(pset_dataset.coords, de_dataset, goodtimes_ds))

    # Compute backgrounds (background counts computed internally)
    pset_dataset.update(
        pset_backgrounds(
            pset_dataset.coords,
            background_df,
            de_dataset,
            goodtimes_ds,
            pset_dataset["exposure_times"],
        )
    )

    return pset_dataset


def empty_pset_dataset(
    l1b_met: float,
    l1b_energy_steps: xr.DataArray,
    cal_prod_numbers: npt.NDArray[np.int_],
    sensor_str: str,
) -> xr.Dataset:
    """
    Allocate an empty xarray.Dataset with appropriate pset coordinates.

    Parameters
    ----------
    l1b_met : float
        Any met from the input L1B DE dataset. This is used to query the
        repoint-table data to get the start and end times of the pointing.
    l1b_energy_steps : xarray.DataArray
        The array of esa_energy_step data from the L1B DE product.
    cal_prod_numbers : numpy.ndarray
        Array of calibration product numbers from the configuration file.
        These can be arbitrary integers, not necessarily starting at 0.
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

    # Get the Pointing start and end times
    pointing_mets = get_pointing_times(l1b_met)
    epochs = met_to_ttj2000ns(np.asarray(pointing_mets))

    # epoch coordinate has only 1 entry for pointing set
    epoch_attrs = attr_mgr.get_variable_attributes("epoch", check_schema=False)
    epoch_attrs.update(
        attr_mgr.get_variable_attributes("hi_pset_epoch", check_schema=False)
    )
    coords["epoch"] = xr.DataArray(
        np.array([epochs[0]], dtype=np.int64),
        name="epoch",
        dims=["epoch"],
        attrs=epoch_attrs,
    )
    # Create the esa_energy_step coordinate
    attrs = attr_mgr.get_variable_attributes(
        "hi_pset_esa_energy_step", check_schema=False
    ).copy()
    dtype = attrs.pop("dtype")
    # Find the unique esa_energy_steps from the L1B data
    # Exclude 0 and FILLVAL
    esa_energy_steps = np.array(
        sorted(set(l1b_energy_steps.values) - {0, l1b_energy_steps.attrs["FILLVAL"]}),
        dtype=dtype,
    )
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
        cal_prod_numbers.astype(dtype),
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
    # Generate the epoch_delta variable
    data_vars["epoch_delta"] = xr.DataArray(
        np.diff(epochs),
        name="epoch_delta",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes(
            "hi_pset_epoch_delta", check_schema=False
        ),
    )
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
        Keys are variable names, and values are data arrays.
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


def add_pset_geometric_factor(
    pset_ds: xr.Dataset,
    gain_config_df: pd.DataFrame | None,
) -> xr.Dataset:
    """
    Add the geometric_factor variable to a pset dataset in place.

    Parameters
    ----------
    pset_ds : xarray.Dataset
        The PSET dataset being built. Must have "esa_energy_step" and
        "calibration_prod" coordinates.
    gain_config_df : pandas.DataFrame or None
        This pointing's matched gain state configuration (see
        CalibrationProductConfig.select_gain_config()), indexed by
        (calibration_prod, esa_energy_step), or None if the pointing's HV
        deltas didn't match exactly one gain_config_id.

    Returns
    -------
    xarray.Dataset
        The input pset_ds, updated in place with a "geometric_factor"
        variable, dims (epoch, esa_energy_step, calibration_prod).

    Notes
    -----
    A pointing's gain state is constant for the whole pointing (see
    `hi_l1b.de_gain_test_filter`), so the L1B DE product only records the
    pointing's reference detector voltage deltas as global attributes rather
    than duplicating the geometric factor across every direct event. Records
    the geometric_factor value for each (esa_energy_step, calibration_prod)
    pair directly from gain_config_df's rows. Not yet consumed by L2 processing
    (deferred to a follow-on ticket that handles combining PSETs from different
    gain states into a single map).
    """
    geometric_factor_var = create_dataset_variables(
        ["geometric_factor"],
        coords=pset_ds.coords,
        att_manager_lookup_str="hi_pset_{0}",
    )
    if gain_config_df is not None:
        # gain_config_df is indexed by (calibration_prod, esa_energy_step).
        # Convert to xarray and reindex onto the pset's own coordinate
        # values so it broadcasts directly into the output array (which
        # only has dims, not coordinate labels, to reindex_like).
        gain_factor_da = gain_config_df["geometric_factor"].to_xarray()
        gain_factor_da = gain_factor_da.reindex(
            esa_energy_step=pset_ds["esa_energy_step"].data,
            calibration_prod=pset_ds["calibration_prod"].data,
        )
        geometric_factor_var["geometric_factor"].values[0] = gain_factor_da.transpose(
            "esa_energy_step", "calibration_prod"
        ).values
    pset_ds.update(geometric_factor_var)
    return pset_ds


def pset_counts(
    pset_coords: dict[str, xr.DataArray],
    gain_config_df: pd.DataFrame | None,
    l1b_de_dataset: xr.Dataset,
    goodtimes_ds: xr.Dataset,
) -> dict[str, xr.DataArray]:
    """
    Bin direct events into PSET spin-bins.

    Parameters
    ----------
    pset_coords : dict[str, xarray.DataArray]
        The PSET coordinates from the xarray.Dataset.
    gain_config_df : pandas.DataFrame or None
        This pointing's matched gain state configuration indexed by
        (calibration_prod, esa_energy_step), or None if the pointing's HV
        deltas didn't match exactly one gain_config_id.
    l1b_de_dataset : xarray.Dataset
        The L1B dataset for the pointing being processed.
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset with cull_flags.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dictionary containing counts DataArray. All zero if gain_config_df
        is None.
    """
    # Generate counts variable filled with zeros
    counts_var = create_dataset_variables(
        ["counts"],
        coords=pset_coords,
        att_manager_lookup_str="hi_pset_{0}",
        fill_value=0,
    )
    if gain_config_df is None:
        return counts_var

    # Create mapping from calibration product numbers to array indices
    cal_prod_to_index = {
        cal_prod: idx
        for idx, cal_prod in enumerate(pset_coords["calibration_prod"].values)
    }

    # Drop events with FILLVAL for trigger_id. This should only occur for a
    # pointing with no events that gets a single fill event
    de_ds = l1b_de_dataset.drop_dims("epoch")

    # Remove DEs with invalid trigger_id. This should only occur for a
    # pointing with no events that gets a single fill event
    good_mask = de_ds["trigger_id"].data != de_ds["trigger_id"].attrs["FILLVAL"]
    if not np.any(good_mask):
        return counts_var

    # Remove DEs not in Goodtimes/angles
    # For direct events, use nominal_bin (spacecraft spin bin 0-89) to look up goodtimes
    goodtimes_mask = good_time_and_phase_mask(
        de_ds.event_met.values,
        de_ds.nominal_bin.values,
        goodtimes_ds,
    )
    de_ds = de_ds.isel(event_met=goodtimes_mask)

    # Get esa_energy_step for each event (recorded per packet, use ccsds_index)
    esa_energy_steps = l1b_de_dataset["esa_energy_step"].data[de_ds["ccsds_index"].data]

    # The calibration product configuration potentially has different coincidence
    # types for each ESA and different TOF windows for each calibration product,
    # esa energy step combination. Use the shared generator to iterate over all
    # config combinations and get qualified event masks.
    for esa_energy, config_row, qualified_mask in iter_qualified_events_by_config(
        de_ds, gain_config_df, esa_energy_steps
    ):
        # Filter events using the qualified mask
        filtered_de_ds = de_ds.isel(event_met=qualified_mask)

        # Bin remaining DEs into spin-bins
        i_esa = np.flatnonzero(pset_coords["esa_energy_step"].data == esa_energy)[0]
        # spin_phase is in the range [0, 1). Multiplying by N_SPIN_BINS and
        # truncating to an integer gives the correct bin index
        spin_bin_indices = (filtered_de_ds["spin_phase"].data * N_SPIN_BINS).astype(int)
        # When iterating over rows of a dataframe, the names of the multi-index
        # are not preserved. Below, `config_row.Index[0]` gets the
        # calibration_prod value (index level 0 of gain_config_df's
        # (calibration_prod, esa_energy_step) MultiIndex, already sliced to
        # this pointing's single gain_config_id above) from the namedtuple
        # representing the dataframe row. We map this to the array index
        # using cal_prod_to_index.
        i_cal_prod = cal_prod_to_index[config_row.Index[0]]
        np.add.at(
            counts_var["counts"].data[0, i_esa, i_cal_prod],
            spin_bin_indices,
            1,
        )

    return counts_var


def _compute_background_counts(
    pset_coords: dict[str, xr.DataArray],
    background_config_df: pd.DataFrame,
    l1b_de_dataset: xr.Dataset,
    goodtimes_ds: xr.Dataset,
) -> xr.DataArray:
    """
    Compute background counts by filtering and binning direct events.

    Background counts are computed across all esa_energy_steps and spin_angle_bins
    since backgrounds are isotropic and do not depend on ESA energy step or spin angle.

    Parameters
    ----------
    pset_coords : dict[str, xarray.DataArray]
        The PSET coordinates from the xarray.Dataset.
    background_config_df : pandas.DataFrame
        Background configuration DataFrame with MultiIndex
        (calibration_prod, background_index, esa_energy_step).
    l1b_de_dataset : xarray.Dataset
        The L1B dataset for the pointing being processed.
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset with cull_flags.

    Returns
    -------
    xarray.DataArray
        Background counts with dims (epoch, calibration_prod, background_index).
    """
    # Create background_counts as xarray DataArray with proper coordinates
    # Note: esa_energy_step and spin_angle_bin are NOT included since backgrounds
    # are isotropic and computed across all ESA steps and spin angles
    background_indices = (
        background_config_df.index.get_level_values("background_index")
        .unique()
        .sort_values()
        .values
    )

    bg_coords = {
        "epoch": pset_coords["epoch"],
        "calibration_prod": pset_coords["calibration_prod"],
        "background_index": background_indices,
    }

    background_counts = xr.DataArray(
        np.zeros(
            (
                len(bg_coords["epoch"]),
                len(bg_coords["calibration_prod"]),
                len(bg_coords["background_index"]),
            ),
        ),
        dims=[
            "epoch",
            "calibration_prod",
            "background_index",
        ],
        coords=bg_coords,
    )

    # Process direct events
    de_ds = l1b_de_dataset.drop_dims("epoch")

    good_mask = de_ds["trigger_id"].data != de_ds["trigger_id"].attrs["FILLVAL"]
    if not np.any(good_mask):
        return background_counts

    # Remove DEs not in Goodtimes/angles
    goodtimes_mask = good_time_and_phase_mask(
        de_ds.event_met.values,
        de_ds.nominal_bin.values,
        goodtimes_ds,
    )
    de_ds = de_ds.isel(event_met=goodtimes_mask)

    n_events = len(de_ds["event_met"])
    if n_events == 0:
        return background_counts

    # Get TOF configuration (one row per calibration_prod, background_index)
    # TOF windows are the same across ESA steps, so we use get_tof_config()
    tof_config = background_config_df.background_config.get_tof_config()

    for cal_prod in pset_coords["calibration_prod"].values:
        # Check that cal_prod exists in tof_config
        if cal_prod not in tof_config.index.get_level_values("calibration_prod"):
            raise ValueError(
                f"Calibration product {cal_prod} not found in background "
                f"configuration. Available calibration products: "
                f"{sorted(tof_config.index.get_level_values('calibration_prod').unique().tolist())}"
            )

        # Take a cross-section of the TOF configuration DataFrame
        # to get rows relevant to the current calibration product
        cal_prod_rows = tof_config.xs(cal_prod, level="calibration_prod")

        # Use iter_background_events_by_config to get filtered events
        for config_row, filtered_de_ds in iter_background_events_by_config(
            de_ds, cal_prod_rows
        ):
            background_idx = config_row.Index

            if len(filtered_de_ds["event_met"]) == 0:
                continue

            # Count all filtered events
            # (no binning by spin angle since backgrounds are isotropic)
            count = len(filtered_de_ds["event_met"])

            background_counts.loc[
                dict(
                    epoch=pset_coords["epoch"].values[0],
                    calibration_prod=cal_prod,
                    background_index=background_idx,
                )
            ] += count

    return background_counts


def pset_backgrounds(
    pset_coords: dict[str, xr.DataArray],
    background_config_df: pd.DataFrame,
    l1b_de_dataset: xr.Dataset,
    goodtimes_ds: xr.Dataset,
    exposure_times: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """
    Calculate pointing set backgrounds from direct events.

    Computes background counts internally by filtering and binning events
    according to the background configuration, then calculates background
    rates and uncertainties. Scaling factors and uncertainties are applied
    per ESA energy step.

    After computing the combined background rate, a constant offset
    (HiConstants.EXCESS_BACKGROUND_COUNT_RATE) is subtracted to correct for
    excess counts from the outer ESA during background testing. The result
    is clipped to zero to prevent negative rates.

    Parameters
    ----------
    pset_coords : dict[str, xarray.DataArray]
        The PSET coordinates from the xarray.Dataset.
    background_config_df : pandas.DataFrame
        Background configuration DataFrame with MultiIndex
        (calibration_prod, background_index, esa_energy_step).
    l1b_de_dataset : xarray.Dataset
        The L1B dataset for the pointing being processed.
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset with cull_flags.
    exposure_times : xarray.DataArray
        Exposure times with dims (epoch, esa_energy_step, spin_angle_bin).

    Returns
    -------
    dict[str, xarray.DataArray]
        Dictionary containing background_rates and background_rates_uncertainty
        DataArrays to be added to the PSET dataset.
    """
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)

    # Create output arrays
    output_vars = {
        var_name: full_dataarray(
            var_name,
            attr_mgr.get_variable_attributes(f"hi_pset_{var_name}", check_schema=False),
            pset_coords,
        )
        for var_name in ["background_rates", "background_rates_uncertainty"]
    }

    # Get total exposure time
    total_exposure_time = float(exposure_times.sum())

    if total_exposure_time <= 0:
        output_vars["background_rates"].values[:] = 0
        output_vars["background_rates_uncertainty"].values[:] = 0
        return output_vars

    # Compute background counts: shape (epoch, calibration_prod, background_index)
    background_counts = _compute_background_counts(
        pset_coords, background_config_df, l1b_de_dataset, goodtimes_ds
    )

    # Compute count rates: shape (epoch, calibration_prod, background_index)
    count_rates = background_counts / total_exposure_time

    # Convert background config DataFrame to xarray Dataset
    # Config now has dims: (calibration_prod, background_index, esa_energy_step)
    config_ds = background_config_df.to_xarray()

    # Validate calibration products match (compare values, not DataArray metadata)
    if not np.array_equal(
        config_ds["calibration_prod"].values, pset_coords["calibration_prod"].values
    ):
        raise ValueError(
            f"Calibration products in pset_coords and background_config_df "
            f"do not match. pset_coords: {pset_coords['calibration_prod'].values}, "
            f"background_config_df: {config_ds['calibration_prod'].values}"
        )

    # Validate ESA energy steps match (compare values, not DataArray metadata)
    if not np.array_equal(
        config_ds["esa_energy_step"].values, pset_coords["esa_energy_step"].values
    ):
        raise ValueError(
            f"ESA energy steps in pset_coords and background_config_df "
            f"do not match. pset_coords: {pset_coords['esa_energy_step'].values}, "
            f"background_config_df: {config_ds['esa_energy_step'].values}"
        )

    # scaling_factors_da: (calibration_prod, background_index, esa_energy_step)
    scaling_factors_da = config_ds["scaling_factor"]
    uncertainties_da = config_ds["uncertainty"]

    # Compute scaled rates
    # count_rates: (epoch, calibration_prod, background_index)
    # scaling_factors_da: (calibration_prod, background_index, esa_energy_step)
    # scaled_rates: (epoch, calibration_prod, background_index, esa_energy_step)
    scaled_rates = count_rates * scaling_factors_da

    # Compute uncertainties: Poisson + scaling factor (combined in quadrature)
    poisson_unc = (
        np.sqrt(background_counts) / total_exposure_time
    ) * scaling_factors_da
    scaling_unc = count_rates * uncertainties_da
    # combined_unc: (epoch, calibration_prod, background_index, esa_energy_step)
    combined_unc = np.sqrt(poisson_unc**2 + scaling_unc**2)

    # Sum over background_index dimension to get final rates
    # total_rates: (epoch, calibration_prod, esa_energy_step)
    total_rates = scaled_rates.sum(dim="background_index", skipna=True)
    total_unc = np.sqrt((combined_unc**2).sum(dim="background_index", skipna=True))

    # Apply outer ESA background offset correction (do not go negative)
    # This corrects for excess counts from the outer ESA during background testing.
    total_rates = np.maximum(total_rates - HiConstants.EXCESS_BACKGROUND_COUNT_RATE, 0)

    # Add uncertainty related to above excess count rate correction
    # ESAs 7, 8, 9 get an extra 0.0025/s uncertainty to account for possible
    # unidentified additional background in these ESA steps. The constant
    # UPPER_ESA_EXTRA_BACKGROUND_UNC is defined as a xr.DataArray with the correct
    # esa_energy_step coordinate such that it broadcasts appropriately across each
    # calibration product.
    # Fill zeros for any esa_energy_steps not in the extra background DataArray
    upper_esa_unc = HiConstants.UPPER_ESA_EXTRA_BACKGROUND_UNC.reindex(
        esa_energy_step=pset_coords["esa_energy_step"].values,
        fill_value=0.0,
    )
    total_unc = np.sqrt(
        total_unc**2
        + HiConstants.EXCESS_BACKGROUND_COUNT_RATE_UNC**2
        + upper_esa_unc**2
    )

    # Broadcast to output variable dimensions (e.g., epoch, esa_energy_step,
    # calibration_prod, spin_angle_bin). Backgrounds are isotropic, so we
    # broadcast across the last dimension (spin_angle_bin).
    # Get the output dimension order from the output variable (excluding last dim)
    output_dims = output_vars["background_rates"].dims[:-1]
    total_rates_transposed = total_rates.transpose(*output_dims)
    total_unc_transposed = total_unc.transpose(*output_dims)

    output_vars["background_rates"].values[:] = total_rates_transposed.values[
        ..., np.newaxis
    ]
    output_vars["background_rates_uncertainty"].values[:] = total_unc_transposed.values[
        ..., np.newaxis
    ]

    return output_vars


def pset_exposure(
    pset_coords: dict[str, xr.DataArray],
    l1b_de_dataset: xr.Dataset,
    goodtimes_ds: xr.Dataset,
) -> dict[str, xr.DataArray]:
    """
    Calculate PSET exposure time.

    Parameters
    ----------
    pset_coords : dict[str, xarray.DataArray]
        The PSET coordinates from the xarray.Dataset.
    l1b_de_dataset : xarray.Dataset
        The L1B dataset for the pointing being processed.
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset with cull_flags.

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
    data_subset = find_last_de_packet_data(l1b_de_dataset)

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
        mid_tick_mets = clock_tick_mets + HiConstants.HALF_CLOCK_TICK_S

        # Compute spacecraft spin phase first (used for goodtimes filtering)
        spacecraft_spin_phase = np.atleast_1d(get_spacecraft_spin_phase(mid_tick_mets))

        # Convert spacecraft spin phase to nominal_bins (0-89) for goodtimes lookup
        nominal_bins = (spacecraft_spin_phase * 90).astype(np.int32)
        nominal_bins = np.clip(nominal_bins, 0, 89)

        # Compute instrument spin phase from spacecraft spin phase
        # This implementation is identical to spin.get_instrument_spin_phase and
        # is replicated here to avoid querying the spin dataframe again.
        instrument_frame = SpiceFrame[f"IMAP_HI_{sensor_number}"]
        phase_offset = get_spacecraft_to_instrument_spin_phase_offset(instrument_frame)
        spin_phases = (spacecraft_spin_phase + phase_offset) % 1.0

        # Remove ticks not in good times/angles
        good_mask = good_time_and_phase_mask(
            clock_tick_mets, nominal_bins, goodtimes_ds
        )
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
    exposure_var["exposure_times"].values *= HiConstants.DE_CLOCK_TICK_S

    return exposure_var


def find_last_de_packet_data(l1b_dataset: xr.Dataset) -> xr.Dataset:
    """
    Find the telemetry entries for the last packet at an ESA step.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The L1B Direct Event Dataset for the current pointing.

    Returns
    -------
    reduced_dataset : xarray.Dataset
        A dataset containing only the entries for the last packet at an ESA step.
    """
    epoch_dataset = l1b_dataset.drop_dims("event_met")
    # We should get 2, 4, or 8 CCSDS packets per 8-spin ESA step.
    # Get the indices of the packet before each ESA change.
    esa_step = epoch_dataset["esa_step"].values
    esa_energy_step = epoch_dataset["esa_energy_step"].values
    # A change in esa_step should indicate the location of the last packet in
    # each pair of DE packets at an esa_energy_step. In practice, during some
    # calibration activities, it was observed that the esa_energy_step can change
    # when the esa_step did not. So, we look for either to change and use the
    # indices of those changes to identify the last packet in each set. We
    # also need to add the final packet index and assume an energy step change
    # occurs after the final packet.
    last_esa_packet_idx = np.append(
        np.flatnonzero((np.diff(esa_step) != 0) | (np.diff(esa_energy_step) != 0)),
        len(esa_step) - 1,
    )
    # Remove esa energy steps at 0 - these are calibrations
    keep_mask = esa_energy_step[last_esa_packet_idx] != 0
    # Remove esa energy steps at FILLVAL - these are unidentified
    keep_mask &= (
        esa_energy_step[last_esa_packet_idx]
        != l1b_dataset["esa_energy_step"].attrs["FILLVAL"]
    )
    last_esa_packet_idx = last_esa_packet_idx[keep_mask]

    # We don't need to worry about checking that the right number of packets
    # is present for each ESA step because that is done in the Goodtimes processing.

    # Reduce the dataset to just the last packet entries
    data_subset = epoch_dataset.isel(epoch=last_esa_packet_idx)
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
    # CCSDS MET has one second resolution, add two to it to make sure it is
    # greater than the spin start time it ended on. Theotretically, adding
    # one second should be sufficeint, but in practice, with flight data, adding
    # two seconds was found to be necessary.
    end_time_ind = np.flatnonzero(ccsds_met + 2 >= spin_start_mets).max()

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
    clock_tick_mets: np.ndarray = np.arange(
        spin_start_mets[end_time_ind - 8],
        spin_start_mets[end_time_ind],
        HiConstants.DE_CLOCK_TICK_S,
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
    ) / HiConstants.DE_CLOCK_TICK_S
    return clock_tick_mets, clock_tick_weights


def good_time_and_phase_mask(
    mets: np.ndarray,
    nominal_bins: np.ndarray,
    goodtimes_ds: xr.Dataset,
) -> npt.NDArray[np.bool_]:
    """
    Filter out times that are not in good times based on the goodtimes dataset.

    Parameters
    ----------
    mets : np.ndarray
        MET times for each event or clock tick.
    nominal_bins : np.ndarray
        Spacecraft spin bins (0-89) for each event or clock tick.
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset with cull_flags variable dimensioned (met, spin_bin).

    Returns
    -------
    keep_mask : np.ndarray
        Boolean mask indicating which events/ticks are in good times.
    """
    gt_mets = goodtimes_ds["met"].values
    cull_flags = goodtimes_ds["cull_flags"].values

    # Map each event/tick to the nearest goodtimes MET interval
    # searchsorted with side='right' - 1 gives the largest MET <= query MET
    met_indices = np.searchsorted(gt_mets, mets, side="right") - 1
    met_indices = np.clip(met_indices, 0, len(gt_mets) - 1)

    # Convert nominal_bins to int32 for indexing
    spin_bins: npt.NDArray[np.int32] = nominal_bins.astype(np.int32)

    # Look up cull_flags for each event/tick
    # Events are good if cull_flags == 0
    event_cull_flags = cull_flags[met_indices, spin_bins]

    return event_cull_flags == 0
