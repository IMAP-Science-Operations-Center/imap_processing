"""Functions for retrieving spin-table data."""

import functools
import logging
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import typing as npt

from imap_processing.spice import config
from imap_processing.spice.geometry import (
    SpiceFrame,
    get_spacecraft_to_instrument_spin_phase_offset,
)

logger = logging.getLogger(__name__)


def set_global_spin_table_paths(paths: list[Path]) -> None:
    """
    Set the paths to input spin-table csv files.

    Parameters
    ----------
    paths : list[pathlib.Path]
        List of paths to spin-table csv files that will be used to supply
        spin-table data.
    """
    # If paths is an empty list, do nothing
    if not paths:
        return
    logger.info(
        f"Using the following spin-tables in processing: {[p.name for p in paths]}"
    )
    config._spin_table_paths = paths


def get_spin_data(use_corrected_spin_start: bool = True) -> pd.DataFrame:
    """
    Read spin-tables and return spin data.

    The spin-tables to read are stored in the mutable module level attribute
    named `spin_table_paths`.

    Parameters
    ----------
    use_corrected_spin_start : bool
        If True (default), the corrected spin start times replace the original ones: the
        values of each `*_corr` column below are returned under the column name
        without the `_corr` suffix, and no `_corr` columns are returned. If False, all
        the columns below are returned, with the un-suffixed columns computed from the
        original spin start times.

    Returns
    -------
    spin_data : pandas.DataFrame
        Spin data. The DataFrame will have the following columns:

            * `spin_number`: Unique integer spin number.
            * `spin_start_sec_sclk`: MET seconds of spin start time.
            * `spin_start_subsec_sclk`: MET microseconds of spin start time.
            * `spin_start_met`: Floating point MET seconds of spin start.
            * `spin_start_utc`: UTC string of spin start time.
            * `spin_start_sec_sclk_corr`: MET seconds of the corrected spin start
              time.
            * `spin_start_subsec_sclk_corr`: MET microseconds of the corrected
              spin start time.
            * `spin_start_utc_corr`: UTC string of the corrected spin start time.
            * `spin_period_sec`: Floating point spin period in seconds (estimated).
            * `actual_spin_period`: Floating point actual spin period computed from
              consecutive spin start times. More accurate than spin_period_sec.
            * `spin_period_valid`: Boolean indicating whether spin period is valid.
            * `spin_phase_valid`: Boolean indicating whether spin phase is valid.
            * `spin_period_source`: Source used for determining spin period.
            * `thruster_firing`: Boolean indicating whether thruster is firing.
            * `spin_start_met_corr`: Floating point MET seconds of the corrected
              spin start.
            * `actual_spin_period_corr`: Floating point actual spin period computed
              from consecutive corrected spin start times.

    Raises
    ------
    ValueError
        If no spin-table paths have been set.
    """
    if config._spin_table_paths is None or len(config._spin_table_paths) == 0:
        # Handle the case where the module attribute is not set
        raise ValueError(
            "Spin-table paths have not been defined in spin.py "
            "module attribute spin_table_paths."
        )

    return _load_spin_data_with_cache(
        tuple(config._spin_table_paths), use_corrected_spin_start
    )


@functools.cache
def _load_spin_data_with_cache(
    csv_paths: tuple[Path], use_corrected_spin_start: bool
) -> pd.DataFrame:
    """
    Load spin-table data from csv files and combine them.

    Parameters
    ----------
    csv_paths : tuple[Path]
        Locations of spin-table csv files.
    use_corrected_spin_start : bool
        If True, the values of each `*_corr` column replace those of the column
        without the `_corr` suffix, and the `_corr` columns are dropped.

    Returns
    -------
    combined_df: pandas.DataFrame
        The dataframe containing all spin data.

    Raises
    ------
    ValueError
        If a spin-table lacks the corrected spin start time columns.
    """
    logger.debug(
        f"Merging the following spin tables files: {[sp.name for sp in csv_paths]}"
    )

    # Reversed sorting is used so that the proper precedence is applied in
    # the below use of DataFrame.combine_first()
    sorted_paths = sorted(csv_paths, reverse=True)
    spin_dataframes = [
        pd.read_csv(
            spin_table_path,
            comment="#",
            index_col="spin_number",
            dtype={
                "spin_number": int,
                "spin_start_sec_sclk": int,
                "spin_start_subsec_sclk": int,
                "spin_start_utc": str,
                "spin_start_sec_sclk_corr": int,
                "spin_start_subsec_sclk_corr": int,
                "spin_start_utc_corr": str,
                "spin_period_sec": float,
                "spin_period_valid": bool,
                "spin_phase_valid": bool,
                "spin_period_source": int,
                "thruster_firing": bool,
            },
        )
        for spin_table_path in sorted_paths
    ]
    corrected_columns = [
        "spin_start_sec_sclk_corr",
        "spin_start_subsec_sclk_corr",
        "spin_start_utc_corr",
    ]
    for spin_table_path, spin_dataframe in zip(
        sorted_paths, spin_dataframes, strict=True
    ):
        missing = [col for col in corrected_columns if col not in spin_dataframe]
        if missing:
            raise ValueError(
                f"Spin table {spin_table_path.name} is missing the corrected spin "
                f"start time columns {missing}."
            )
    combined_df = reduce(
        lambda left, right: left.combine_first(right),
        spin_dataframes,
    )
    # Duplicate the index so that users can access "spin_numer" by name
    combined_df.insert(0, "spin_number", combined_df.index)
    spin_numbers = combined_df["spin_number"].values
    spin_periods_sec = combined_df["spin_period_sec"].values
    # Compute spin_start_met / actual_spin_period from the original spin start
    # times, and spin_start_met_corr / actual_spin_period_corr from the corrected
    # ones.
    for suffix in ("", "_corr"):
        # Combine spin_start_sec_sclk and spin_start_subsec_sclk to get the spin
        # start time in seconds. The spin start subseconds are in microseconds.
        spin_start_met = (
            combined_df[f"spin_start_sec_sclk{suffix}"]
            + combined_df[f"spin_start_subsec_sclk{suffix}"] / 1e6
        )
        combined_df[f"spin_start_met{suffix}"] = spin_start_met
        # Precompute actual spin periods from consecutive spin start times
        # Only use actual periods when spin numbers increment by exactly 1
        # This prevents invalid times from appearing valid when spins are missing
        # Otherwise use the estimated spin_period_sec
        actual_spin_periods = np.where(
            np.diff(spin_numbers) == 1,
            np.diff(spin_start_met.values),
            spin_periods_sec[:-1],
        )
        # For the last spin, use the provided spin_period_sec since there's no
        # next spin
        combined_df[f"actual_spin_period{suffix}"] = np.append(
            actual_spin_periods, spin_periods_sec[-1]
        )

    if use_corrected_spin_start:
        corr_columns = [c for c in combined_df.columns if c.endswith("_corr")]
        for column in corr_columns:
            combined_df[column.removesuffix("_corr")] = combined_df[column]
        combined_df = combined_df.drop(columns=corr_columns)
    return combined_df


def interpolate_spin_data(
    query_met_times: float | npt.NDArray, use_corrected_spin_start: bool = True
) -> pd.DataFrame:
    """
    Interpolate spin table data to the queried MET times.

    All columns in the spin table csv file are interpolated to the previous
    table entry. A sc_spin_phase column is added that is the computed spacecraft
    spin phase at the queried MET times. Note that spin phase is by definition,
    in the interval [0, 1) where 1 is equivalent to 360 degrees.

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).
    use_corrected_spin_start : bool
        If True (default), rows are selected, and `sc_spin_phase` computed, using the
        corrected spin start times, and the columns are those of
        :py:func:`get_spin_data` with corrected spin start times replacing the
        original ones. If False, rows are selected, and
        `sc_spin_phase` computed, using the original spin start times; the
        `spin_number_corr` and `sc_spin_phase_corr` columns are also added,
        giving the spin number and spin phase computed using the corrected spin
        start times.

    Returns
    -------
    spin_df : pandas.DataFrame
        Spin table data interpolated for each queried MET time. In addition to
        the columns output from :py:func:`get_spin_data`, the `sc_spin_phase`
        column (and the `spin_number_corr` and `sc_spin_phase_corr` columns if
        `use_corrected_spin_start` is False) is added and is uniquely computed
        for each queried MET time.
    """
    spin_df = get_spin_data(use_corrected_spin_start)

    # Ensure query_met_times is an array
    query_met_times = np.asarray(query_met_times)
    is_scalar = query_met_times.ndim == 0
    if is_scalar:
        # Force scalar to array because np.asarray() will not
        # convert scalar to array
        query_met_times = np.atleast_1d(query_met_times)

    # Compute the spin number and spin phase using the un-suffixed spin start
    # times (corrected or original per use_corrected_spin_start), then, if the
    # original ones were used, using the corrected ones.
    suffixes = ("",) if use_corrected_spin_start else ("", "_corr")
    for suffix in suffixes:
        # Cache frequently accessed arrays to avoid repeated .values calls
        spin_start_met = spin_df[f"spin_start_met{suffix}"].values
        actual_spin_periods = spin_df[f"actual_spin_period{suffix}"].values

        # Make sure input times are within the bounds of spin data
        spin_df_start_time = spin_start_met[0]
        spin_df_end_time = spin_start_met[-1] + actual_spin_periods[-1]
        input_start_time = query_met_times.min()
        input_end_time = query_met_times.max()
        if input_start_time < spin_df_start_time or input_end_time >= spin_df_end_time:
            raise ValueError(
                f"Query times, {query_met_times} are outside of the spin data range, "
                f"{spin_df_start_time, spin_df_end_time}."
            )

        # Find all spin time that are less or equal to query_met_times.
        # To do that, use side right, a[i-1] <= v < a[i], in the searchsorted.
        # Eg.
        # >>> df['a']
        # array([0, 15, 30, 45, 60])
        # >>> np.searchsorted(df['a'], [0, 13, 15, 32, 70], side='right')
        # array([1, 1, 2, 3, 5])
        last_spin_indices = (
            np.searchsorted(spin_start_met, query_met_times, side="right") - 1
        )

        # Calculate spin phase using actual computed periods
        spin_phases = (
            query_met_times - spin_start_met[last_spin_indices]
        ) / actual_spin_periods[last_spin_indices]

        # Check for invalid spin phase using below checks:
        # 1. Check that the spin phase is in valid range, [0, 1).
        # 2. Check invalid spin phase using spin_phase_valid,
        #   spin_period_valid columns.
        invalid_spin_phase_range = (spin_phases < 0) | (spin_phases >= 1)

        # TODO: add optional to filter this if this flag means
        # that repointing is happening. otherwise, then keep it.
        # This needs to be discussed and receive guidance at
        # the project level.
        invalid_spins = (spin_df["spin_phase_valid"].values[last_spin_indices] == 0) | (
            spin_df["spin_period_valid"].values[last_spin_indices] == 0
        )
        bad_spin_phases = invalid_spin_phase_range | invalid_spins
        spin_phases[bad_spin_phases] = np.nan

        if suffix == "":
            # Generate a dataframe with one row per query time
            out_df = spin_df.iloc[last_spin_indices].copy()
        # Add spin_number and spin_phase columns to output dataframe
        # (spin_number is unchanged for the un-suffixed spin start times)
        out_df[f"spin_number{suffix}"] = spin_df["spin_number"].values[
            last_spin_indices
        ]
        out_df[f"sc_spin_phase{suffix}"] = spin_phases

    return out_df


def get_spin_number(
    met_time: float | npt.NDArray, use_corrected_spin_start: bool = True
) -> int | npt.NDArray:
    """
    Get the spin number for the input query time.

    The spin number is the index of the spin table row that contains the
    spin data for the input query time.

    Parameters
    ----------
    met_time : float or np.ndarray
        Query time in Mission Elapsed Time (MET).
    use_corrected_spin_start : bool
        If True (default), use the corrected spin start times
        (`spin_start_sec_sclk_corr`, `spin_start_subsec_sclk_corr`); otherwise
        use the original spin start times (`spin_start_sec_sclk`,
        `spin_start_subsec_sclk`).

    Returns
    -------
    spin_number : int or np.ndarray
        Spin number for the input query time.
    """
    spin_df = interpolate_spin_data(met_time, use_corrected_spin_start)
    spin_numbers = spin_df["spin_number"].values

    return spin_numbers.item() if np.asarray(met_time).ndim == 0 else spin_numbers


def get_spin_angle(
    spin_phases: float | npt.NDArray,
    degrees: bool = False,
) -> float | npt.NDArray:
    """
    Convert spin_phases to radians or degrees.

    Parameters
    ----------
    spin_phases : float or np.ndarray
        Instrument or spacecraft spin phases. Spin phase is a
        floating point number in the range [0, 1) corresponding to the
        spin angle / 360.
    degrees : bool
        If degrees parameter is True, return angle in degrees otherwise return angle in
        radians. Default is False.

    Returns
    -------
    spin_phases : float or np.ndarray
        Spin angle in degrees or radians for the input query times.
    """
    if np.any(spin_phases < 0) or np.any(spin_phases >= 1):
        raise ValueError(
            f"Spin phases, {spin_phases} are outside of the expected spin phase range, "
            f"[0, 1) "
        )
    if degrees:
        # Convert to degrees
        return spin_phases * 360
    else:
        # Convert to radians
        return spin_phases * 2 * np.pi


def get_spacecraft_spin_phase(
    query_met_times: float | npt.NDArray, use_corrected_spin_start: bool = True
) -> float | npt.NDArray:
    """
    Get the spacecraft spin phase for the input query times.

    Formula to calculate spin phase:
        spin_phase = (query_met_times - spin_start_met) / spin_period_sec

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).
    use_corrected_spin_start : bool
        If True (default), use the corrected spin start times
        (`spin_start_sec_sclk_corr`, `spin_start_subsec_sclk_corr`); otherwise
        use the original spin start times (`spin_start_sec_sclk`,
        `spin_start_subsec_sclk`).

    Returns
    -------
    spin_phase : float or np.ndarray
        Spin phase for the input query times.
    """
    spin_df = interpolate_spin_data(query_met_times, use_corrected_spin_start)
    if np.asarray(query_met_times).ndim == 0:
        return spin_df["sc_spin_phase"].values[0]
    return spin_df["sc_spin_phase"].values


def get_instrument_spin_phase(
    query_met_times: float | npt.NDArray,
    instrument: SpiceFrame,
    use_corrected_spin_start: bool = True,
) -> float | npt.NDArray:
    """
    Get the instrument spin phase for the input query times.

    Formula to calculate spin phase:
        instrument_spin_phase = (spacecraft_spin_phase + instrument_spin_offset) % 1

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).
    instrument : SpiceFrame
        Instrument frame to calculate spin phase for.
    use_corrected_spin_start : bool
        If True (default), use the corrected spin start times
        (`spin_start_sec_sclk_corr`, `spin_start_subsec_sclk_corr`); otherwise
        use the original spin start times (`spin_start_sec_sclk`,
        `spin_start_subsec_sclk`).

    Returns
    -------
    spin_phase : float or np.ndarray
        Instrument spin phase for the input query times. Spin phase is a
        floating point number in the range [0, 1) corresponding to the
        spin angle / 360.
    """
    spacecraft_spin_phase = get_spacecraft_spin_phase(
        query_met_times, use_corrected_spin_start
    )
    instrument_spin_phase_offset = get_spacecraft_to_instrument_spin_phase_offset(
        instrument
    )
    return (spacecraft_spin_phase + instrument_spin_phase_offset) % 1
