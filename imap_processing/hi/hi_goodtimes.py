"""IMAP-HI Goodtimes processing module."""

import logging
import re
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.hi.utils import CoincidenceBitmap, parse_sensor_number
from imap_processing.quality_flags import ImapHiL1bDeFlags

logger = logging.getLogger(__name__)

# Structured dtype for good time intervals
INTERVAL_DTYPE = np.dtype(
    [
        ("met_start", np.float64),
        ("met_end", np.float64),
        ("spin_bin_low", np.uint8),
        ("spin_bin_high", np.uint8),
        ("n_good_bins", np.uint8),
        ("esa_step", np.uint8),
    ]
)


class CullCode(IntEnum):
    """Cull reason codes for good/bad time classification."""

    GOOD = 0
    LOOSE = 1


def create_goodtimes_dataset(l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Create goodtimes dataset from L1B Direct Event data.

    Initializes all times and spin bins as good (cull_flags=0). The goodtimes
    dataset is created with one entry per unique MET timestamp found in the
    L1B DE data. Culling functions (e.g., mark_incomplete_spin_sets) should be
    called after creation to identify and flag bad times.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        L1B direct event data for this pointing. Used to extract MET timestamps
        for each 8-spin interval.

    Returns
    -------
    xarray.Dataset
        Initialized goodtimes dataset with cull_flags set to 0 (all good).
        Access goodtimes methods via the .goodtimes accessor
        (e.g., dataset.goodtimes.remove_times()).
    """
    logger.info("Creating Goodtimes from L1B Direct Event data")

    # Extract MET times from esa_step_met
    # Each MET represents one 8-spin histogram packet interval
    met_all = l1b_de["esa_step_met"]
    logger.debug(f"Extracted {len(met_all)} total MET entries from L1B DE data")

    # Find unique MET values and indices of first occurrences
    unique_mets, first_indices = np.unique(met_all.values, return_index=True)
    logger.info(f"Found {len(unique_mets)} unique MET values")

    # Extract data for unique METs (use first occurrence of each)
    met = met_all.isel(epoch=first_indices)
    esa_step = l1b_de["esa_step"].isel(epoch=first_indices)

    # Create coordinates
    coords = {
        "met": met.values,
        "spin_bin": np.arange(90),
    }

    # Create data variables
    # Initialize cull_flags - all good (0) by default
    # Shape: (n_met_timestamps, 90 spin_bins)
    # Per alg doc Section 2.3.2: 90-element arrays, one per histogram packet
    # Culling functions will set non-zero cull codes for bad times
    data_vars = {
        "cull_flags": xr.DataArray(
            np.zeros((len(met), 90), dtype=np.uint8),
            dims=["met", "spin_bin"],
        ),
        "esa_step": esa_step,
    }

    # Create attributes
    sensor_number = parse_sensor_number(l1b_de.attrs["Logical_source"])
    match = re.match(r"repoint(?P<pointing_num>\d{5})", l1b_de.attrs["Repointing"])
    if not match:
        raise ValueError(
            f"Unable to parse pointing number from l1b_de Repointing "
            f"attribute: {l1b_de.attrs['Repointing']}"
        )
    attrs = {
        "sensor": f"Hi{sensor_number}",
        "pointing": int(match["pointing_num"]),
    }

    return xr.Dataset(data_vars, coords, attrs)


@xr.register_dataset_accessor("goodtimes")
class GoodtimesAccessor:
    """
    Extend xarray.Dataset with accessor for IMAP-Hi Good Times operations.

    Provides methods to track and manage good/bad time intervals for a single
    Pointing based on validation checks defined in the IMAP-Hi Algorithm
    Document Section 2.2.4 and 2.3.2.

    The accessor operates on xr.Dataset objects created by create_goodtimes_dataset().
    The dataset maintains a cull_flags array initialized to all zeros (good).
    As bad times are identified by validation algorithms, they are flagged via
    the `remove_times()` method with a non-zero cull code.

    Cull Codes:
      * 0 : Good time (default)
      * 1-N : Bad time, with specific cull reason code

    Expected xarray.Dataset structure:
      * Dimensions:
        * met : int
          Number of MET timestamps (one per 8-spin histogram packet, ~90 per pointing)
        * spin_bin : int
          Number of spin angle bins (90 bins covering 0-360 degrees)
      * Coordinates
        * met : numpy.ndarray
          Mission Elapsed Time values for each 8-spin interval
        * spin_bin : numpy.ndarray
          Spin bin indices (0-89)
      * Data Variables
        * cull_flags : xarray.DataArray (met, spin_bin)
          Cull flags where 0=good time, non-zero=bad time with cull reason code
        * esa_step : xarray.DataArray (met,)
          ESA energy step for each MET timestamp
      * Attributes
        * sensor : str
         Sensor identifier ('Hi45' or 'Hi90')
        * pointing : int
         Pointing number for this dataset

    Parameters
    ----------
    xarray_obj : xarray.Dataset
        The xarray Dataset to wrap with goodtimes accessor functionality.

    Examples
    --------
    >>> gt_dataset = create_goodtimes_dataset(l1b_de)
    >>> gt_dataset.goodtimes.mark_bad_times(met=1000.5, cull=CullCode.LOOSE)
    >>> intervals = gt_dataset.goodtimes.get_good_intervals()
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialize the accessor with an xarray Dataset."""
        self._obj = xarray_obj

    def mark_bad_times(
        self,
        met: np.ndarray | float | tuple[float, float],
        bins: np.ndarray | int | None = None,
        cull: int = 1,
    ) -> None:
        """
        Flag specific MET times and spin bins as bad times with a cull code.

        This method is called by external validation algorithms when bad times
        are identified. It sets the cull_flags to the specified non-zero cull code
        for the given MET timestamps and spin bins.

        Parameters
        ----------
        met : numpy.ndarray, float, or tuple of (float, float)
            MET timestamp(s) to flag as bad. Can be:
            - Single float: one MET timestamp
            - Tuple of (start, end): time range (inclusive)
            - Array of floats: multiple MET timestamps
        bins : numpy.ndarray, int, or None
            Spin bin(s) to flag as bad. Can be:
            - None: flag all spin bins (0-89) for the given MET(s)
            - Single int: one spin bin
            - Array of ints: multiple spin bins
        cull : int
            Cull reason code (non-zero). Different validation checks can use
            different codes to identify the reason for culling:
            - 1: Loose criterion
            - etc.

        Notes
        -----
        If a time/bin is already flagged with a different cull code, this method
        will overwrite it with the new cull code. Consider implementing logic to
        preserve or combine cull codes if needed.

        Examples
        --------
        >>> # Flag all spin bins for MET=1000.5 as loose (cull=1)
        >>> goodtimes.mark_bad_times(met=1000.5, bins=None, cull=CullCode.LOOSE)

        >>> # Flag spin bins 0-10 for MET=1000.5
        >>> goodtimes.mark_bad_times(
        ...     met=1000.5, bins=np.arange(11), cull=CullCode.LOOSE
        ... )

        >>> # Flag time range around a repoint (240s before/after)
        >>> repoint_time = 1000.0
        >>> goodtimes.mark_bad_times(
        ...     met=(repoint_time - 240, repoint_time + 240),
        ...     cull=CullCode.LOOSE
        ... )

        >>> # Flag multiple specific METs, all bins
        >>> goodtimes.mark_bad_times(
        ...     met=np.array([1000.5, 1001.5]), bins=None, cull=CullCode.LOOSE
        ... )
        """
        if cull == 0:
            raise ValueError("Cull code must be non-zero. Use 0 only for good times.")

        # Handle bins parameter
        if bins is None:
            # Flag all spin bins (0-89)
            bins_array = np.arange(90)
        else:
            # Convert to array for consistent handling
            bins_array = np.atleast_1d(bins)

        # Validate bin indices
        if np.any((bins_array < 0) | (bins_array >= 90)):
            raise ValueError("Spin bins must be in range [0, 89]")

        met_values = self._obj.coords["met"].values

        # check for met times out of range
        met_array = np.atleast_1d(met)
        # Add the difference between the last two MET values to the valid range
        # to get the time of the last MET + 8_spins
        if len(met_values) >= 2:
            met_interval = np.diff(met_values[-2:])[0]
        elif len(met_values) == 1:
            # Only one MET value - use a default interval (120 seconds)
            met_interval = 120.0
        else:
            # No MET values - can't validate range
            met_interval = 0.0

        valid_met_range = (met_values[0], met_values[-1] + met_interval)
        invalid_met_mask = (met_array < valid_met_range[0]) | (
            met_array > valid_met_range[-1]
        )
        if np.any(invalid_met_mask):
            raise ValueError(
                f"MET value(s) {met_array[invalid_met_mask]} are "
                f"outside valid range: {valid_met_range}"
            )

        # Handle time range input (tuple of start, end)
        if isinstance(met, tuple) and len(met) == 2:
            met_start, met_end = met
            # Find all MET indices within the range
            in_range = (met_values >= met_start) & (met_values <= met_end)
            met_indices = np.nonzero(in_range)[0]
        else:
            # Find indices of largest MET that is <= each met_val (vectorized)
            # searchsorted with side='right' gives first index where value would go
            # Subtract 1 to get the largest value <= met_val
            met_indices = np.searchsorted(met_values, met_array, side="right") - 1

        # Set cull_flags for all indices
        n_times = len(met_indices)
        n_bins = len(bins_array)
        logger.debug(
            f"Flagging {n_times} MET time(s) x {n_bins} spin bin(s) with "
            f"cull code {cull}"
        )
        self._obj["cull_flags"].values[np.ix_(met_indices, bins_array)] = cull

    def get_good_intervals(self) -> np.ndarray:
        """
        Extract good time intervals for each MET timestamp.

        Creates an interval for each MET time that has good bins. Since ESA step
        changes at each MET, each MET gets its own interval(s).

        If good bins wrap around the 89->0 boundary (e.g., bins 88,89,0,1), multiple
        intervals are created for the same MET time, one for each contiguous set.

        Returns
        -------
        numpy.ndarray
            Structured array with dtype INTERVAL_DTYPE containing:
            - met_start: MET timestamp of interval
            - met_end: MET timestamp of interval (same as met_start)
            - spin_bin_low: Lowest good spin bin in interval
            - spin_bin_high: Highest good spin bin in interval
            - n_good_bins: Number of good bins
            - esa_step: ESA energy step for this MET

        Notes
        -----
        This is used for generating the Good Times output files per algorithm
        document Section 2.3.2.5.
        """
        logger.debug("Extracting good time intervals")
        intervals: list[np.void] = []
        met_values = self._obj.coords["met"].values
        cull_flags = self._obj["cull_flags"].values
        esa_steps = self._obj["esa_step"].values

        if len(met_values) == 0:
            logger.warning("No MET values found, returning empty intervals array")
            return np.array([], dtype=INTERVAL_DTYPE)

        # Process each MET time
        for met_idx in range(len(met_values)):
            self._add_intervals_for_pattern(
                intervals,
                met_values[met_idx],
                met_values[met_idx],  # met_start == met_end
                cull_flags[met_idx, :],
                esa_steps[met_idx],
            )

        logger.info(f"Extracted {len(intervals)} good time intervals")
        return np.array(intervals, dtype=INTERVAL_DTYPE)

    def _add_intervals_for_pattern(
        self,
        intervals: list,
        met_start: float,
        met_end: float,
        pattern: np.ndarray,
        esa_step: int,
    ) -> None:
        """
        Add interval(s) for a cull_flags pattern, splitting if bins wrap around.

        Parameters
        ----------
        intervals : list
            List to append interval tuples to.
        met_start : float
            Start MET timestamp.
        met_end : float
            End MET timestamp.
        pattern : numpy.ndarray
            Cull flags pattern for spin bins.
        esa_step : int
            ESA energy step for this MET.
        """
        good_bins = np.nonzero(pattern == 0)[0]

        if len(good_bins) == 0:
            return

        # Check for gaps in good_bins (indicating separate contiguous regions)
        # Bins are contiguous if difference between consecutive bins is 1
        gaps = np.nonzero(np.diff(good_bins) > 1)[0]

        if len(gaps) == 0:
            # No gaps - single contiguous region
            interval = (
                met_start,
                met_end,
                good_bins[0],
                good_bins[-1],
                len(good_bins),
                esa_step,
            )
            intervals.append(interval)
        else:
            # Multiple contiguous regions - split at gaps
            start_idx = 0
            for gap_idx in gaps:
                # Create interval for bins before the gap
                bins_segment = good_bins[start_idx : gap_idx + 1]
                interval = (
                    met_start,
                    met_end,
                    bins_segment[0],
                    bins_segment[-1],
                    len(bins_segment),
                    esa_step,
                )
                intervals.append(interval)
                start_idx = gap_idx + 1

            # Handle final segment after last gap
            bins_segment = good_bins[start_idx:]
            interval = (
                met_start,
                met_end,
                bins_segment[0],
                bins_segment[-1],
                len(bins_segment),
                esa_step,
            )
            intervals.append(interval)

    def get_cull_statistics(self) -> dict:
        """
        Calculate statistics on cull codes for diagnostics.

        Returns
        -------
        dict
            Dictionary with cull code statistics:
            - total_bins: Total number of MET × spin_bin combinations
            - good_bins: Number of bins with cull_flags=0
            - culled_bins: Number of bins with cull_flags>0
            - fraction_good: Fraction of bins that are good
            - cull_code_counts: Dict mapping cull codes to counts
        """
        total_bins = self._obj["cull_flags"].size
        culled_bins = np.count_nonzero(self._obj["cull_flags"])
        good_bins = total_bins - culled_bins

        # Count occurrences of each cull code
        unique_codes, counts = np.unique(
            self._obj["cull_flags"].values[self._obj["cull_flags"].values > 0],
            return_counts=True,
        )
        cull_code_counts = dict(
            zip(unique_codes.tolist(), counts.tolist(), strict=False)
        )

        return {
            "total_bins": int(total_bins),
            "good_bins": int(good_bins),
            "culled_bins": int(culled_bins),
            "fraction_good": good_bins / total_bins if total_bins > 0 else 0.0,
            "cull_code_counts": cull_code_counts,
        }

    def write_txt(self, output_path: Path) -> Path:
        """
        Write good times to text file in the format specified by algorithm document.

        Format per Section 2.3.2.5:
        pointing MET_start MET_end spin_bin_low spin_bin_high sensor esa_step
        [rate/sigma values...]

        Parameters
        ----------
        output_path : pathlib.Path
            Path where the text file should be written.

        Returns
        -------
        pathlib.Path
            Path to the created file.
        """
        logger.info(f"Writing good times to file: {output_path}")
        intervals = self.get_good_intervals()

        with open(output_path, "w") as f:
            for interval in intervals:
                pointing = self._obj.attrs.get("pointing", 0)
                sensor = self._obj.attrs["sensor"]

                # Format:
                # pointing met_start met_end spin_bin_low spin_bin_high sensor esa_step
                line = (
                    f"{pointing:05d} "
                    f"{int(interval['met_start'])} "
                    f"{int(interval['met_end'])} "
                    f"{interval['spin_bin_low']} "
                    f"{interval['spin_bin_high']} "
                    f"{sensor} "
                    f"{interval['esa_step']}"
                )

                # TODO: Add rate/sigma values for each ESA step

                f.write(line + "\n")

        logger.info(f"Wrote {len(intervals)} intervals to {output_path}")
        return output_path


# ==============================================================================
# Culling/Filtering Functions
# Based on culling.c - Reference: IMAP-Hi Algorithm Document Sections 2.2.4, 2.3.2
# ==============================================================================


def mark_incomplete_spin_sets(
    goodtimes_ds: xr.Dataset,
    l1b_de: xr.Dataset,
    cull_code: int = CullCode.LOOSE,
) -> None:
    """
    Filter out incomplete 8-spin histogram periods.

    Ensures data completeness by removing histogram packets that don't represent
    complete 8-spin periods. Histogram packets are the fundamental time unit for
    IMAP-Hi science data, and incomplete periods indicate data gaps or telemetry
    issues that would compromise scientific analysis.

    Algorithm Document Reference:
        Section 2.3.2: Good times selection requiring complete data coverage

    Background:
        Direct Event (DE) packets contain the "last_spin_num" field indicating
        which spin number (1-8) was the last spin included in that packet. The
        instrument can operate in different cadences:
          - Every 4th spin: last_spin_num values of 4 and 8 only
          - Every 2nd spin: last_spin_num values of 2, 4, 6, 8
          - Every spin: last_spin_num values of 1-8

        For a complete 8-spin period, we must see all the expected last_spin_num values
        with no gaps. The cadence cannot change during HVSCI mode.

    Parameters
    ----------
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset to update with cull flags.
    l1b_de : xarray.Dataset
        L1B Direct Event data containing DE packets with last_spin_num field
        and ccsds_qf quality flag.
    cull_code : int, optional
        Cull code to use for marking bad times (default: CullCode.LOOSE).

    Notes
    -----
    This function modifies goodtimes_ds in place by calling mark_bad_times()
    for MET timestamps with incomplete spin coverage.
    """
    logger.info("Running mark_incomplete_spin_sets culling")

    met_values = goodtimes_ds.coords["met"].values

    # Get DE packet MET times directly from esa_step_met
    de_met = l1b_de["esa_step_met"]

    # Assign each DE packet to nearest goodtimes MET using searchsorted
    # This maps each DE packet to a MET index
    met_indices = np.searchsorted(met_values, de_met.values, side="right") - 1

    # Clip to valid range
    met_indices = np.clip(met_indices, 0, len(met_values) - 1)

    # Calculate actual distance to assigned MET
    time_slop = 10.0  # seconds tolerance
    distances = np.abs(de_met.values - met_values[met_indices])
    valid_assignment = distances <= time_slop

    # Create a new coordinate in l1b_de for grouping
    l1b_de_with_group = l1b_de.assign_coords(met_group=("epoch", met_indices))

    # Only keep packets with valid time assignment
    l1b_de_valid = l1b_de_with_group.isel(epoch=valid_assignment)

    # Valid pattern bitmasks
    valid_pattern_1 = 0b10001000  # bits 3,7: every 4th spin (last_spin_num 4,8)
    valid_pattern_2 = 0b10101010  # bits 1,3,5,7: every 2nd spin (2,4,6,8)
    valid_pattern_3 = 0b11111111  # bits 0-7: every spin (1-8)
    valid_patterns = [valid_pattern_1, valid_pattern_2, valid_pattern_3]

    # Group by MET and validate each group
    bad_mets = []

    for met_idx, group in l1b_de_valid.groupby("met_group"):
        met_time = met_values[met_idx]

        # Check for invalid spins flag (bit 1 in ccsds_qf)
        if np.any((group["ccsds_qf"].values & ImapHiL1bDeFlags.BADSPIN) != 0):
            bad_mets.append(met_time)
            continue

        # Get last_spin_num values for this group
        last_spin_num_values = group["last_spin_num"].values

        # Count occurrences of each last_spin_num value (1-8)
        last_spin_num_counts = np.bincount(
            last_spin_num_values,
            minlength=9,
        )[1:9]  # bins 1-8, ignore 0

        # Check if we have exactly one of each expected last_spin_num value
        # has_exactly_one[i] corresponds to last_spin_num i+1
        # bit i in pattern_bits represents last_spin_num i+1
        has_exactly_one = last_spin_num_counts == 1
        pattern_bits = np.packbits(has_exactly_one, bitorder="little")[0]

        if pattern_bits not in valid_patterns:
            bad_mets.append(met_time)

    # Also mark MET times with no DE packets as bad
    mets_with_packets = np.unique(met_indices[valid_assignment])
    all_met_indices = np.arange(len(met_values))
    mets_without_packets = np.setdiff1d(all_met_indices, mets_with_packets)
    bad_mets.extend(met_values[mets_without_packets])

    # Remove all bad times at once
    if bad_mets:
        goodtimes_ds.goodtimes.mark_bad_times(met=np.array(bad_mets), cull=cull_code)

    logger.info(f"Dropped {len(bad_mets)} incomplete 8-spin period(s)")


def mark_drf_times(
    goodtimes_ds: xr.Dataset,
    hk: xr.Dataset,
    cull_code: int = CullCode.LOOSE,
) -> None:
    """
    Remove times during spacecraft drift restabilization.

    Filters out data collected during and immediately after Drift Restabilization
    Flag (DRF) periods. When the spacecraft drift rate exceeds acceptable limits,
    the DRF is asserted and the spacecraft performs a restabilization maneuver.
    During restabilization, the spacecraft pointing is unstable, making the data
    unsuitable for science.

    Algorithm Document Reference:
        Section 2.2.4: Housekeeping checks for spacecraft attitude and pointing
        Section 2.2.7: Bad times during spacecraft maneuvers

    Background:
        The spacecraft must maintain precise pointing for Hi sensors to correctly
        measure ENA arrival directions. When DRF is asserted, the spacecraft is
        performing active stabilization, and pointing may be off-nominal for up to
        30 minutes after DRF deasserts. This implementation conservatively removes
        all times within 30 minutes following DRF deassertion.

    Parameters
    ----------
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset to update with cull flags.
    hk : xarray.Dataset
        Housekeeping data containing DRF status in fsw_thruster_warn field.
    cull_code : int, optional
        Cull code to use for marking bad times (default: CullCode.LOOSE).

    Notes
    -----
    This function modifies goodtimes_ds in place. If no housekeeping data is
    available, a warning is logged but no times are removed.
    """
    logger.info("Running mark_drf_times culling")

    if len(hk.epoch) == 0:
        logger.warning("No NHK loaded to check for DRF times")
        return

    # Get HK times and DRF status from fsw_thruster_warn
    hk_met = hk["ccsds_met"]
    drf_status = hk["fsw_thruster_warn"].values != 0

    # Find transitions from DRF active (1) to inactive (0) using numpy.diff
    drf_diff = np.diff(drf_status.astype(int))
    # Transition from 1->0 shows as -1 in diff
    # diff[i] = status[i+1] - status[i], so add 1 to get index where it became 0
    transition_indices = np.nonzero(drf_diff == -1)[0] + 1
    # Ensure transition_indices is always iterable, even if a scalar is returned
    transition_indices = np.atleast_1d(transition_indices)

    # For each DRF deactivation, remove times in 30-minute window before
    for idx in transition_indices:
        drf_end_time = hk_met.values[idx]
        window_start = drf_end_time - 30 * 60  # 30 minutes before

        # Remove time range using tuple input
        goodtimes_ds.goodtimes.mark_bad_times(
            met=(window_start, drf_end_time), cull=cull_code
        )

    logger.info(
        f"Dropped times during {len(transition_indices)} DRF restabilization period(s)"
    )


def mark_overflow_packets(
    goodtimes_ds: xr.Dataset,
    l1b_de: xr.Dataset,
    config_df: pd.DataFrame,
    cull_code: int = CullCode.LOOSE,
) -> None:
    """
    Remove times when DE packets overflow with qualified events.

    Filters out 8-spin periods where a Direct Event packet contains the maximum
    number of events (664) and the final event qualifies for a calibration product.
    When a packet is full and ends with a qualified event, additional events may
    have been lost, making the count data incomplete.

    Algorithm Document Reference:
        Section 2.3.2.2: Good Times Exclusions due to High Count Rate

    Background:
        Each DE packet can hold a maximum of 664 direct events. When a packet fills
        completely, any additional events that occur are lost. If the final event
        in a full packet has a coincidence type that is part of a defined calibration
        product, the packet is considered to have potentially lost science-quality
        events, and the entire 8-spin period should be excluded from analysis.

    Parameters
    ----------
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset to update with cull flags.
    l1b_de : xarray.Dataset
        L1B Direct Event data containing:
        - ccsds_index: Index mapping each event to its packet
        - coincidence_type: Coincidence type bitmap for each event
        - event_met: MET timestamp for each event
    config_df : pandas.DataFrame
        Calibration product configuration DataFrame with coincidence_type_values
        column containing tuples of valid coincidence type integers for each
        calibration product. Use CalibrationProductConfig.from_csv() to load.
    cull_code : int, optional
        Cull code to use for marking bad times (default: CullCode.LOOSE).

    Notes
    -----
    This function modifies goodtimes_ds in place by calling mark_bad_times()
    for MET timestamps with overflow packets containing qualified final events.

    The check for qualified events uses the coincidence_type_values from the
    calibration product configuration, which defines which coincidence types
    are considered valid for science analysis.
    """
    logger.info("Running mark_overflow_packets culling")

    ccsds_indices = l1b_de["ccsds_index"].values
    coincidence_types = l1b_de["coincidence_type"].values
    event_mets = l1b_de["event_met"].values

    if len(ccsds_indices) == 0:
        logger.info("No events in L1B DE data")
        return

    # Maximum number of DEs per packet
    max_des_per_packet = 664

    # Count events per packet using bincount
    # bincount[i] = number of events with ccsds_index == i
    packet_event_counts = np.bincount(ccsds_indices)

    # Find packets that are full (have exactly 664 events)
    full_packet_indices = np.nonzero(packet_event_counts == max_des_per_packet)[0]

    if len(full_packet_indices) == 0:
        logger.info("No full packets found")
        return

    # Use DEBUG level for per-packet logging if more than 10 full packets
    log_per_packet = logger.info if len(full_packet_indices) <= 10 else logger.debug

    # Build set of all valid coincidence types from calibration products
    all_valid_coin_types = set()
    for coin_types in config_df["coincidence_type_values"]:
        all_valid_coin_types.update(coin_types)

    # Find the last event index for each packet (vectorized)
    # We need to find, for each full packet, the index of its final event.
    # Since events within a packet appear consecutively in the array, the
    # "last" event for packet P is the event with the largest array index
    # where ccsds_indices == P.
    #
    # We use np.maximum.at to efficiently compute this:
    # - last_event_per_packet[P] will hold the max event index for packet P
    # - np.maximum.at updates last_event_per_packet[ccsds_indices[i]] with
    #   event_indices[i] if it's larger than the current value
    # - After processing all events, last_event_per_packet[P] contains the
    #   index of the last event belonging to packet P
    max_packet_idx = int(np.max(ccsds_indices))
    last_event_per_packet = np.full(max_packet_idx + 1, -1, dtype=np.intp)
    event_indices = np.arange(len(ccsds_indices))
    np.maximum.at(last_event_per_packet, ccsds_indices, event_indices)

    # Get the final event indices for full packets
    final_event_indices = last_event_per_packet[full_packet_indices]

    # Get coincidence types for final events
    final_coin_types = coincidence_types[final_event_indices]

    # Log each full packet
    for i, packet_idx in enumerate(full_packet_indices):
        log_per_packet(
            f"Packet {packet_idx} is full with final event "
            f"(coincidence_type={final_coin_types[i]})"
        )

    # Check which final events are qualified (in a calibration product)
    qualified_mask = np.isin(final_coin_types, list(all_valid_coin_types))

    # Get METs for qualified packets
    mets_to_cull = event_mets[final_event_indices[qualified_mask]]

    # Mark all identified times as bad (all spin bins)
    if len(mets_to_cull) > 0:
        goodtimes_ds.goodtimes.mark_bad_times(met=mets_to_cull, cull=cull_code)

    logger.info(
        f"Found {len(full_packet_indices)} full packet(s), "
        f"dropped {len(mets_to_cull)} 8-spin period(s) due to overflow packets"
    )


def _get_sweep_indices(esa_step: np.ndarray) -> np.ndarray:
    """
    Assign sweep indices to each MET based on ESA step transitions.

    A new sweep starts when ESA step transitions from high to low
    (e.g., 9 -> 1), detected using np.diff().

    Parameters
    ----------
    esa_step : numpy.ndarray
        ESA step values for each MET (epoch dimension).

    Returns
    -------
    sweep_indices : numpy.ndarray
        Sweep index for each MET. First sweep is index 0.
    """
    if len(esa_step) == 0:
        return np.array([], dtype=np.int32)

    # Find sweep boundaries where ESA step transitions from high to low
    esa_diff = np.diff(esa_step.astype(np.int32))
    # Negative diff indicates high-to-low transition (e.g., 9 -> 1 = -8)
    sweep_boundaries = esa_diff < 0

    # Create sweep indices using cumsum on boundaries
    # Prepend False so first MET is in sweep 0
    sweep_indices = (
        np.concatenate([[False], sweep_boundaries]).cumsum().astype(np.int32)
    )

    return sweep_indices


def _add_sweep_indices(l1b_de: xr.Dataset) -> xr.Dataset:
    """
    Add esa_sweep coordinate to the dataset based on ESA step transitions.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with esa_sweep coordinate added on epoch dimension.
    """
    sweep_indices = _get_sweep_indices(l1b_de["esa_step"].values)
    return l1b_de.assign_coords(esa_sweep=("epoch", sweep_indices))


def _compute_normalized_counts_per_sweep(
    l1b_de: xr.Dataset,
    tof_ab_limit_ns: int,
) -> xr.Dataset:
    """
    Compute normalized AB coincidence counts per ESA sweep and reshape dataset.

    This function:
    1. Computes normalized AB coincidence counts per sweep
    2. Removes all data associated with the event_met coordinate
    3. Reshapes the dataset so esa_sweep becomes a dimension (removing epoch)
    4. Returns the updated dataset with all epoch-based variables

    Parameters
    ----------
    l1b_de : xarray.Dataset
        L1B Direct Event dataset with esa_sweep coordinate on epoch dimension.
    tof_ab_limit_ns : int
        Maximum absolute value of tof_ab in nanoseconds.

    Returns
    -------
    xarray.Dataset
        Reshaped dataset with esa_sweep as a dimension containing:
        - normalized_count: normalized AB coincidence counts per sweep
        - All other variables from the input dataset (first value per sweep)
    """
    if "esa_sweep" not in l1b_de.coords:
        raise ValueError("Dataset must have esa_sweep coordinate")

    # Filter to valid AB coincidences
    tof_ab = l1b_de["tof_ab"]
    coincidence_type = l1b_de["coincidence_type"]
    ccsds_index = l1b_de["ccsds_index"]

    ab_coincidence_type = CoincidenceBitmap.detector_hit_str_to_int("AB")
    is_valid_ab = (coincidence_type == ab_coincidence_type) & (
        np.abs(tof_ab) <= tof_ab_limit_ns
    )

    # Map events to sweeps via ccsds_index -> esa_sweep
    event_epoch_idx = ccsds_index.values
    event_sweep_idx = l1b_de["esa_sweep"].values[event_epoch_idx]

    # Count valid AB events per sweep
    n_sweeps = int(l1b_de["esa_sweep"].max().values) + 1
    counts_per_sweep = np.zeros(n_sweeps, dtype=np.int64)
    np.add.at(counts_per_sweep, event_sweep_idx[is_valid_ab.values], 1)

    # Normalize by number of unique ESA steps
    n_unique_esa_steps = len(np.unique(l1b_de["esa_step"].values))
    normalized_counts = counts_per_sweep / n_unique_esa_steps

    # Remove all variables that depend on event_met dimension
    ds = l1b_de.drop_dims("event_met", errors="ignore")

    # Set esa_sweep and esa_step as a multi-index on epoch dimension
    ds = ds.set_index(epoch=["esa_sweep", "esa_step"])

    # Drop duplicates, keeping first occurrence of each (esa_sweep, esa_step) pair
    # This handles cases where multiple packets have the same esa_sweep and esa_step
    ds = ds.drop_duplicates(dim="epoch", keep="first")

    # Unstack to make esa_sweep and esa_step into separate dimensions
    # This creates a 2D array with dimensions (esa_sweep, esa_step)
    ds_reshaped = ds.unstack("epoch")

    # Add normalized_count as a new variable
    # It only has esa_sweep dimension (no esa_step variation within a sweep)
    ds_reshaped["normalized_count"] = xr.DataArray(
        normalized_counts,
        dims=["esa_sweep"],
        coords={"esa_sweep": np.arange(n_sweeps)},
    )

    return ds_reshaped


def mark_statistical_filter_0(
    goodtimes_ds: xr.Dataset,
    l1b_de_datasets: list[xr.Dataset],
    current_index: int,
    threshold_factor: float = 1.5,
    tof_ab_limit_ns: int = 15,
    cull_code: int = CullCode.LOOSE,
    min_pointings: int = 4,
) -> None:
    """
    Apply Statistical Filter 0 to detect drastic penetrating background changes.

    Statistical Filter 0 from Algorithm Document Section 2.3.2.3 detects when
    the penetrating background rate has changed drastically, compromising
    background subtraction accuracy. For each ESA sweep across all input
    Pointings, it computes the normalized AB coincidence count (total count
    divided by number of ESA steps). It then marks ESA sweeps in the current
    Pointing where the normalized count exceeds 150% of the median.

    Parameters
    ----------
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset for the current Pointing to update.
    l1b_de_datasets : list[xarray.Dataset]
        List of L1B DE datasets for surrounding Pointings. Typically includes
        current plus preceding and following Pointings
        (e.g., [P-3, P-2, P-1, P(current), P+1, P+2, P+3]).
    current_index : int
        Index of the current Pointing in l1b_de_datasets.
    threshold_factor : float, optional
        Multiplier for median comparison. Default is 1.5 (150% of median).
    tof_ab_limit_ns : int, optional
        Maximum |tof_ab| in nanoseconds for AB coincidences. Default is 15.
    cull_code : int, optional
        Cull code to use for marking bad times. Default is CullCode.LOOSE.
    min_pointings : int, optional
        Minimum number of Pointings required. Default is 4.

    Raises
    ------
    ValueError
        If current_index is out of range or if fewer than min_pointings
        datasets are provided.

    Notes
    -----
    This function modifies goodtimes_ds in place. Only ESA sweeps in the
    current Pointing where the normalized count exceeds `threshold_factor *
    median` are marked as bad. Other sweeps remain unaffected.

    Algorithm:
    1. For each complete ESA sweep across all Pointings, count AB coincidences
       where |tof_ab| <= 15ns and divide by number of ESA steps
    2. Calculate median of all normalized sweep counts
    3. For each sweep in current Pointing, mark all METs in that sweep as bad
       if normalized count > threshold_factor * median
    """
    logger.info("Running mark_statistical_filter_0 culling")

    # Validate current_index is in range
    if current_index < 0 or current_index >= len(l1b_de_datasets):
        raise ValueError(
            f"current_index {current_index} out of range for list of "
            f"length {len(l1b_de_datasets)}"
        )

    # Validate that we have the minimum number of datasets
    if len(l1b_de_datasets) < min_pointings:
        raise ValueError(
            f"At least {min_pointings} valid Pointings required, "
            f"got {len(l1b_de_datasets)}"
        )

    # Add esa_sweep coordinate, reshape, and compute normalized_count for each dataset
    all_normalized_counts: list[np.ndarray] = []
    reshaped_datasets: dict[int, xr.Dataset] = {}

    for i, l1b_de in enumerate(l1b_de_datasets):
        # Add esa_sweep coordinate
        l1b_de_with_sweep = _add_sweep_indices(l1b_de)

        # Compute normalized counts and reshape dataset. This removes epoch
        # dimension, adds esa_sweep dimension, and includes normalized_count.
        reshaped_ds = _compute_normalized_counts_per_sweep(
            l1b_de_with_sweep, tof_ab_limit_ns
        )

        # Store reshaped dataset and normalized counts
        reshaped_datasets[i] = reshaped_ds
        all_normalized_counts.append(reshaped_ds["normalized_count"].values)

        offset = i - current_index
        logger.debug(
            f"Pointing {offset:+d}: "
            f"{len(reshaped_ds['normalized_count'])} complete ESA sweeps"
        )

    current_ds = reshaped_datasets[current_index]

    # Calculate median from all sweep counts
    all_counts = np.concatenate(all_normalized_counts)
    median_count = float(np.median(all_counts))
    threshold = median_count * threshold_factor

    logger.info(
        f"Statistical Filter 0: median={median_count:.2f}, "
        f"threshold={threshold:.2f} ({len(all_counts)} sweeps)"
    )

    # Find and mark bad sweeps in current dataset
    bad_sweep_mask = current_ds["normalized_count"] > threshold
    n_bad_sweeps = int(bad_sweep_mask.sum())

    # Get MET time ranges for bad sweeps using xarray boolean indexing
    # Select only the bad sweeps using the mask
    bad_sweeps_ds = current_ds.isel(esa_sweep=bad_sweep_mask)

    # For each bad sweep, mark the time range from first to last ccsds_met
    for sweep_idx in range(len(bad_sweeps_ds["esa_sweep"])):
        # Get all ccsds_met values for this sweep across all esa_steps
        sweep_mets = bad_sweeps_ds["ccsds_met"].isel(esa_sweep=sweep_idx).values

        # Get min and max MET values, ignoring NaNs
        met_start: float = float(np.nanmin(sweep_mets))
        met_end: float = float(np.nanmax(sweep_mets))

        # Mark the entire time range for this sweep as bad
        goodtimes_ds.goodtimes.mark_bad_times(
            met=(met_start, met_end), bins=None, cull=cull_code
        )

    if n_bad_sweeps > 0:
        logger.info(
            f"Statistical Filter 0: Marked {n_bad_sweeps}/"
            f"{len(current_ds['normalized_count'])} ESA sweeps as bad"
        )
    else:
        logger.info("No bad ESA sweeps identified by Statistical Filter 0")
