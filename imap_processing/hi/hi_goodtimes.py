"""IMAP-HI Goodtimes processing module."""

import logging
import re
from enum import IntEnum
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.hi.utils import parse_sensor_number

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


def create_goodtimes_dataset(l1a_de: xr.Dataset) -> xr.Dataset:
    """
    Create goodtimes dataset from L1A Direct Event data.

    Initializes all times and spin bins as good (cull_flags=0). The goodtimes
    dataset is created with one entry per unique MET timestamp found in the
    L1A DE data. Culling functions (e.g., drop_partial_packets) should be
    called after creation to identify and flag bad times.

    Parameters
    ----------
    l1a_de : xarray.Dataset
        L1A direct event data for this pointing. Used to extract MET timestamps
        for each 8-spin interval.

    Returns
    -------
    xarray.Dataset
        Initialized goodtimes dataset with cull_flags set to 0 (all good).
        Access goodtimes methods via the .goodtimes accessor
        (e.g., dataset.goodtimes.remove_times()).
    """
    logger.info("Creating Goodtimes from L1A Direct Event data")

    # Extract MET times from packet metadata
    # Each MET represents one 8-spin histogram packet interval
    # Format: seconds + subseconds/1000
    met_all = (
        l1a_de["meta_seconds"].astype(float)
        + l1a_de["meta_subseconds"].astype(float) / 1000
    )
    logger.debug(f"Extracted {len(met_all)} total MET entries from L1A DE data")

    # Find unique MET values and indices of first occurrences
    unique_mets, first_indices = np.unique(met_all.values, return_index=True)
    logger.info(f"Found {len(unique_mets)} unique MET values")

    # Extract data for unique METs (use first occurrence of each)
    met = met_all.isel(epoch=first_indices)
    esa_step = l1a_de["esa_step"].isel(epoch=first_indices)

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
    sensor_number = parse_sensor_number(l1a_de.attrs["Logical_source"])
    match = re.match(r"repoint(?P<pointing_num>\d{5})", l1a_de.attrs["Repointing"])
    if not match:
        raise ValueError(
            f"Unable to parse pointing number from l1a_de Repointing "
            f"attribute: {l1a_de.attrs['Repointing']}"
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
    >>> gt_dataset = create_goodtimes_dataset(l1a_de)
    >>> gt_dataset.goodtimes.remove_times(met=1000.5, cull=CullCode.LOOSE)
    >>> intervals = gt_dataset.goodtimes.get_good_intervals()
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialize the accessor with an xarray Dataset."""
        self._obj = xarray_obj

    def remove_times(
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
        >>> goodtimes.remove_times(met=1000.5, bins=None, cull=CullCode.LOOSE)

        >>> # Flag spin bins 0-10 for MET=1000.5
        >>> goodtimes.remove_times(met=1000.5, bins=np.arange(11), cull=CullCode.LOOSE)

        >>> # Flag time range around a repoint (240s before/after)
        >>> repoint_time = 1000.0
        >>> goodtimes.remove_times(
        ...     met=(repoint_time - 240, repoint_time + 240),
        ...     cull=CullCode.LOOSE
        ... )

        >>> # Flag multiple specific METs, all bins
        >>> goodtimes.remove_times(
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


def drop_partial_packets(
    goodtimes_ds: xr.Dataset,
    l1a_de: xr.Dataset,
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
    l1a_de : xarray.Dataset
        L1A Direct Event data containing DE packets with last_spin_num field.
    cull_code : int, optional
        Cull code to use for marking bad times (default: CullCode.LOOSE).

    Notes
    -----
    This function modifies goodtimes_ds in place by calling remove_times()
    for MET timestamps with incomplete spin coverage.
    """
    logger.info("Running drop_partial_packets culling")

    met_values = goodtimes_ds.coords["met"].values

    # Calculate DE packet MET times
    de_met = (
        l1a_de["meta_seconds"].astype(float)
        + l1a_de["meta_subseconds"].astype(float) / 1000
    )

    # Assign each DE packet to nearest goodtimes MET using searchsorted
    # This maps each DE packet to a MET index
    met_indices = np.searchsorted(met_values, de_met.values, side="right") - 1

    # Clip to valid range
    met_indices = np.clip(met_indices, 0, len(met_values) - 1)

    # Calculate actual distance to assigned MET
    time_slop = 10.0  # seconds tolerance
    distances = np.abs(de_met.values - met_values[met_indices])
    valid_assignment = distances <= time_slop

    # Create a new coordinate in l1a_de for grouping
    l1a_de_with_group = l1a_de.assign_coords(met_group=("epoch", met_indices))

    # Only keep packets with valid time assignment
    l1a_de_valid = l1a_de_with_group.isel(epoch=valid_assignment)

    # Valid pattern bitmasks
    valid_pattern_1 = 0b10001000  # bits 3,7: every 4th spin (last_spin_num 4,8)
    valid_pattern_2 = 0b10101010  # bits 1,3,5,7: every 2nd spin (2,4,6,8)
    valid_pattern_3 = 0b11111111  # bits 0-7: every spin (1-8)
    valid_patterns = [valid_pattern_1, valid_pattern_2, valid_pattern_3]

    # Group by MET and validate each group
    bad_mets = []

    for met_idx, group in l1a_de_valid.groupby("met_group"):
        met_time = met_values[met_idx]

        # Check for invalid spins flag
        if np.any(group["spin_invalids"].values != 0):
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
        goodtimes_ds.goodtimes.remove_times(met=np.array(bad_mets), cull=cull_code)

    logger.info(f"Dropped {len(bad_mets)} partial DE packet(s)")


def drop_drf_times(
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
        Housekeeping data containing DRF status in scstatus field.
    cull_code : int, optional
        Cull code to use for marking bad times (default: CullCode.LOOSE).

    Notes
    -----
    This function modifies goodtimes_ds in place. If no housekeeping data is
    available, a warning is logged but no times are removed.
    """
    logger.info("Running drop_drf_times culling")

    if len(hk.epoch) == 0:
        logger.warning("No NHK loaded to check for DRF times")
        return

    # Extract DRF status bit from housekeeping
    # Assuming scstatus field exists with DRF bit
    if "scstatus" not in hk:
        logger.warning("No scstatus field in housekeeping data")
        return

    # Get HK times and DRF status
    hk_met = (
        hk["meta_seconds"].astype(float) + hk["meta_subseconds"].astype(float) / 1000
    )
    # DRF status bit - need to check the bit position
    # Assuming STAT_DRF is a specific bit in scstatus
    stat_drf_bit = 0x01  # TODO: Verify actual bit position
    drf_status = (hk["scstatus"].values & stat_drf_bit) != 0

    n_dropped = 0
    drf_was_set = False

    # Scan for DRF transitions from 1->0 (end of restabilization)
    for i in range(len(hk_met)):
        if drf_status[i]:
            drf_was_set = True
        elif drf_was_set:
            # Just transitioned from DRF=1 to DRF=0
            # Remove all times within 30 minutes prior
            drf_end_time = hk_met.values[i]
            window_start = drf_end_time - 30 * 60  # 30 minutes before

            # Remove times in this window
            met_values = goodtimes_ds.coords["met"].values
            in_window = (met_values >= window_start) & (met_values < drf_end_time)

            for met_time in met_values[in_window]:
                goodtimes_ds.goodtimes.remove_times(met=met_time, cull=cull_code)
                n_dropped += 1

            drf_was_set = False

    logger.info(f"Dropped {n_dropped} packet(s) during DRF restabilization")


def drop_bad_voltages(
    goodtimes_ds: xr.Dataset,
    hk: xr.Dataset,
    cull_code: int = CullCode.LOOSE,
) -> None:
    """
    Remove times when detector voltages are out of range.

    Filters out data when detector high voltages are outside their nominal science
    ranges. Incorrect voltages indicate the instrument is in a non-science mode
    (e.g., gain testing, calibration, or transitioning between modes) and the data
    are not suitable for scientific analysis.

    Algorithm Document Reference:
        Section 2.2.4: Housekeeping checks for instrument operating state
        Section 2.2.7: Bad times during non-science operations

    Background:
        The IMAP-Hi detector system uses several high voltage power supplies.
        During science operations (HVSCI mode, opmode=4), these voltages must be
        within tight tolerances. Voltages outside tolerance indicate gain testing,
        instrument ramping, or potential anomalies.

    Voltage Tolerances (nominal ± tolerance):
        - DeflP: 6300 ± 700 V
        - DeflN: -6300 ± 700 V
        - UcanV: -8000 ± 50 V
        - MCPF: -3000 ± 25 V
        - CEMF: -4500 ± 50 V
        - CEMAV: -2350 ± 100 V (head-dependent)
        - CEMBV: -2350 ± 100 V (head-dependent)
        - MCPV: -2125 ± 50 V

    Parameters
    ----------
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset to update with cull flags.
    hk : xarray.Dataset
        Housekeeping data containing voltage telemetry.
    cull_code : int, optional
        Cull code to use for marking bad times (default: CullCode.LOOSE).

    Notes
    -----
    Only removes times when in HVSCI mode (opmode=4). Out-of-range voltages
    in HVENG (opmode=3) or HVSTANDBY (opmode=5) are expected and acceptable.
    """
    logger.info("Running drop_bad_voltages culling")

    if len(hk.epoch) == 0:
        logger.warning("No NHK loaded to check for unusual HV times")
        return

    # Get HK times
    hk_met = (
        hk["meta_seconds"].astype(float) + hk["meta_subseconds"].astype(float) / 1000
    )

    # Define voltage limits (nominal, tolerance)
    voltage_checks = [
        ("defl_p", 6300.0, 700.0),
        ("defl_n", -6300.0, 700.0),
        ("ucan_v", -8000.0, 50.0),
        ("mcp_f", -3000.0, 25.0),
        ("cem_f", -4500.0, 50.0),
        ("cem_av", -2350.0, 100.0),
        ("cem_bv", -2350.0, 100.0),
        ("mcp_v", -2125.0, 50.0),
    ]

    n_dropped = 0

    # Check each HK packet for voltage anomalies
    for i in range(len(hk_met)):
        # Check if any voltage is out of range
        voltage_bad = False
        for field_name, nominal, tolerance in voltage_checks:
            # Convert field name to actual HK field name (may need adjustment)
            # Assuming lowercase field names in HK dataset
            if field_name in hk:
                voltage = hk[field_name].values[i]
                if np.abs(voltage - nominal) > tolerance:
                    voltage_bad = True
                    break

        if voltage_bad:
            # Only cull if in HVSCI mode (opmode=4)
            if "opmode" in hk:
                opmode = hk["opmode"].values[i]
                if opmode == 4:  # HVSCI mode
                    # Find corresponding MET time in goodtimes
                    hk_time = hk_met.values[i]
                    met_values = goodtimes_ds.coords["met"].values

                    # Find MET times near this HK time
                    time_slop = 20 * 60  # 20 minute window
                    near_time = np.abs(met_values - hk_time) <= time_slop

                    for met_time in met_values[near_time]:
                        goodtimes_ds.goodtimes.remove_times(
                            met=met_time, cull=cull_code
                        )
                        n_dropped += 1

    logger.info(f"Dropped {n_dropped} packet(s) during unusual HV times")


def drop_bad_tdc_diagfee(  # noqa: PLR0912
    goodtimes_ds: xr.Dataset,
    diagfee: xr.Dataset,
    check_tdc1: bool = True,
    check_tdc2: bool = True,
    check_tdc3: bool = True,
    cull_code: int = CullCode.LOOSE,
) -> None:
    """
    Remove times with failed TDC calibration (DIAG_FEE method).

    Filters out data when Time-to-Digital Converter (TDC) calibration has failed,
    as indicated by the DIAG_FEE diagnostic packet. Failed TDC calibration results
    in incorrect time-of-flight measurements, which corrupts the energy and mass
    determination for detected particles.

    Algorithm Document Reference:
        Section 2.2.4: Housekeeping checks for instrument calibration state

    Background:
        The IMAP-Hi sensor uses three TDC channels to measure time-of-flight between
        detector pairs:
          - TDC1: Measures time between detector A and detector B (t_AB)
          - TDC2: Measures time between detector A and detector C (t_AC)
          - TDC3: Measures time between detector C and coincidence detector (t_CC)

        The DIAG_FEE packet contains calibration status fields (cal1stat, cal2stat,
        cal3stat) with bit 1 indicating successful calibration (bit=1 => good).
        This is the simpler of two TDC validation methods.

    Parameters
    ----------
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset to update with cull flags.
    diagfee : xarray.Dataset
        DIAG_FEE diagnostic data containing TDC calibration status.
    check_tdc1 : bool, optional
        If True, remove times when TDC1 calibration failed (default: True).
    check_tdc2 : bool, optional
        If True, remove times when TDC2 calibration failed (default: True).
    check_tdc3 : bool, optional
        If True, remove times when TDC3 calibration failed (default: True).
    cull_code : int, optional
        Cull code to use for marking bad times (default: CullCode.LOOSE).

    Notes
    -----
    Two DIAG_FEE packets are generated when entering HVSCI mode. This function
    skips the first packet if two packets appear within 10 seconds.
    """
    logger.info("Running drop_bad_tdc_diagfee culling")

    if len(diagfee.epoch) < 2:
        logger.warning("No DIAG_FEE data to use for selecting good times")
        return

    # Get DIAG_FEE times
    df_met = (
        diagfee["meta_seconds"].astype(float)
        + diagfee["meta_subseconds"].astype(float) / 1000
    )

    # Track calibration state for each TDC
    bad_tdc1 = False
    bad_tdc2 = False
    bad_tdc3 = False

    n_times_removed = 0

    # Scan through DIAG_FEE packets chronologically
    for i in range(len(df_met)):
        # Skip duplicate DIAG_FEE packets (within 10 seconds of next)
        if i < len(df_met) - 1:
            if df_met.values[i] + 10 > df_met.values[i + 1]:
                continue

        # Check TDC calibration status (bit 1: 1=good, 0=bad)
        any_tdc_failed = False

        # Check TDC1
        if "cal1stat" in diagfee:
            cal1_good = (diagfee["cal1stat"].values[i] & 2) != 0
            if not cal1_good:
                if not bad_tdc1:
                    logger.debug(f"TDC1 cal failed as of {df_met.values[i]:.1f}")
                bad_tdc1 = True
                if check_tdc1:
                    any_tdc_failed = True
            else:
                if bad_tdc1:
                    logger.debug(f"TDC1 cal GOOD as of {df_met.values[i]:.1f}")
                bad_tdc1 = False

        # Check TDC2
        if "cal2stat" in diagfee:
            cal2_good = (diagfee["cal2stat"].values[i] & 2) != 0
            if not cal2_good:
                if not bad_tdc2:
                    logger.debug(f"TDC2 cal failed as of {df_met.values[i]:.1f}")
                bad_tdc2 = True
                if check_tdc2:
                    any_tdc_failed = True
            else:
                if bad_tdc2:
                    logger.debug(f"TDC2 cal GOOD as of {df_met.values[i]:.1f}")
                bad_tdc2 = False

        # Check TDC3
        if "cal3stat" in diagfee:
            cal3_good = (diagfee["cal3stat"].values[i] & 2) != 0
            if not cal3_good:
                if not bad_tdc3:
                    logger.debug(f"TDC3 cal failed as of {df_met.values[i]:.1f}")
                bad_tdc3 = True
                if check_tdc3:
                    any_tdc_failed = True
            else:
                if bad_tdc3:
                    logger.debug(f"TDC3 cal GOOD as of {df_met.values[i]:.1f}")
                bad_tdc3 = False

        # If any requested TDC failed, remove times until next DIAG_FEE packet
        if any_tdc_failed:
            df_time = df_met.values[i]
            if i >= len(df_met) - 1:
                # Last DIAG_FEE packet - remove all times after this
                next_df_time = np.inf
            else:
                next_df_time = df_met.values[i + 1]

            met_values = goodtimes_ds.coords["met"].values
            in_window = (met_values >= df_time) & (met_values < next_df_time)

            for met_time in met_values[in_window]:
                goodtimes_ds.goodtimes.remove_times(met=met_time, cull=cull_code)
                n_times_removed += 1

    logger.info(f"Dropped {n_times_removed} time(s) due to bad TDC calibration")


def drop_overflow_packets(
    goodtimes_ds: xr.Dataset,
    l1a_de: xr.Dataset,
    howfar: int = 1,
    cull_code: int = CullCode.LOOSE,
) -> None:
    """
    Filter out times when DE packet buffers overflowed.

    Removes times when the Direct Event packet buffer reached capacity (664 events),
    indicating potential data loss. When the buffer fills, lower-priority events
    may be discarded, compromising the scientific integrity of the data.

    Algorithm Document Reference:
        Section 2.2.4: Housekeeping checks including buffer overflow detection

    Background:
        DE packets can hold up to 664 events and use a priority system:
          - Gold: Highest quality coincidence events (triple/quadruple coincidences)
          - Silver: Medium quality events (double coincidences)
          - Bronze: Lower quality events (single detector hits)
          - Other: Background or calibration events

        When the buffer fills (nde == 664), the last event's type indicates what
        category filled the buffer. If high-priority events were lost, data quality
        is compromised.

    Parameters
    ----------
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset to update with cull flags.
    l1a_de : xarray.Dataset
        L1A Direct Event data containing DE packets.
    howfar : int, optional
        Aggressiveness level for overflow culling:
          1: Remove only if gold events overflowed (most conservative, default)
          2: Remove if silver or gold overflowed
          3: Remove if bronze, silver, or gold overflowed
          4+: Remove if any overflow occurred (most aggressive).
    cull_code : int, optional
        Cull code to use for marking bad times (default: CullCode.LOOSE).

    Notes
    -----
    This function requires a method to classify the last event type (gold/silver/
    bronze/other). The classification logic should be implemented based on the
    coincidence type and detector hit pattern.
    """
    logger.info(f"Running drop_overflow_packets culling (howfar={howfar})")

    # Maximum DE packet capacity
    max_de_capacity = 664

    # Counters for different overflow types
    gold_overflow = 0
    silver_overflow = 0
    bronze_overflow = 0
    other_overflow = 0
    total_overflow = 0

    # Get DE packet MET times
    de_met = (
        l1a_de["meta_seconds"].astype(float)
        + l1a_de["meta_subseconds"].astype(float) / 1000
    )

    # Check if we have nde field (number of direct events)
    if "nde" not in l1a_de:
        logger.warning("No 'nde' field in L1A DE data, cannot check for overflow")
        return

    n_removed = 0

    # Check each DE packet for overflow
    for i in range(len(de_met)):
        nde = l1a_de["nde"].values[i]

        if nde == max_de_capacity:
            # Buffer is full - classify the last event
            # TODO: Implement gsbo_detype classification
            # For now, assume we can check coincidence_type or similar field
            event_type = _classify_de_event_type(l1a_de, i)

            should_cull = False

            if event_type == 1:  # Gold
                gold_overflow += 1
                total_overflow += 1
                should_cull = True  # Always cull gold overflow
            elif event_type == 2:  # Silver
                silver_overflow += 1
                total_overflow += 1
                should_cull = howfar >= 2
            elif event_type == 3:  # Bronze
                bronze_overflow += 1
                total_overflow += 1
                should_cull = howfar >= 3
            elif event_type == 4:  # Other
                other_overflow += 1
                total_overflow += 1
                should_cull = howfar >= 4

            if should_cull:
                # Find corresponding MET time
                de_time = de_met.values[i]
                met_values = goodtimes_ds.coords["met"].values

                # Find closest MET time
                time_slop = 10.0
                close_enough = np.abs(met_values - de_time) <= time_slop

                for met_time in met_values[close_enough]:
                    goodtimes_ds.goodtimes.remove_times(met=met_time, cull=cull_code)
                    n_removed += 1

    if total_overflow > 0:
        logger.info(
            f"{total_overflow} DE packets were full/overfull: "
            f"{gold_overflow}/{silver_overflow}/{bronze_overflow}/{other_overflow} "
            f"lacking gold/silver/bronze/other and removed {n_removed} from good times"
        )


def _classify_de_event_type(l1a_de: xr.Dataset, packet_idx: int) -> int:
    """
    Classify the last event in a DE packet as gold/silver/bronze/other.

    This is a placeholder for the gsbo_detype function from the C code.

    Parameters
    ----------
    l1a_de : xarray.Dataset
        L1A DE dataset.
    packet_idx : int
        Index of the packet to check.

    Returns
    -------
    int
        Event type: 1=gold, 2=silver, 3=bronze, 4=other, -1=unknown.
    """
    # TODO: Implement actual classification logic based on:
    # - Coincidence type (AB, AC, BC, CC)
    # - Number of detectors hit
    # - Quality criteria from MEMDMP if available

    # Placeholder implementation
    logger.warning("Event type classification not yet implemented, returning -1")
    return -1


def drop_toobusy(
    goodtimes_ds: xr.Dataset,
    l1a_de: xr.Dataset,
    n_counts: int,
    m_ticks: int,
    esa_mask: int = 0xFFFF,
    cull_code: int = CullCode.LOOSE,
) -> None:
    """
    Remove times with excessive count rates in short windows.

    Filters out time periods where the instrument sees abnormally high count rates
    in short time windows. Such bursts of counts often indicate contamination from
    sources other than the heliospheric ENA signal being measured.

    Algorithm Document Reference:
        Section 2.2.7: Bad times from anomalous count rates

    Background:
        The IMAP-Hi sensor should see relatively steady ENA count rates from the
        heliosphere. Short bursts of many counts (e.g., >n counts in m ticks) are
        non-physical for heliospheric ENAs and indicate contamination. This is
        particularly problematic in certain ESA energy steps (notably ESA 9).

    Parameters
    ----------
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset to update with cull flags.
    l1a_de : xarray.Dataset
        L1A Direct Event data containing event times and bins.
    n_counts : int
        Maximum number of qualified counts allowed in the time window.
    m_ticks : int
        Size of sliding time window in ticks (should be < 1 spacecraft rotation).
    esa_mask : int, optional
        Bitmask of ESA steps to apply filter to (bit k = ESA step k+1).
        Default 0xFFFF applies to all ESA steps. Example: 0x0100 = only ESA 9.
    cull_code : int, optional
        Cull code to use for marking bad times (default: CullCode.LOOSE).

    Notes
    -----
    This function requires qualified event selection criteria (isqualde).
    Currently uses a placeholder qualification check. The caller must ensure
    m_ticks is less than one spacecraft rotation period.
    """
    logger.info(
        f"Running drop_toobusy culling (n={n_counts}, m={m_ticks} ticks, "
        f"esa_mask=0x{esa_mask:04x})"
    )

    # Get DE packet data
    de_met = (
        l1a_de["meta_seconds"].astype(float)
        + l1a_de["meta_subseconds"].astype(float) / 1000
    )

    # Check required fields
    if "esa_step" not in l1a_de:
        logger.warning("No esa_step in L1A DE data")
        return

    # Track bins removed per ESA step
    bins_removed = np.zeros(16, dtype=int)  # MAXE ESA steps

    met_values = goodtimes_ds.coords["met"].values
    time_slop = 10.0

    # Process each MET time
    for met_idx, met_time in enumerate(met_values):
        # Find DE packets at this time
        at_time = np.abs(de_met.values - met_time) <= time_slop

        if not np.any(at_time):
            continue

        # Get ESA step for this time (from goodtimes or DE data)
        esa_step = goodtimes_ds["esa_step"].values[met_idx]

        # Check if we should process this ESA step
        if not (esa_mask & (1 << (esa_step - 1))):
            continue

        # Collect events for this ESA step at this time
        # This is a simplified version - full implementation needs:
        # 1. Event timetags within each packet
        # 2. Spin bin for each event
        # 3. Qualification criteria (isqualde)

        # TODO: Implement full sliding window analysis
        # For now, just check if total number of events exceeds threshold
        total_events = 0
        for i in np.where(at_time)[0]:
            if "nde" in l1a_de:
                total_events += l1a_de["nde"].values[i]

        # Simple threshold check (placeholder for sliding window)
        if total_events > n_counts:
            # Mark all bins at this time as bad for this ESA step
            goodtimes_ds.goodtimes.remove_times(met=met_time, cull=cull_code)
            bins_removed[esa_step - 1] += 90  # All bins

    logger.info("Overly bright short sequences: angle bins removed:")
    for i, count in enumerate(bins_removed):
        if count > 0:
            logger.info(f"  ESA {i + 1}: {count} bins")

    logger.warning(
        "drop_toobusy is using simplified logic. Full sliding window "
        "analysis needs implementation."
    )
