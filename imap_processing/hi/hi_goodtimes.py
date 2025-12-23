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

    Initializes all times and spin bins as good (cull_flags=0) for complete
    8-spin periods. Since we receive one packet every 4 spins but only record
    MET every 8 spins, we expect MET values to appear in pairs. Only MET values
    that appear as duplicates (pairs) are included, as single occurrences indicate
    incomplete 8-spin periods.

    Parameters
    ----------
    l1a_de : xarray.Dataset
        L1A direct event data for this pointing. Used to extract MET timestamps
        for each 8-spin interval.

    Returns
    -------
    xarray.Dataset
        Initialized goodtimes dataset with cull_flags set to 0 (all good) for
        complete 8-spin periods only. Access goodtimes methods via the
        .goodtimes accessor (e.g., dataset.goodtimes.remove_times()).
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

    # Find unique MET values, their counts, and indices of first occurrences
    unique_mets, first_indices, counts = np.unique(
        met_all.values, return_index=True, return_counts=True
    )
    logger.debug(f"Found {len(unique_mets)} unique MET values")

    # Keep only MET values that appear as pairs (count == 2)
    paired_mask = counts == 2
    first_occurrence_indices = first_indices[paired_mask]

    n_paired = int(np.sum(paired_mask))
    n_unpaired = len(unique_mets) - n_paired
    logger.info(
        f"Filtered to {n_paired} complete 8-spin periods "
        f"(excluded {n_unpaired} incomplete periods)"
    )

    # Extract data for paired METs only
    met = met_all.isel(epoch=first_occurrence_indices)
    esa_step = l1a_de["esa_step"].isel(epoch=first_occurrence_indices)

    # Create coordinates
    coords = {
        "met": met.values,
        "spin_bin": np.arange(90),
    }

    # Create data variables
    # Initialize cull_flags - all good (0) by default
    # Shape: (n_met_timestamps, 90 spin_bins)
    # Per alg doc Section 2.2.4: 90-element arrays, one per histogram packet
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
        valid_met_range = (met_values[0], met_values[-1] + np.diff(met_values[-2:])[0])
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
