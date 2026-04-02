"""IMAP-HI Goodtimes processing module."""

import logging
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import convolve1d

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hi.utils import (
    CalibrationProductConfig,
    CoincidenceBitmap,
    HiConstants,
    compute_qualified_event_mask,
    parse_sensor_number,
)
from imap_processing.quality_flags import ImapHiL1bDeFlags
from imap_processing.spice.repoint import get_repoint_data
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)

# Structured dtype for good time intervals
INTERVAL_DTYPE: np.dtype = np.dtype(
    [
        ("met_start", np.float64),
        ("met_end", np.float64),
        ("spin_bin_low", np.uint8),
        ("spin_bin_high", np.uint8),
        ("n_bins", np.uint8),
        ("esa_step_mask", np.uint16),  # Bitmask for ESA steps 1-10 (bit i = step i+1)
        ("cull_value", np.uint8),
    ]
)


class CullCode(IntEnum):
    """Cull reason codes for good/bad time classification (bit flags)."""

    GOOD = 0
    INCOMPLETE_SPIN = 1 << 0  # 1
    DRF = 1 << 1  # 2
    BAD_TDC_CAL = 1 << 2  # 4
    OVERFLOW = 1 << 3  # 8
    STAT_FILTER_0 = 1 << 4  # 16
    STAT_FILTER_1 = 1 << 5  # 32
    STAT_FILTER_2 = 1 << 6  # 64


def hi_goodtimes(
    current_repointing: str,
    l1b_de_datasets: list[xr.Dataset],
    l1b_hk: xr.Dataset,
    l1a_diagfee: xr.Dataset,
    cal_product_config_path: Path,
) -> list[xr.Dataset]:
    """
    Generate goodtimes dataset for IMAP-Hi L1B processing.

    This is the top-level function that orchestrates all goodtimes culling
    operations for a single pointing. It applies the following filters in order:

    1. mark_incomplete_spin_sets - Remove incomplete 8-spin histogram periods
    2. mark_drf_times - Remove times during spacecraft drift restabilization
    3. mark_bad_tdc_cal - Remove times with failed TDC calibration
    4. mark_overflow_packets - Remove times when DE packets overflow
    5. mark_statistical_filter_0 - Detect drastic penetrating background changes
    6. mark_statistical_filter_1 - Detect isotropic count rate increases
    7. mark_statistical_filter_2 - Detect short-lived event pulses

    Parameters
    ----------
    current_repointing : str
        Repointing identifier for the current pointing (e.g., "repoint00001").
        Used to identify which dataset in l1b_de_datasets is the current one.
    l1b_de_datasets : list[xr.Dataset]
        L1B DE datasets for surrounding pointings. Typically includes
        current plus 3 preceding and 3 following pointings (7 total).
        Statistical filters 0 and 1 use all datasets; other filters use
        only the current pointing.
    l1b_hk : xr.Dataset
        L1B housekeeping dataset containing DRF status.
    l1a_diagfee : xr.Dataset
        L1A DIAG_FEE dataset containing TDC calibration status.
    cal_product_config_path : Path
        Path to calibration product configuration CSV file.

    Returns
    -------
    list[xr.Dataset]
        List containing the goodtimes dataset ready for CDF writing,
        or an empty list if processing cannot proceed yet.

    Notes
    -----
    See IMAP-Hi Algorithm Document Sections 2.2.4 and 2.3.2 for details
    on each culling algorithm.

    Processing requires that repointing + 3 has occurred (so that statistical
    filters can use surrounding pointings). Due to challenges with dependency
    management in the batch starter, it was decided to design the Hi goodtimes
    to set the L1B DE dependencies as not required and handle the final logic for
    checking L1B DE dependencies in this function. If repointing + 3 has not yet
    completed, an empty list is returned. If repointing + 3 has occurred but
    not all 7 DE files are available, all times are marked as bad.
    """
    logger.info("Starting Hi goodtimes processing")

    # Parse the current repoint ID and check if we can process yet
    current_repoint_id = int(current_repointing.replace("repoint", ""))
    future_repoint_id = current_repoint_id + 3

    # Check if the future repointing has finished by checking that the next
    # repoint is in the repoint dataframe.
    repoint_df = get_repoint_data()
    required_repoints_complete = (
        future_repoint_id + 1 in repoint_df["repoint_id"].values
    )

    if not required_repoints_complete:
        raise ValueError(
            f"Goodtimes cannot yet be processed for {current_repointing}: "
            f"repoint{future_repoint_id:05d} has not yet been completed "
            f"according to the repoint table."
        )

    # Find the current pointing index in the datasets
    current_index = _find_current_pointing_index(l1b_de_datasets, current_repointing)
    current_l1b_de = l1b_de_datasets[current_index]

    # Create the goodtimes dataset from the current pointing
    goodtimes_ds = create_goodtimes_dataset(current_l1b_de)

    # Check if we have the full set of 7 DE files for nominal processing
    if len(l1b_de_datasets) == 7:
        _apply_goodtimes_filters(
            goodtimes_ds,
            l1b_de_datasets,
            current_index,
            l1b_hk,
            l1a_diagfee,
            cal_product_config_path,
        )
    else:
        # Incomplete DE file set - mark all times as bad
        logger.warning(
            f"Incomplete DE file set for {current_repointing}: "
            f"expected 7 files, got {len(l1b_de_datasets)}. "
            "Marking all times as bad."
        )
        goodtimes_ds["cull_flags"][:, :] = CullCode.INCOMPLETE_SPIN

    # Log final statistics
    stats = goodtimes_ds.goodtimes.get_cull_statistics()
    logger.info(
        f"Final statistics: {stats['good_bins']}/{stats['total_bins']} good "
        f"({stats['fraction_good'] * 100:.1f}%)"
    )
    if stats["cull_code_counts"]:
        logger.info(f"Cull code counts: {stats['cull_code_counts']}")

    # Finalize dataset for CDF output
    logger.info("Finalizing goodtimes dataset for CDF output")
    cdf_ready_ds = goodtimes_ds.goodtimes.finalize_dataset()

    logger.info("Hi goodtimes processing complete")
    return [cdf_ready_ds]


def _find_current_pointing_index(
    l1b_de_datasets: list[xr.Dataset],
    current_repointing: str,
) -> int:
    """
    Find the index of the current pointing in the datasets list.

    Parameters
    ----------
    l1b_de_datasets : list[xr.Dataset]
        L1B DE datasets.
    current_repointing : str
        Repointing identifier for the current pointing.

    Returns
    -------
    current_index : int
        Index of the current pointing in the datasets list.

    Raises
    ------
    ValueError
        If the current repointing is not found in the datasets.
    """
    for i, ds in enumerate(l1b_de_datasets):
        if ds.attrs.get("Repointing") == current_repointing:
            logger.info(f"Current pointing index: {i} of {len(l1b_de_datasets)}")
            return i

    raise ValueError(
        f"Could not find current repointing {current_repointing} "
        f"in L1B DE datasets. Available repointings: "
        f"{[ds.attrs.get('Repointing') for ds in l1b_de_datasets]}"
    )


def _apply_goodtimes_filters(
    goodtimes_ds: xr.Dataset,
    l1b_de_datasets: list[xr.Dataset],
    current_index: int,
    l1b_hk: xr.Dataset,
    l1a_diagfee: xr.Dataset,
    cal_product_config_path: Path,
) -> None:
    """
    Apply all goodtimes culling filters to the dataset.

    Modifies goodtimes_ds in place by applying filters 1-7.

    Parameters
    ----------
    goodtimes_ds : xr.Dataset
        Goodtimes dataset to modify.
    l1b_de_datasets : list[xr.Dataset]
        All L1B DE datasets (current + surrounding pointings).
    current_index : int
        Index of the current pointing in l1b_de_datasets.
    l1b_hk : xr.Dataset
        L1B housekeeping dataset.
    l1a_diagfee : xr.Dataset
        L1A DIAG_FEE dataset containing TDC calibration status.
    cal_product_config_path : Path
        Path to calibration product configuration CSV file.
    """
    current_l1b_de = l1b_de_datasets[current_index]

    # Load calibration product config
    logger.info(f"Loading cal product config: {cal_product_config_path}")
    cal_product_config = CalibrationProductConfig.from_csv(cal_product_config_path)

    # Log initial statistics
    stats = goodtimes_ds.goodtimes.get_cull_statistics()
    logger.info(f"Initial good bins: {stats['good_bins']}/{stats['total_bins']}")

    # Pre-compute qualified event masks for each dataset
    # These masks check BOTH coincidence_type AND TOF windows
    for l1b_de in l1b_de_datasets:
        ccsds_index = l1b_de["ccsds_index"].values

        # Handle invalid events (FILLVAL trigger_id) to avoid IndexError
        # For pointings with no valid events, trigger_id will be at FILLVAL
        trigger_id_fillval = l1b_de["trigger_id"].attrs.get("FILLVAL", 65535)
        valid_events = l1b_de["trigger_id"].values != trigger_id_fillval

        # Initialize with -1 (won't match any config row since ESA energy steps > 0)
        esa_energy_steps: np.ndarray = np.full(len(ccsds_index), -1, dtype=np.int32)
        if np.any(valid_events):
            esa_energy_steps[valid_events] = l1b_de["esa_energy_step"].values[
                ccsds_index[valid_events]
            ]

        l1b_de["qualified_mask"] = xr.DataArray(
            compute_qualified_event_mask(l1b_de, cal_product_config, esa_energy_steps),
            dims=["event_met"],
        )
    logger.info("Pre-computed qualified event masks for all datasets")

    # === Apply culling filters ===

    # 1. Mark incomplete spin sets
    logger.info("Applying filter: mark_incomplete_spin_sets")
    mark_incomplete_spin_sets(goodtimes_ds, current_l1b_de)

    # 2. Mark DRF times (drift restabilization)
    logger.info("Applying filter: mark_drf_times")
    mark_drf_times(goodtimes_ds, l1b_hk)

    # 3. Mark bad TDC calibration times
    logger.info("Applying filter: mark_bad_tdc_cal")
    mark_bad_tdc_cal(goodtimes_ds, l1a_diagfee)

    # 4. Mark overflow packets
    logger.info("Applying filter: mark_overflow_packets")
    mark_overflow_packets(goodtimes_ds, current_l1b_de, cal_product_config)

    # 5. Statistical Filter 0 - drastic background changes
    logger.info("Applying filter: mark_statistical_filter_0")
    mark_statistical_filter_0(goodtimes_ds, l1b_de_datasets, current_index)

    # 6. Statistical Filter 1 - isotropic count rate increases
    logger.info("Applying filter: mark_statistical_filter_1")
    mark_statistical_filter_1(
        goodtimes_ds,
        l1b_de_datasets,
        current_index,
    )

    # 7. Statistical Filter 2 - short-lived event pulses
    logger.info("Applying filter: mark_statistical_filter_2")
    mark_statistical_filter_2(
        goodtimes_ds,
        current_l1b_de,
    )


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
        "esa_step": xr.DataArray(esa_step.values, dims=["met"]),
    }

    # Create attributes
    sensor_number = parse_sensor_number(l1b_de.attrs["Logical_source"])
    repointing = l1b_de.attrs.get("Repointing", "repoint-9999")
    attrs = {
        "Sensor": f"{sensor_number}sensor",
        "Repointing": repointing,
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
          ESA step for each MET timestamp
      * Attributes
        * sensor : str
         Sensor identifier ('45sensor' or '90sensor')
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

        # Set cull_flags for all indices using bitwise OR to combine flags
        n_times = len(met_indices)
        n_bins = len(bins_array)
        logger.debug(
            f"Flagging {n_times} MET time(s) x {n_bins} spin bin(s) with "
            f"cull code {cull}"
        )
        self._obj["cull_flags"].values[np.ix_(met_indices, bins_array)] |= np.uint8(
            cull
        )

    def get_good_intervals(self) -> np.ndarray:
        """
        Extract time intervals grouped by contiguous cull flag patterns.

        Merges consecutive MET timestamps that have identical cull_flags patterns
        into single intervals. Each interval spans a contiguous time range where
        cull flags don't change.

        If cull flags have multiple contiguous regions with different values
        (e.g., bins 0-44 good, 45-89 bad), multiple intervals are created for
        the same time range, one per contiguous bin region.

        Returns
        -------
        numpy.ndarray
            Structured array with dtype INTERVAL_DTYPE containing:
            - met_start: First MET timestamp of interval
            - met_end: Last MET timestamp of interval
            - spin_bin_low: Lowest spin bin in this contiguous region
            - spin_bin_high: Highest spin bin in this contiguous region
            - n_bins: Number of bins in this region
            - esa_step_mask: Bitmask of ESA steps (1-10) included in interval
            - cull_value: Cull flag value for this region (0=good, >0=bad)

        Notes
        -----
        This is used for generating the Good Times output files per algorithm
        document Section 2.3.2.5.
        """
        logger.debug("Extracting time intervals")
        met_values = self._obj["met"].values
        cull_flags = self._obj["cull_flags"].values
        esa_steps = self._obj["esa_step"].values

        if len(met_values) == 0:
            logger.warning("No MET values found, returning empty intervals array")
            return np.array([], dtype=INTERVAL_DTYPE)

        # Group consecutive METs with identical cull patterns
        # Each group becomes one or more intervals (one per contiguous bin region)
        intervals: list[tuple] = []

        # Start first group
        group_start_idx = 0
        current_pattern = cull_flags[0]
        # Cast to int to avoid uint8 overflow when esa_step > 8
        esa_step_mask = 1 << int(esa_steps[0] - 1)  # Bit i represents ESA step i+1

        for met_idx in range(1, len(met_values)):
            if np.array_equal(cull_flags[met_idx], current_pattern):
                # Same pattern - extend current group
                esa_step_mask |= 1 << int(esa_steps[met_idx] - 1)
            else:
                # Different pattern - close current group and start new one
                self._add_intervals_for_pattern(
                    intervals,
                    met_values[group_start_idx],
                    met_values[met_idx - 1],
                    current_pattern,
                    esa_step_mask,
                )

                # Start new group
                group_start_idx = met_idx
                current_pattern = cull_flags[met_idx]
                esa_step_mask = 1 << int(esa_steps[met_idx] - 1)

        # Close final group
        self._add_intervals_for_pattern(
            intervals,
            met_values[group_start_idx],
            met_values[-1],
            current_pattern,
            esa_step_mask,
        )

        logger.info(f"Extracted {len(intervals)} time intervals")
        return np.array(intervals, dtype=INTERVAL_DTYPE)

    @staticmethod
    def _add_intervals_for_pattern(
        intervals: list,
        met_start: float,
        met_end: float,
        pattern: np.ndarray,
        esa_step_mask: int,
    ) -> None:
        """
        Add interval(s) for a cull_flags pattern, one per contiguous bin region.

        Creates an interval for each contiguous region of bins that share the
        same cull value. This includes both good (cull=0) and bad (cull>0) regions.

        Parameters
        ----------
        intervals : list
            List to append interval tuples to.
        met_start : float
            Start MET timestamp.
        met_end : float
            End MET timestamp.
        pattern : numpy.ndarray
            Cull flags pattern for spin bins (90 values).
        esa_step_mask : int
            Bitmask of ESA steps included in this time range.
        """
        # Find contiguous regions of bins with the same cull value
        # diff != 0 indicates a change in cull value
        changes = np.nonzero(np.diff(pattern) != 0)[0]

        # Build list of (start_bin, end_bin) for each contiguous region
        # If no changes, entire range is one region
        if len(changes) == 0:
            regions = [(0, 89)]
        else:
            regions = []
            start_bin = 0
            for change_idx in changes:
                regions.append((start_bin, change_idx))
                start_bin = change_idx + 1
            # Add final region
            regions.append((start_bin, 89))

        # Create an interval for each region
        for start_bin, end_bin in regions:
            cull_value = pattern[start_bin]
            n_bins = end_bin - start_bin + 1
            interval = (
                met_start,
                met_end,
                start_bin,
                end_bin,
                n_bins,
                esa_step_mask,
                cull_value,
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
        Write time intervals to text file in the format specified by algorithm document.

        Format per Section 2.3.2.5:
        pointing MET_start MET_end`tab`spin_bin_low spin_bin_high sensor`tab`
        esa_steps[10] cull_value

        The esa_steps field consists of 10 binary values (0 or 1) indicating whether
        each ESA step (1-10) is included in this interval.

        Parameters
        ----------
        output_path : pathlib.Path
            Path where the text file should be written.

        Returns
        -------
        pathlib.Path
            Path to the created file.
        """
        logger.info(f"Writing intervals to file: {output_path}")
        pointing = int(self._obj.attrs["Repointing"].replace("repoint", ""))
        sensor = (
            parse_sensor_number(self._obj.attrs["Logical_source"])
            if "Logical_source" in self._obj.attrs
            else self._obj.attrs["Sensor"].replace("sensor", "")
        )

        intervals = self.get_good_intervals()

        with open(output_path, "w") as f:
            # Write header info
            file_id = self._obj.attrs.get("Logical_file_id")
            if file_id is not None:
                f.write(
                    f"# Goodtimes txt file generated for input CDF: {file_id}" + "\n"
                )
            for interval in intervals:
                # Convert esa_step_mask bitmask to 10 binary values
                # Bit i represents ESA step i+1, so check bits 0-9
                esa_step_mask = int(interval["esa_step_mask"])
                esa_step_flags = " ".join(
                    "1" if (esa_step_mask >> i) & 1 else "0" for i in range(10)
                )

                # Format:
                # pointing met_start met_end spin_bin_low spin_bin_high sensor
                # esa_steps[10] cull_value
                line = (
                    f"{pointing:05d} "
                    f"{int(interval['met_start'])} "
                    f"{int(interval['met_end'])}\t"
                    f"{interval['spin_bin_low']} "
                    f"{interval['spin_bin_high']} "
                    f"{sensor}\t"
                    f"{esa_step_flags}\t"
                    f"{interval['cull_value']}"
                )

                # TODO: Add rate/sigma values for each ESA step

                f.write(line + "\n")

        logger.info(f"Wrote {len(intervals)} intervals to {output_path}")
        return output_path

    def finalize_dataset(self) -> xr.Dataset:
        """
        Finalize the goodtimes dataset for CDF output.

        Converts the dataset from using MET as the primary dimension to using
        epoch (TT2000 nanoseconds), and adds all CDF attributes required for
        L1B CDF file writing.

        Returns
        -------
        xarray.Dataset
            CDF-ready dataset with epoch dimension and all CDF attributes.

        Notes
        -----
        This method should be called after all goodtimes filtering is complete,
        just before writing to CDF.

        Requires SPICE kernels to be loaded for MET to epoch conversion.
        """
        logger.info("Finalizing goodtimes dataset for CDF output")

        # Initialize CDF attribute manager
        attr_mgr = ImapCdfAttributes()
        attr_mgr.add_instrument_global_attrs("hi")
        attr_mgr.add_instrument_variable_attrs("hi")

        # Convert MET coordinate to epoch coordinate (TT2000 nanoseconds)
        met_values = self._obj.coords["met"].values
        epoch_values = met_to_ttj2000ns(met_values)

        # Rename met dimension to epoch and assign new epoch coordinate values
        ds = self._obj.rename({"met": "epoch"})
        ds = ds.assign_coords(epoch=epoch_values)

        # Move met from coordinate to data variable
        ds["met"] = xr.DataArray(met_values, dims=["epoch"])

        # Add spin_bin_label coordinate
        spin_bin_label = np.array([f"{i}" for i in ds.coords["spin_bin"].values])
        ds = ds.assign_coords(spin_bin_label=("spin_bin", spin_bin_label))

        # Add coordinate attributes
        ds["epoch"].attrs = attr_mgr.get_variable_attributes(
            "epoch", check_schema=False
        )
        for coord_name in ds.coords:
            attr_mgr_key = (
                f"hi_goodtimes_{coord_name}" if coord_name != "epoch" else "epoch"
            )
            ds[coord_name].attrs = attr_mgr.get_variable_attributes(
                attr_mgr_key, check_schema=False
            )

        # Add variable attributes
        for var_name in ds.data_vars:
            ds[var_name].attrs.update(
                attr_mgr.get_variable_attributes(f"hi_goodtimes_{var_name}")
            )

        # Update global attributes
        sensor_str = ds.attrs.pop("Sensor")
        ds.attrs = attr_mgr.get_global_attributes("imap_hi_l1b_goodtimes_attrs")

        # Update Logical_source with sensor string
        ds.attrs["Logical_source"] = ds.attrs["Logical_source"].format(
            sensor=sensor_str
        )

        return ds


# ==============================================================================
# Culling/Filtering Functions
# Based on culling.c - Reference: IMAP-Hi Algorithm Document Sections 2.2.4, 2.3.2
# ==============================================================================


def mark_incomplete_spin_sets(
    goodtimes_ds: xr.Dataset,
    l1b_de: xr.Dataset,
    cull_code: int = CullCode.INCOMPLETE_SPIN,
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
    cull_code: int = CullCode.DRF,
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
    hk_met = hk["shcoarse"]
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
    cull_code: int = CullCode.OVERFLOW,
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
    last_event_per_packet: np.ndarray = np.full(max_packet_idx + 1, -1, dtype=np.intp)
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


def mark_bad_tdc_cal(
    goodtimes_ds: xr.Dataset,
    diagfee: xr.Dataset,
    cull_code: int = CullCode.BAD_TDC_CAL,
) -> None:
    """
    Remove times with failed TDC calibration (DIAG_FEE method).

    Based on C reference: drop_bad_tdc_diagfee in culling_v2.c provided by
    IMAP-Hi team.

    This function scans DIAG_FEE packets chronologically and checks the TDC
    calibration status for each packet. If any TDC has failed calibration,
    all times from that DIAG_FEE packet until the next DIAG_FEE packet are
    marked as bad.

    Parameters
    ----------
    goodtimes_ds : xr.Dataset
        Goodtimes dataset to update with cull flags.
    diagfee : xr.Dataset
        DIAG_FEE dataset containing TDC calibration status fields:
        - shcoarse: Mission Elapsed Time (MET)
        - tdc1_cal_ctrl_stat: TDC1 calibration status (bit 1 = success)
        - tdc2_cal_ctrl_stat: TDC2 calibration status (bit 1 = success)
        - tdc3_cal_ctrl_stat: TDC3 calibration status (bit 1 = success)
    cull_code : int, optional
        Cull code to use for marking bad times. Default is CullCode.LOOSE.

    Notes
    -----
    This function modifies goodtimes_ds in place.

    Quirk: Two DIAG_FEE packets are generated when entering HVSCI mode.
    The first packet is skipped if two packets appear within 10 seconds.
    """
    logger.info("Running mark_bad_tdc_cal culling")

    # Based on sample code in culling_v2.c, skip this check if we have fewer
    # than two diag_fee packets.
    if len(diagfee.epoch) < 2:
        logger.warning(
            f"Insufficient DIAG_FEE packets to select good times "
            f"(found {len(diagfee.epoch)}, need at least 2)"
        )
        return

    diagfee_met = diagfee["shcoarse"].values
    goodtimes_met = goodtimes_ds.coords["met"].values

    # Identify duplicate packets: skip if followed by another within 10 seconds
    time_gaps = np.diff(diagfee_met)
    is_duplicate = np.concatenate([time_gaps < 10, [False]])

    # Identify any packets where any of the three TDC calibrations failed.
    # TDC failure check (bit 1: 1=good, 0=bad)
    tdc_failed = (
        ((diagfee["tdc1_cal_ctrl_stat"].values & 2) == 0)
        | ((diagfee["tdc2_cal_ctrl_stat"].values & 2) == 0)
        | ((diagfee["tdc3_cal_ctrl_stat"].values & 2) == 0)
    )

    # Only loop over non-duplicate packets with TDC failures
    tdc_failed_indices = np.nonzero(~is_duplicate & tdc_failed)[0]

    n_times_removed = 0
    for i in tdc_failed_indices:
        # Remove times from this DIAG_FEE packet until next. We are skipping the
        # first packet of a duplicate pair, so determining the window based on the
        # current packet met and next packet met covers the time window between
        # non-duplicate DIAG_FEE packets. We can ignore the ~10 seconds of slop
        # around duplicate packets because these packets should only be produced
        # when IMAP-Hi is transitioning to HVSCI mode which means that there will
        # be no DE packets being produced.
        df_time = diagfee_met[i]
        next_df_time = diagfee_met[i + 1] if i < len(diagfee_met) - 1 else np.inf

        in_window = (goodtimes_met >= df_time) & (goodtimes_met < next_df_time)
        mets_to_cull = goodtimes_met[in_window]

        if len(mets_to_cull) > 0:
            goodtimes_ds.goodtimes.mark_bad_times(met=mets_to_cull, cull=cull_code)
            n_times_removed += len(mets_to_cull)

    logger.info(f"Dropped {n_times_removed} time(s) due to bad TDC calibration")


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
    counts_per_sweep: np.ndarray = np.zeros(n_sweeps, dtype=np.int64)
    np.add.at(counts_per_sweep, event_sweep_idx[is_valid_ab.values], 1)

    # Normalize by number of unique ESA energy steps
    n_unique_esa_energy_steps = len(np.unique(l1b_de["esa_energy_step"].values))
    normalized_counts = counts_per_sweep / n_unique_esa_energy_steps

    # Remove all variables that depend on event_met dimension
    ds = l1b_de.drop_dims("event_met", errors="ignore")

    # Set esa_sweep and esa_energy_step as a multi-index on epoch dimension
    ds = ds.set_index(epoch=["esa_sweep", "esa_energy_step"])

    # Drop duplicates, keeping first occurrence of each (esa_sweep, esa_energy_step)
    # pair. This handles cases where multiple packets have the same esa_sweep
    # and esa_energy_step.
    ds = ds.drop_duplicates(dim="epoch", keep="first")

    # Unstack to make esa_sweep and esa_energy_step into separate dimensions
    # This creates a 2D array with dimensions (esa_sweep, esa_energy_step)
    ds_reshaped = ds.unstack("epoch")

    # Add normalized_count as a new variable
    # It only has esa_sweep dimension (no esa_energy_step variation within a sweep)
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
    threshold_factor: float = HiConstants.STAT_FILTER_0_THRESHOLD_FACTOR,
    tof_ab_limit_ns: int = HiConstants.STAT_FILTER_0_TOF_AB_LIMIT_NS,
    cull_code: int = CullCode.STAT_FILTER_0,
    min_pointings: int = HiConstants.STAT_FILTER_MIN_POINTINGS,
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
        Multiplier for median comparison.
        Default is HiConstants.STAT_FILTER_0_THRESHOLD_FACTOR.
    tof_ab_limit_ns : int, optional
        Maximum |tof_ab| in nanoseconds for AB coincidences.
        Default is HiConstants.STAT_FILTER_0_TOF_AB_LIMIT_NS.
    cull_code : int, optional
        Cull code to use for marking bad times. Default is CullCode.LOOSE.
    min_pointings : int, optional
        Minimum number of Pointings required.
        Default is HiConstants.STAT_FILTER_MIN_POINTINGS.

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
       where |tof_ab| <= tof_ab_limit_ns and divide by number of ESA steps
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
        # Get all ccsds_met values for this sweep across all esa_energy_steps
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


def _compute_qualified_counts_per_sweep(
    l1b_de: xr.Dataset,
    qualified_mask: np.ndarray,
) -> xr.Dataset:
    """
    Compute qualified calibration product counts per 8-spin interval and reshape.

    Uses the (esa_sweep, esa_energy_step) multi-index to identify unique 8-spin sets,
    following the same pattern as _compute_normalized_counts_per_sweep.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        L1B Direct Event dataset with esa_sweep coordinate on epoch dimension.
    qualified_mask : np.ndarray
        Boolean mask indicating which events qualify for calibration products.
        This mask should check BOTH coincidence_type AND TOF windows.

    Returns
    -------
    xarray.Dataset
        Reshaped dataset with dimensions (esa_sweep, esa_energy_step) containing:
        - qualified_count: total qualified counts per 8-spin interval
        - ccsds_met: first MET for each 8-spin interval
    """
    if "esa_sweep" not in l1b_de.coords:
        raise ValueError("Dataset must have esa_sweep coordinate")

    # Get values needed for counting
    ccsds_index = l1b_de["ccsds_index"].values
    esa_sweep = l1b_de.coords["esa_sweep"].values
    esa_energy_step = l1b_de["esa_energy_step"].values

    # Use pre-computed qualified mask
    is_qualified = qualified_mask

    # Map qualified events to their packet's (esa_sweep, esa_energy_step)
    qualified_packet_idx = ccsds_index[is_qualified]
    qualified_sweep = esa_sweep[qualified_packet_idx]
    qualified_energy_step = esa_energy_step[qualified_packet_idx]

    # Count qualified events per (esa_sweep, esa_energy_step) using 2D array
    n_sweeps = int(esa_sweep.max()) + 1
    n_esa_energy_steps = int(esa_energy_step.max()) + 1
    counts_2d: np.ndarray = np.zeros((n_sweeps, n_esa_energy_steps), dtype=np.float64)
    np.add.at(counts_2d, (qualified_sweep, qualified_energy_step), 1)

    # Remove event_met dimension and reshape using multi-index
    ds = l1b_de.drop_dims("event_met", errors="ignore")
    ds = ds.set_index(epoch=["esa_sweep", "esa_energy_step"])
    ds = ds.drop_duplicates(dim="epoch", keep="first")
    ds_reshaped = ds.unstack("epoch")

    # Add qualified_count - aligns with (esa_sweep, esa_energy_step) coordinates
    ds_reshaped["qualified_count"] = xr.DataArray(
        counts_2d,
        dims=["esa_sweep", "esa_energy_step"],
        coords={
            "esa_sweep": np.arange(n_sweeps),
            "esa_energy_step": np.arange(n_esa_energy_steps),
        },
    )

    # Set missing (sweep, energy_step) pairs to NaN so they don't affect statistics
    missing_mask = ds_reshaped["ccsds_met"].isnull()
    ds_reshaped["qualified_count"] = ds_reshaped["qualified_count"].where(~missing_mask)

    return ds_reshaped


def _build_per_sweep_datasets(
    l1b_de_datasets: list[xr.Dataset],
) -> dict[int, xr.Dataset]:
    """
    Build per-sweep datasets with qualified counts for each Pointing.

    Parameters
    ----------
    l1b_de_datasets : list[xarray.Dataset]
        List of L1B DE datasets for multiple Pointings. Each dataset must
        contain a "qualified_mask" DataArray indicating which events qualify
        for calibration products.

    Returns
    -------
    dict[int, xarray.Dataset]
        Dictionary mapping dataset index to 2D Dataset with
        (esa_sweep, esa_energy_step) dims.
    """
    per_sweep_datasets: dict[int, xr.Dataset] = {}

    for i, l1b_de in enumerate(l1b_de_datasets):
        # Add esa_sweep coordinate and compute counts per 8-spin interval
        l1b_de_with_sweep = _add_sweep_indices(l1b_de)
        per_sweep = _compute_qualified_counts_per_sweep(
            l1b_de_with_sweep, l1b_de["qualified_mask"].values
        )
        per_sweep_datasets[i] = per_sweep

    return per_sweep_datasets


def _compute_median_and_sigma_per_esa(
    per_sweep_datasets: dict[int, xr.Dataset],
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute median and sigma for each ESA energy step using xarray.

    Combines all per-sweep datasets and computes the median qualified count
    per ESA energy step across all sweeps and pointings.

    Parameters
    ----------
    per_sweep_datasets : dict[int, xarray.Dataset]
        Dictionary mapping dataset index to 2D Dataset with
        (esa_sweep, esa_energy_step) dims.

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray]
        Tuple of (median_per_esa, sigma_per_esa) DataArrays with esa_energy_step
        coordinate. ESA energy steps with zero/nan median or esa_energy_step=0
        are set to NaN/0.
    """
    if not per_sweep_datasets:
        empty = xr.DataArray(
            [], dims=["esa_energy_step"], coords={"esa_energy_step": []}
        )
        return empty, empty.astype(int)

    # Concatenate datasets along esa_sweep dimension using xarray. This handles
    # different esa_energy_step coordinates by aligning and filling with NaN.
    combined = xr.concat(
        [ds["qualified_count"] for ds in per_sweep_datasets.values()],
        dim="esa_sweep",
    )

    # Compute median along esa_sweep dimension using xarray
    median_per_esa = combined.median(dim="esa_sweep", skipna=True)

    # Compute sigma: sigma ≈ √(median + 1) rounded to closest integer
    sigma_per_esa = np.sqrt(median_per_esa + 1).round().astype(int)

    # Set invalid ESA energy steps (zero/nan median or esa_energy_step=0) to NaN/0
    esa_energy_step_coords = median_per_esa.coords["esa_energy_step"]
    invalid_mask = (
        (esa_energy_step_coords == 0) | (median_per_esa <= 0) | median_per_esa.isnull()
    )
    median_per_esa = median_per_esa.where(~invalid_mask)
    sigma_per_esa = sigma_per_esa.where(~invalid_mask, 0)

    # Log warnings for invalid ESA energy steps (excluding esa_energy_step=0)
    invalid_esa_energy_steps = esa_energy_step_coords.values[
        (esa_energy_step_coords != 0).values & invalid_mask.values
    ]
    for esa in invalid_esa_energy_steps:
        logger.warning(
            f"Statistical Filter 1: Median is zero/nan for ESA energy step {esa}, "
            "skipping this ESA energy step"
        )

    # Log valid ESA energy steps
    valid_esa_energy_steps = esa_energy_step_coords.values[~invalid_mask.values]
    for esa in valid_esa_energy_steps:
        logger.debug(
            f"Statistical Filter 1: ESA {esa}: "
            f"median={median_per_esa.sel(esa_energy_step=esa).values:.2f}, "
            f"sigma={sigma_per_esa.sel(esa_energy_step=esa).values}"
        )

    return median_per_esa, sigma_per_esa


def _identify_cull_pattern(
    current_counts: xr.DataArray,
    median_per_esa: xr.DataArray,
    sigma_per_esa: xr.DataArray,
    consecutive_threshold_sigma: float = HiConstants.STAT_FILTER_1_CONSECUTIVE_SIGMA,
    extreme_threshold_sigma: float = HiConstants.STAT_FILTER_1_EXTREME_SIGMA,
    min_consecutive: int = HiConstants.STAT_FILTER_1_MIN_CONSECUTIVE,
) -> xr.DataArray:
    """
    Identify 2D cull pattern for statistical filter 1 using convolution.

    Detects three patterns:
    1. Consecutive runs: 3+ consecutive sweeps exceeding threshold with ESA neighbor
       confirmation (isotropic excursion pattern from C implementation)
    2. Isolated intervals: Good intervals surrounded by bad on both sides in time
    3. Extreme outliers: Any position exceeding 5-sigma threshold

    Parameters
    ----------
    current_counts : xr.DataArray
        2D array of qualified counts with dims (esa_sweep, esa_energy_step).
    median_per_esa : xr.DataArray
        Median counts per ESA energy step.
    sigma_per_esa : xr.DataArray
        Sigma values per ESA energy step.
    consecutive_threshold_sigma : float
        Sigma multiplier for consecutive interval check.
        Default is HiConstants.STAT_FILTER_1_CONSECUTIVE_SIGMA.
    extreme_threshold_sigma : float
        Sigma multiplier for extreme outlier check.
        Default is HiConstants.STAT_FILTER_1_EXTREME_SIGMA.
    min_consecutive : int
        Minimum consecutive intervals above threshold.
        Default is HiConstants.STAT_FILTER_1_MIN_CONSECUTIVE.

    Returns
    -------
    xr.DataArray
        Boolean mask with dims (esa_sweep, esa_energy_step) where
        True = cull this position.
    """
    # Compute thresholds using xarray broadcasting
    consecutive_threshold = median_per_esa + consecutive_threshold_sigma * sigma_per_esa
    extreme_threshold = median_per_esa + extreme_threshold_sigma * sigma_per_esa

    # Compute exceeds masks - handle NaN by treating as False
    exceeds_consecutive = (current_counts > consecutive_threshold).fillna(False)
    exceeds_extreme = (current_counts > extreme_threshold).fillna(False)

    # Get underlying numpy arrays for convolution (dims: esa_sweep x esa_energy_step)
    exceeds_arr = exceeds_consecutive.values.astype(int)

    # Initialize cull mask
    cull_arr = np.zeros_like(exceeds_arr, dtype=bool)

    # === Pass 1: Find consecutive runs with ESA neighbor confirmation ===
    # Use convolution to find runs of min_consecutive in time (axis=0 = esa_sweep)
    time_kernel = np.ones(min_consecutive)
    consecutive_sum = convolve1d(exceeds_arr, time_kernel, axis=0, mode="constant")

    # Dilate the consecutive detection to mark all positions in runs
    # convolve1d centers the kernel, so we dilate to capture run edges
    run_kernel = np.ones(min_consecutive)
    run_positions = convolve1d(
        (consecutive_sum >= min_consecutive).astype(int),
        run_kernel,
        axis=0,
        mode="constant",
    )
    in_consecutive_run = (run_positions >= 1) & exceeds_arr.astype(bool)

    # Check ESA neighbors at same time position using convolution along ESA axis
    # Kernel [1, 0, 1] sums neighbors without counting self
    # Use cval=1 so edges pass the neighbor check (matches C implementation where
    # edges are treated as "not good", i.e., the check passes at boundaries)
    esa_neighbor_kernel = np.array([1, 0, 1])
    esa_neighbor_exceeds = convolve1d(
        exceeds_arr, esa_neighbor_kernel, axis=1, mode="constant", cval=1
    )
    has_esa_neighbor = esa_neighbor_exceeds >= 1

    # Combine: in a consecutive run AND has ESA neighbor exceeding at same time
    cull_arr |= in_consecutive_run & has_esa_neighbor

    # === Pass 2: Mark isolated good intervals (orphans) ===
    # Pattern: [bad, good, bad] in time dimension
    # Sum neighbors in time - if both neighbors are bad (in cull_arr), sum = 2
    # Kernel [1, 0, 1] sums neighbors without counting self
    neighbor_kernel = np.array([1, 0, 1])
    bad_neighbor_sum = convolve1d(
        cull_arr.astype(int), neighbor_kernel, axis=0, mode="constant"
    )
    # Current position is good (not in cull_arr) but both time neighbors are bad
    isolated = ~cull_arr & (bad_neighbor_sum == 2)
    cull_arr |= isolated

    # Log isolated intervals found
    n_isolated = int(isolated.sum())
    if n_isolated > 0:
        logger.debug(f"Statistical Filter 1: Found {n_isolated} isolated intervals")

    # === Pass 3: Mark extreme outliers (5-sigma) ===
    extreme_arr = exceeds_extreme.values
    n_extreme = int((extreme_arr & ~cull_arr).sum())
    if n_extreme > 0:
        logger.debug(f"Statistical Filter 1: Found {n_extreme} extreme outliers")
    cull_arr |= extreme_arr

    # Convert back to xarray DataArray with same coordinates
    cull_mask = xr.DataArray(
        cull_arr,
        dims=current_counts.dims,
        coords=current_counts.coords,
    )

    return cull_mask


def mark_statistical_filter_1(
    goodtimes_ds: xr.Dataset,
    l1b_de_datasets: list[xr.Dataset],
    current_index: int,
    consecutive_threshold_sigma: float = HiConstants.STAT_FILTER_1_CONSECUTIVE_SIGMA,
    extreme_threshold_sigma: float = HiConstants.STAT_FILTER_1_EXTREME_SIGMA,
    min_consecutive_intervals: int = HiConstants.STAT_FILTER_1_MIN_CONSECUTIVE,
    cull_code: int = CullCode.STAT_FILTER_1,
    min_pointings: int = HiConstants.STAT_FILTER_MIN_POINTINGS,
) -> None:
    """
    Apply Statistical Filter 1 to detect isotropic count rate increases.

    Statistical Filter 1 from Algorithm Document Section 2.3.2.3 detects times
    when qualified calibration product counts increase fairly isotropically for
    a limited time. It operates per sensor, per ESA energy step, per 8-spin
    interval, summing counts over all angles.

    The filter applies three passes:
    1. Mark intervals where counts exceed median + consecutive_threshold_sigma
       for at least min_consecutive_intervals AND in at least one adjacent ESA step.
    2. Remove isolated good intervals (good sandwiched between two bad).
    3. Mark remaining intervals where counts exceed median + extreme_threshold_sigma.

    Parameters
    ----------
    goodtimes_ds : xarray.Dataset
        Goodtimes dataset for the current Pointing to update.
    l1b_de_datasets : list[xarray.Dataset]
        List of L1B DE datasets for surrounding Pointings. Typically includes
        current plus 3 preceding and 3 following Pointings. Each dataset must
        contain a "qualified_mask" DataArray indicating which events qualify
        for calibration products (checking both coincidence_type AND TOF).
    current_index : int
        Index of the current Pointing in l1b_de_datasets.
    consecutive_threshold_sigma : float, optional
        Sigma multiplier for consecutive interval check.
        Default is HiConstants.STAT_FILTER_1_CONSECUTIVE_SIGMA.
    extreme_threshold_sigma : float, optional
        Sigma multiplier for extreme outlier check.
        Default is HiConstants.STAT_FILTER_1_EXTREME_SIGMA.
    min_consecutive_intervals : int, optional
        Minimum consecutive intervals above threshold.
        Default is HiConstants.STAT_FILTER_1_MIN_CONSECUTIVE.
    cull_code : int, optional
        Cull code to use for marking bad times. Default is CullCode.LOOSE.
    min_pointings : int, optional
        Minimum number of Pointings required.
        Default is HiConstants.STAT_FILTER_MIN_POINTINGS.

    Raises
    ------
    ValueError
        If current_index is out of range or if fewer than min_pointings
        datasets are provided.

    Notes
    -----
    This function modifies goodtimes_ds in place. Should be called after
    Statistical Filter 0 and other angle-independent filters.
    """
    logger.info("Running mark_statistical_filter_1 culling")

    # Validate inputs
    if current_index < 0 or current_index >= len(l1b_de_datasets):
        raise ValueError(
            f"current_index {current_index} out of range for list of "
            f"length {len(l1b_de_datasets)}"
        )

    if len(l1b_de_datasets) < min_pointings:
        raise ValueError(
            f"At least {min_pointings} valid Pointings required, "
            f"got {len(l1b_de_datasets)}"
        )

    # Step 1: Build per-sweep datasets with qualified counts for each Pointing
    per_sweep_datasets = _build_per_sweep_datasets(l1b_de_datasets)

    # Step 2: Compute median and sigma per ESA energy step using xarray
    median_per_esa, sigma_per_esa = _compute_median_and_sigma_per_esa(
        per_sweep_datasets
    )

    if np.all(np.isnan(median_per_esa.values)):
        logger.warning(
            "Statistical Filter 1: No valid ESA energy steps "
            "with non-zero median, skipping"
        )
        return

    # Get current Pointing's per-sweep data (2D: esa_sweep x esa_energy_step)
    current_ds = per_sweep_datasets[current_index]
    current_counts = current_ds["qualified_count"]

    # Identify cull pattern using convolution-based detection
    cull_mask = _identify_cull_pattern(
        current_counts,
        median_per_esa,
        sigma_per_esa,
        consecutive_threshold_sigma=consecutive_threshold_sigma,
        extreme_threshold_sigma=extreme_threshold_sigma,
        min_consecutive=min_consecutive_intervals,
    )

    # Apply culling to goodtimes - get METs where cull_mask is True
    if cull_mask.any():
        # Use xarray's where to get METs for culled intervals, then flatten
        mets_to_cull = current_ds["ccsds_met"].where(cull_mask).values.ravel()
        # Remove NaN values
        mets_to_cull = mets_to_cull[~np.isnan(mets_to_cull)]

        if len(mets_to_cull) > 0:
            goodtimes_ds.goodtimes.mark_bad_times(met=mets_to_cull, cull=cull_code)

        logger.info(
            f"Statistical Filter 1: Marked {len(mets_to_cull)} 8-spin intervals as bad"
        )
    else:
        logger.info("Statistical Filter 1: No bad intervals identified")


def _find_event_clusters(
    event_times: np.ndarray,
    min_events: int,
    max_time_delta: float,
) -> list[tuple[int, int]]:
    """
    Find clusters of events that occur within a maximum time window.

    Uses vectorized numpy operations to find groups of min_events or more
    events that all occur within max_time_delta seconds of each other.

    Parameters
    ----------
    event_times : np.ndarray
        Sorted array of event times (event_met values in seconds).
    min_events : int
        Minimum number of events to form a cluster.
    max_time_delta : float
        Maximum time span in seconds for events to be considered clustered.

    Returns
    -------
    list[tuple[int, int]]
        List of (start_idx, end_idx) tuples marking cluster boundaries
        in the input array. Indices are inclusive.
    """
    if len(event_times) < min_events:
        return []

    # Compute time span for each window of min_events consecutive events
    # window_spans[i] = event_times[i + min_events - 1] - event_times[i]
    window_spans = event_times[min_events - 1 :] - event_times[: -(min_events - 1)]

    # Mask where windows fit within max_time_delta
    cluster_mask = window_spans <= max_time_delta

    if not np.any(cluster_mask):
        return []

    # Find contiguous regions of True in the mask. Each contiguous region
    # [i, j] in the mask corresponds to cluster [i, j + min_events - 1]

    # Pad with False to handle edge cases
    padded = np.concatenate(([False], cluster_mask, [False]))

    # Find transitions: +1 = start of group, -1 = end of group
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) + min_events - 2  # Adjust for window size

    return list(zip(starts.tolist(), ends.tolist(), strict=False))


def _compute_bins_for_cluster(
    nominal_bins: np.ndarray,
    cluster_start: int,
    cluster_end: int,
    bin_padding: int,
    n_bins: int = 90,
) -> np.ndarray:
    """
    Compute the spin bins to cull for a cluster of events, with wrapping.

    Parameters
    ----------
    nominal_bins : np.ndarray
        Array of nominal_bin values for events.
    cluster_start : int
        Start index of cluster in nominal_bins array.
    cluster_end : int
        End index of cluster (inclusive).
    bin_padding : int
        Number of bins to add on each side.
    n_bins : int
        Total number of spin bins (default 90).

    Returns
    -------
    np.ndarray
        Array of bin indices to cull, with wrapping handled.
        For example, if cluster spans bins 88-91 with n_bins=90,
        returns [87, 88, 89, 0, 1, 2] (with padding=1).
    """
    cluster_bins: np.ndarray = nominal_bins[cluster_start : cluster_end + 1].astype(
        np.int32
    )

    # Unwrap to handle clusters spanning the 0/n_bins boundary
    unwrapped = np.unwrap(cluster_bins, period=n_bins)
    bin_min = int(np.min(unwrapped))
    bin_max = int(np.max(unwrapped))

    # Add padding
    bin_low = bin_min - bin_padding
    bin_high = bin_max + bin_padding

    # Generate bin indices with wrapping using modulo
    bins_to_mark: np.ndarray = np.arange(bin_low, bin_high + 1) % n_bins

    logger.debug(f"Cluster {cluster_start} to {cluster_end} bins: {bins_to_mark}")

    return bins_to_mark


def mark_statistical_filter_2(
    goodtimes_ds: xr.Dataset,
    l1b_de: xr.Dataset,
    min_events: int = HiConstants.STAT_FILTER_2_MIN_EVENTS,
    max_time_delta: float = HiConstants.STAT_FILTER_2_MAX_TIME_DELTA,
    bin_padding: int = HiConstants.STAT_FILTER_2_BIN_PADDING,
    cull_code: int = CullCode.STAT_FILTER_2,
) -> None:
    """
    Apply Statistical Filter 2 to detect short-lived event pulses.

    Statistical Filter 2 from Algorithm Document Section 2.3.2.3 removes
    occasional short-lived "pulses" of qualified counts that may be
    temporally correlated between sensors. These pulses are usually visible
    only at the highest few energy steps and are not caught by Filter 1.

    For each 8-spin set (grouped by esa_sweep and esa_step), this filter:
    1. Keeps only events qualifying as calibration product 1 or 2
    2. Sorts events by event_met
    3. Finds time ranges where min_events or more events occur within
       max_time_delta seconds
    4. Marks the angle range covered by each pulse (plus bin_padding bins
       on each side, with wrapping) as not good for all METs in that 8-spin set

    Parameters
    ----------
    goodtimes_ds : xr.Dataset
        Goodtimes dataset for the current Pointing to update.
    l1b_de : xr.Dataset
        L1B Direct Event dataset for the current Pointing containing:
        - ccsds_index: packet index for each event
        - ccsds_met: MET timestamp for each packet
        - event_met: MET timestamp for each event
        - coincidence_type: detector coincidence bitmap
        - nominal_bin: spacecraft spin bin (0-89)
        - esa_step: ESA energy step for each packet
        - qualified_mask: boolean mask indicating which events qualify for
          calibration products (checking both coincidence_type AND TOF windows)
    min_events : int, optional
        Minimum events to form a pulse cluster.
        Default is HiConstants.STAT_FILTER_2_MIN_EVENTS.
    max_time_delta : float, optional
        Maximum time span in seconds for events to be considered clustered.
        Default is HiConstants.STAT_FILTER_2_MAX_TIME_DELTA.
    bin_padding : int, optional
        Number of 4-degree bins to add on each side of the pulse angle range.
        Default is HiConstants.STAT_FILTER_2_BIN_PADDING.
    cull_code : int, optional
        Cull code to use for marking bad times. Default is CullCode.LOOSE.

    Notes
    -----
    This function modifies goodtimes_ds in place. It marks specific spin
    bins as bad, unlike Filters 0 and 1 which mark entire time intervals.
    Bin marking wraps around (e.g., bin 91 becomes bin 1).

    The default parameters (min_events=6,
    max_time_delta=HiConstants.STAT_FILTER_2_MAX_TIME_DELTA ≈ 9.995 s)
    imply that seeing 6+ qualified events in an approximately 10-second
    window has probability < 0.06% under normal conditions
    (background rate ~0.1/s).
    """
    logger.info("Running mark_statistical_filter_2 culling")

    # Add esa_sweep coordinate to group packets into 8-spin sets
    l1b_de_with_sweep = _add_sweep_indices(l1b_de)

    # Get packet-level arrays
    ccsds_index = l1b_de_with_sweep["ccsds_index"].values
    esa_sweep = l1b_de_with_sweep.coords["esa_sweep"].values
    esa_step = l1b_de_with_sweep["esa_step"].values

    # Add event-level coordinates for grouping
    l1b_de_with_sweep = l1b_de_with_sweep.assign_coords(
        event_sweep=("event_met", esa_sweep[ccsds_index]),
        event_step=("event_met", esa_step[ccsds_index]),
    )

    # Get qualified mask from the dataset
    qualified_mask = l1b_de["qualified_mask"].values

    if not np.any(qualified_mask):
        logger.info("Statistical Filter 2: No qualified events found")
        return

    qualified_events = l1b_de_with_sweep.isel(event_met=qualified_mask)

    n_clusters_found = 0
    n_bins_marked = 0

    # Process each 8-spin set using xarray groupby
    for (sweep_idx, step_idx), group in qualified_events.groupby(
        ["event_sweep", "event_step"]
    ):
        # Sort by event_met
        sorted_group = group.sortby("event_met")
        sorted_mets = sorted_group["event_met"].values
        sorted_bins = sorted_group["nominal_bin"].values

        # Find clusters
        clusters = _find_event_clusters(sorted_mets, min_events, max_time_delta)

        if not clusters:
            continue

        # Get all METs for this 8-spin set (to mark all packets in the set)
        set_mets = l1b_de_with_sweep["ccsds_met"].values[
            (esa_sweep == sweep_idx) & (esa_step == step_idx)
        ]

        # Mark bins for each cluster
        for cluster_start, cluster_end in clusters:
            bins_to_mark = _compute_bins_for_cluster(
                sorted_bins, cluster_start, cluster_end, bin_padding
            )

            # Mark the bins as bad for all METs in this 8-spin set
            for met in set_mets:
                goodtimes_ds.goodtimes.mark_bad_times(
                    met=met, bins=bins_to_mark, cull=cull_code
                )

            n_clusters_found += 1
            n_bins_marked += len(bins_to_mark) * len(set_mets)

            logger.debug(
                f"Statistical Filter 2: ESA sweep={sweep_idx}, step={step_idx}, "
                f"cluster of {cluster_end - cluster_start + 1} events, "
                f"marking {len(bins_to_mark)} bins across {len(set_mets)} METs"
            )

    if n_clusters_found > 0:
        logger.info(
            f"Statistical Filter 2: Found {n_clusters_found} pulse cluster(s), "
            f"marked {n_bins_marked} bin-intervals as bad"
        )
    else:
        logger.info("Statistical Filter 2: No pulse clusters identified")
