# mypy: disable-error-code="unused-ignore"
"""Data classes for MAG L1D processing."""

import logging
from dataclasses import InitVar, dataclass, field

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag import imap_mag_sdc_configuration_v001 as configuration
from imap_processing.mag.constants import FILLVAL, DataMode
from imap_processing.mag.l1c.interpolation_methods import linear
from imap_processing.mag.l2.mag_l2 import retrieve_matrix_from_l2_calibration
from imap_processing.mag.l2.mag_l2_data import MagL2L1dBase, ValidFrames
from imap_processing.spice import spin
from imap_processing.spice.geometry import frame_transform
from imap_processing.spice.time import ttj2000ns_to_et, ttj2000ns_to_met

logger = logging.getLogger(__name__)


@dataclass
class MagL1dConfiguration:
    """
    Configuration for MAG L1d processing.

    Constructed from the combined ancillary dataset inputs from the L1D calibration
    files and the day we are processing.

    Parameters
    ----------
    calibration_dataset : xr.Dataset
        The combined calibration dataset from the ancillary files. Created as the
        output from MagAncillaryCombiner, which has day values pointing to the
        calibration file for the given day.
    day : np.datetime64
        The day we are processing, in np.datetime64[D] format.

    Attributes
    ----------
    calibration_offsets : np.ndarray
        The offsets for the correct day. Should be size (2, 4, 3) where the first index
        is 0 for MAGo and 1 for MAGi, the second index is the range (0-3), and the
        third index is the axis (0-2).
    mago_calibration : np.ndarray
        Calibration matrix for the correct day for MAGo. Should be size (3, 3, 4).
    magi_calibration : np.ndarray
        Calibration matrix for the correct day for MAGi. Should be size (3, 3, 4).
    spin_count_calibration : int
        The number of spins to average over when calculating spin offsets.
    quality_flag_threshold : np.float64
        The quality flag threshold for the correct day.
    spin_average_application_factor : np.float64
        The spin average application factor for the correct day.
    gradiometer_factor : np.ndarray
        The gradiometer factor for the correct day. Should be size (3,).
    apply_gradiometry : bool
        Whether to apply gradiometry or not. Default is True.
    """

    calibration_offsets: np.ndarray
    mago_calibration: np.ndarray
    magi_calibration: np.ndarray
    spin_count_calibration: int
    quality_flag_threshold: float
    spin_average_application_factor: np.float64
    gradiometer_factor: np.ndarray
    apply_gradiometry: bool = True

    def __init__(self, calibration_dataset: xr.Dataset, day: np.datetime64) -> None:
        """
        Create a MagL1dConfiguration from a calibration dataset and day.

        Parameters
        ----------
        calibration_dataset : xr.Dataset
            The combined calibration dataset from the ancillary files. Created as the
            output from MagAncillaryCombiner, which has day values pointing to the
            calibration file for the given day.
        day : np.datetime64
            The day we are processing, in np.datetime64[D] format.

        """
        self.mago_calibration = retrieve_matrix_from_l2_calibration(
            calibration_dataset, day, use_mago=True
        )

        self.magi_calibration = retrieve_matrix_from_l2_calibration(
            calibration_dataset, day, use_mago=False
        )
        self.calibration_offsets = calibration_dataset.sel(epoch=day)["offsets"].data
        self.spin_count_calibration = calibration_dataset.sel(epoch=day)[
            "number_of_spins"
        ].data
        self.quality_flag_threshold = calibration_dataset.sel(epoch=day)[
            "quality_flag_threshold"
        ].data
        self.spin_average_application_factor = calibration_dataset.sel(epoch=day)[
            "spin_average_application_factor"
        ].data
        self.gradiometer_factor = calibration_dataset.sel(epoch=day)[
            "gradiometer_factor"
        ].data


@dataclass(kw_only=True)
class MagL1d(MagL2L1dBase):  # type: ignore[misc]
    """
    Class for handling IMAP MAG L1d data.

    When the class is created, all the methods are called in the correct order to
    run MAG L1d processing. The resulting instance can then be used to generate an
    xarray dataset with the `generate_dataset` method.

    Example:
        ```
        l1d_norm = MagL1d(
            vectors=mago_vectors,
            epoch=input_mago_norm["epoch"].data,
            range=input_mago_norm["vectors"].data[:, 3],
            global_attributes={},
            quality_flags=np.zeros(len(input_mago_norm["epoch"].data)),
            quality_bitmask=np.zeros(len(input_mago_norm["epoch"].data)),
            data_mode=DataMode.NORM,
            magi_vectors=magi_vectors,
            magi_range=input_magi_norm["vectors"].data[:, 3],
            config=config
        )
        output_dataset = l1d_norm.generate_dataset(attributes, day_to_process)
        ```

    Attributes
    ----------
    magi_vectors : np.ndarray
        The MAGi vectors, shape (N, 3).
    magi_range : np.ndarray
        The MAGi range values, shape (N,).
    magi_epoch : np.ndarray
        The MAGi epoch values, shape (N,).
    config : MagL1dConfiguration
        The configuration for L1d processing, including calibration matrices and
        offsets. This is generated from the input ancillary file and the
        MagL1dConfiguration class.
    spin_offsets : xr.Dataset, optional
        The spin offsets dataset, if already calculated. If not provided, it will be
        calculated during processing if in NORM mode.
    day : np.datetime64
        The day we are processing, in np.datetime64[D] format. This is used to
        truncate the data to exactly 24 hours.
    """

    magi_vectors: np.ndarray
    magi_range: np.ndarray
    magi_epoch: np.ndarray
    config: MagL1dConfiguration
    spin_offsets: xr.Dataset = None
    day: InitVar[np.datetime64]
    data_level: str = field(default="l1d", init=False)

    def __post_init__(self, day: np.datetime64) -> None:
        """
        Run all processing steps to generate L1d data.

        This updates class variables to match L1D outputs.

        Parameters
        ----------
        day : np.datetime64
            The day we are processing, in np.datetime64[D] format. This is used to
            truncate the data to exactly 24 hours.
        """
        # The main data frame is MAGO, even though we have MAGI data included.
        self.frame = ValidFrames.MAGO

        # set the magnitude before truncating
        self.magnitude = np.zeros(self.vectors.shape[0], dtype=np.float64)  # type: ignore[has-type]
        self.truncate_to_24h(day)

        self.vectors, self.magi_vectors = self._calibrate_and_offset_vectors(
            self.config.mago_calibration,
            self.config.magi_calibration,
            self.config.calibration_offsets,
        )
        # We need to be in SRF for the spin offsets application and calculation
        self.rotate_frame(ValidFrames.SRF)

        if self.spin_offsets is None and self.data_mode == DataMode.NORM:
            self.spin_offsets = self.calculate_spin_offsets()

        self.vectors = self.apply_spin_offsets(
            self.spin_offsets,
            self.epoch,  # type: ignore[has-type]
            self.vectors,
            self.config.spin_average_application_factor,
        )
        self.magi_vectors = self.apply_spin_offsets(
            self.spin_offsets,
            self.magi_epoch,
            self.magi_vectors,
            self.config.spin_average_application_factor,
        )

        # we need to be in DSRF for the gradiometry offsets calculation and application
        self.rotate_frame(ValidFrames.DSRF)

        if self.config.apply_gradiometry:
            self.gradiometry_offsets = self.calculate_gradiometry_offsets(
                self.vectors,
                self.epoch,  # type: ignore[has-type]
                self.magi_vectors,
                self.magi_epoch,
                self.config.quality_flag_threshold,
            )
            self.vectors = self.apply_gradiometry_offsets(
                self.gradiometry_offsets, self.vectors, self.config.gradiometer_factor
            )

        self.magnitude = MagL2L1dBase.calculate_magnitude(vectors=self.vectors)
        self.is_l1d = True

    def generate_dataset(
        self,
        attribute_manager: ImapCdfAttributes,
        day: np.datetime64,
    ) -> xr.Dataset:
        """
        Generate an xarray dataset from the dataclass.

        This overrides the parent method to conditionally swap MAGO/MAGI data
        based on the always_output_mago configuration setting, and to construct
        the logical_source_id for L1D files.

        Parameters
        ----------
        attribute_manager : ImapCdfAttributes
            CDF attributes object for the correct level.
        day : np.datetime64
            The 24 hour day to process, as a numpy datetime format.

        Returns
        -------
        xr.Dataset
            Complete dataset ready to write to CDF file.
        """
        always_output_mago = configuration.ALWAYS_OUTPUT_MAGO

        if not always_output_mago:
            # Swap vectors and epochs to use MAGI data instead of MAGO
            original_vectors: np.ndarray = self.vectors.copy()
            original_epoch: np.ndarray = self.epoch.copy()  # type: ignore[has-type]
            original_range: np.ndarray = self.range.copy()  # type: ignore[has-type]

            self.vectors = self.magi_vectors  # type: ignore[no-redef]
            self.epoch = self.magi_epoch  # type: ignore[no-redef]
            self.range = self.magi_range  # type: ignore[no-redef]

            # Call parent generate_dataset method with L1D data level
            dataset = super().generate_dataset(attribute_manager, day)

            # Restore original vectors for any further processing
            self.vectors = original_vectors
            self.epoch = original_epoch
            self.range = original_range
        else:
            # Use MAGO data (default behavior)
            dataset = super().generate_dataset(attribute_manager, day)

        return dataset

    def rotate_frame(self, end_frame: ValidFrames) -> None:
        """
        Rotate the vectors to the desired frame.

        Rotates both the mago vectors (self.vectors) and the magi vectors
        (self.magi_vectors), then set self.frame to end_frame.

        Parameters
        ----------
        end_frame : ValidFrames
            The frame to rotate to. Should be one of the ValidFrames enum.
        """
        # Self.frame should refer to the main data in self.vectors, which is MAGO
        # data. For most frames, MAGO and MAGI are in the same frame, except the
        # instrument reference frame.
        if ValidFrames.MAGI in (self.frame, end_frame):
            raise ValueError(
                "MAGL1d.frame should never be equal to MAGI frame. If the "
                "data is in the instrument frame, use MAGO."
            )

        start_frame = self.frame

        if self.epoch_et is None:
            self.epoch_et: np.ndarray = ttj2000ns_to_et(self.epoch)
            self.magi_epoch_et: np.ndarray = ttj2000ns_to_et(self.magi_epoch)

        self.vectors = frame_transform(
            self.epoch_et,
            self.vectors,
            from_frame=start_frame.spice_frame,
            to_frame=end_frame.spice_frame,
            allow_spice_noframeconnect=True,
        )

        # If we were in MAGO frame, we need to rotate MAGI vectors from MAGI to
        # end_frame
        if start_frame == ValidFrames.MAGO:
            start_frame = ValidFrames.MAGI

        self.magi_vectors = frame_transform(
            self.magi_epoch_et,
            self.magi_vectors,
            from_frame=start_frame.spice_frame,
            to_frame=end_frame.spice_frame,
            allow_spice_noframeconnect=True,
        )

        self.frame = end_frame

    def _calibrate_and_offset_vectors(
        self,
        mago_calibration: np.ndarray,
        magi_calibration: np.ndarray,
        offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply calibration and initial offset calculations from the configuration file.

        Parameters
        ----------
        mago_calibration : np.ndarray
            Calibration matrix for the correct day for MAGo. Should be size (3, 3, 4).
        magi_calibration : np.ndarray
            Calibration matrix for the correct day for MAGi. Should be size (3, 3, 4).
        offsets : np.ndarray
            Offsets for the correct day. Should be size (2, 4, 3) where the first index
            is 0 for MAGo and 1 for MAGi, the second index is the range (0-3), and the
            third index is the axis (0-2).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The calibrated and offset MAGo and MAGi vectors, each shape (N, 3)
            (not including range).
        """
        vectors_plus_range_mago = np.concatenate(
            (self.vectors, self.range[:, np.newaxis]), axis=1
        )

        vectors_plus_range_magi = np.concatenate(
            (self.magi_vectors, self.magi_range[:, np.newaxis]), axis=1
        )

        mago_vectors = MagL2L1dBase.apply_calibration(
            vectors_plus_range_mago, mago_calibration
        )
        magi_vectors = MagL2L1dBase.apply_calibration(
            vectors_plus_range_magi, magi_calibration
        )

        mago_vectors = np.apply_along_axis(
            func1d=self.apply_calibration_offset_single_vector,
            axis=1,
            arr=mago_vectors,
            offsets=offsets,
            is_magi=False,
        )

        magi_vectors = np.apply_along_axis(
            func1d=self.apply_calibration_offset_single_vector,
            axis=1,
            arr=magi_vectors,
            offsets=offsets,
            is_magi=True,
        )

        return mago_vectors[:, :3], magi_vectors[:, :3]

    @staticmethod
    def apply_calibration_offset_single_vector(
        input_vector: np.ndarray, offsets: np.ndarray, is_magi: bool = False
    ) -> np.ndarray:
        """
        Apply the offset to a single vector.

        Parameters
        ----------
        input_vector : np.ndarray
            The input vector to offset, shape (4,) where the last element is the range.
        offsets : np.ndarray
            The offsets array, shape (2, 4, 3) where the first index is 0 for MAGo and
            1 for MAGi, the second index is the range (0-3), and the third index is the
            axis (0-2).

        is_magi : bool
            Whether the input vector is from MAGi (True) or MAGo (False).

        Returns
        -------
        np.ndarray
            The offset vector, shape (4,) where the last element is unchanged.
        """
        # Offsets are in shape (sensor, range, axis)
        updated_vector = input_vector.copy()
        rng = int(input_vector[3])
        x_y_z = input_vector[:3]
        updated_vector[:3] = x_y_z + offsets[int(is_magi), rng, :]
        return updated_vector

    def calculate_spin_offsets(self) -> xr.Dataset:
        """
        Calculate the spin offsets for the current data.

        Algorithm determined by section 7.3.5, step 6 of the algorithm document.

        This should only be called on normal mode data in the SRF frame. It computes
        the average spin during a chunk as specified in the config by
        spin_count_calibration (nominally 240 spins), then creates a dataset containing
        timestamps which correspond to the start of the validity for the offset.

        This is only computed for the x and y axes (indices 0 and 1 of vectors) as the
        z axis is the spinning axis in SRF and should not be affected by spins.

        Any invalid spins are skipped and not included.

        Returns
        -------
        spin_offsets : xr.Dataset
            The spin offsets dataset, with dimensions:
            - epoch: the timestamp where the offset becomes valid
            - x_offset: the x offset values
            - y_offset: the y offset values
        """
        # This needs to only happen for NM data
        if self.data_mode != DataMode.NORM and self.frame != ValidFrames.SRF:
            raise ValueError(
                "Spin offsets can only be calculated in NORM mode and SRF frame."
            )

        epoch_met = ttj2000ns_to_met(self.epoch)
        sc_spin_phase = spin.get_spacecraft_spin_phase(epoch_met)
        # mark vectors as nan where they are nan in sc_spin_phase
        vectors = self.vectors.copy().astype(np.float64)

        vectors[np.isnan(sc_spin_phase), :] = np.nan

        # first timestamp where spin phase is less than the previous value
        # this is when the spin crosses zero
        spin_starts = np.where(np.diff(sc_spin_phase) < 0)[0] + 1

        # if the value switches from nan to a number, or from a number to nan, that
        # is also a spin start
        nan_to_number = np.where(np.diff(np.isnan(sc_spin_phase)) != 0)[0] + 1

        # find the places spins start while skipping over invalid or missing data
        # (marked as nan by get_spacecraft_spin_phase)
        spin_starts = np.sort(np.concatenate((spin_starts, nan_to_number)))

        # Get the expected spin period from the spin table
        # Convert to nanoseconds to match epoch
        spin_data = spin.get_spin_data()
        # Use the median spin period as the expected value
        expected_spin = np.median(spin_data["spin_period_sec"]) * 1e9

        paired_nans = nan_to_number.reshape(-1, 2)

        for start_of_gap, end_of_gap in paired_nans:
            # in nan_to_number, we have the start and end for every nan gap
            # if this gap spans more than 1 spin period, we need to insert
            # additional spin_starts into spin_starts.

            gap_start_time = self.epoch[start_of_gap]
            gap_end_time = self.epoch[end_of_gap]

            # Calculate the number of spins in this gap
            number_of_spins = int((gap_end_time - gap_start_time) // expected_spin)
            if number_of_spins > 1:
                # Insert new spin starts into spin_starts
                for i in range(1, number_of_spins):
                    estimated_start = gap_start_time + i * expected_spin
                    new_spin_index = (np.abs(self.epoch - estimated_start)).argmin()

                    spin_starts = np.append(spin_starts, new_spin_index)

        # Now spin_starts contains all the indices where spins begin, including
        # estimating skipped or missing spins.
        spin_starts = np.sort(spin_starts)

        chunk_start = 0
        offset_epochs = []
        x_avg_calcs: list[np.float64] = []
        y_avg_calcs: list[np.float64] = []
        validity_start_times = []
        validity_end_times = []
        start_spin_counters = []
        end_spin_counters = []

        while chunk_start < len(spin_starts):
            # Take self.spin_count_calibration number of spins and put them into a chunk
            chunk_indices = spin_starts[
                chunk_start : chunk_start + self.config.spin_count_calibration + 1
            ]
            chunk_start_idx = chunk_start

            chunk_vectors = self.vectors[chunk_indices[0] : chunk_indices[-1]]
            chunk_epoch = self.epoch[chunk_indices[0] : chunk_indices[-1]]

            # Check if more than half of the chunk data is NaN before processing
            x_valid_count: int = int(np.sum(~np.isnan(chunk_vectors[:, 0])))
            y_valid_count: int = int(np.sum(~np.isnan(chunk_vectors[:, 1])))
            total_points = len(chunk_vectors)

            # average the x and y axes (z is fixed, as the spin axis)
            avg_x = np.nanmean(chunk_vectors[:, 0])
            avg_y = np.nanmean(chunk_vectors[:, 1])

            # Skip chunk if more than half of x or y data is NaN, or if we have less
            # than half a spin.
            # in this case, we should reuse the previous averages.
            if (
                x_valid_count <= total_points / 2
                or y_valid_count <= total_points / 2
                or total_points <= self.config.spin_count_calibration / 2
            ):
                avg_x = x_avg_calcs[-1] if x_avg_calcs else np.float64(FILLVAL)
                avg_y = y_avg_calcs[-1] if y_avg_calcs else np.float64(FILLVAL)

            if not np.isnan(avg_x) and not np.isnan(avg_y):
                offset_epochs.append(chunk_epoch[0])
                x_avg_calcs.append(avg_x)
                y_avg_calcs.append(avg_y)

                # Add validity time range for this chunk
                validity_start_times.append(chunk_epoch[0])
                validity_end_times.append(chunk_epoch[-1])

                # Add spin counter information
                start_spin_counters.append(chunk_start_idx)
                end_spin_counters.append(
                    min(
                        chunk_start_idx + self.config.spin_count_calibration - 1,
                        len(spin_starts) - 1,
                    )
                )

            chunk_start = chunk_start + self.config.spin_count_calibration

        spin_epoch_dataarray = xr.DataArray(np.array(offset_epochs))

        spin_offsets = xr.Dataset(coords={"epoch": spin_epoch_dataarray})

        spin_offsets["x_offset"] = xr.DataArray(np.array(x_avg_calcs), dims=["epoch"])
        spin_offsets["y_offset"] = xr.DataArray(np.array(y_avg_calcs), dims=["epoch"])
        spin_offsets["validity_start_time"] = xr.DataArray(
            np.array(validity_start_times), dims=["epoch"]
        )
        spin_offsets["validity_end_time"] = xr.DataArray(
            np.array(validity_end_times), dims=["epoch"]
        )
        spin_offsets["start_spin_counter"] = xr.DataArray(
            np.array(start_spin_counters), dims=["epoch"]
        )
        spin_offsets["end_spin_counter"] = xr.DataArray(
            np.array(end_spin_counters), dims=["epoch"]
        )

        return spin_offsets

    def generate_spin_offset_dataset(self) -> xr.Dataset | None:
        """
        Output the spin offsets file as a dataset.

        Returns
        -------
        xr.Dataset | None
            The spin offsets dataset. This function can be used to control the output
            structure of the offsets dataset ancillary file, without affecting how
            the offsets are used inside the class.
        """
        return self.spin_offsets

    @staticmethod
    def apply_spin_offsets(
        spin_offsets: xr.Dataset,
        epoch: np.ndarray,
        vectors: np.ndarray,
        spin_average_application_factor: np.float64,
    ) -> np.ndarray:
        """
        Apply the spin offsets to the input vectors.

        This uses the spin offsets calculated by `calculate_spin_offsets` (or passed in
        to the class in burst mode) to apply the offsets to the input vectors.

        For each vector, we take the nearest offset, multiply it by the
        spin_average_application_factor calibration value, and subtract the offset from
        the appropriate axis.

        These spin offsets act as an automatic smoothing effect on the data over each
        series of spins.

        Parameters
        ----------
        spin_offsets : xr.Dataset
            The spin offsets dataset.
        epoch : np.ndarray
            The epoch values for the input vectors, shape (N,).
        vectors : np.ndarray
            The input vectors to apply offsets to, shape (N, 3). Can be Mago, magi,
            burst or norm. The same offsets file is applied to all.
        spin_average_application_factor : np.float64
            The spin average application factor from the configuration file.

        Returns
        -------
        np.ndarray
            The output vectors with spin offsets applied, shape (N, 3).
        """
        if spin_offsets is None:
            raise ValueError("No spin offsets calculated to apply.")

        output_vectors = np.full(vectors.shape, FILLVAL, dtype=np.float64)

        for index in range(spin_offsets["epoch"].data.shape[0] - 1):
            timestamp = spin_offsets["epoch"].data[index]
            # for the first timestamp, catch all the beginning vectors
            if index == 0:
                timestamp = epoch[0]

            end_timestamp = spin_offsets["epoch"].data[index + 1]

            # for the last timestamp, catch all the ending vectors
            if index + 2 >= len(spin_offsets["epoch"].data):
                end_timestamp = epoch[-1] + 1

            mask = (epoch >= timestamp) & (epoch < end_timestamp)

            mask = mask & (vectors[:, 0] != FILLVAL)

            if not np.any(mask):
                continue

            x_offset = (
                spin_offsets["x_offset"].data[index] * spin_average_application_factor
            )
            y_offset = (
                spin_offsets["y_offset"].data[index] * spin_average_application_factor
            )

            output_vectors[mask, 0] = vectors[mask, 0] - x_offset
            output_vectors[mask, 1] = vectors[mask, 1] - y_offset

        output_vectors[:, 2] = vectors[:, 2]

        return output_vectors

    @staticmethod
    def calculate_gradiometry_offsets(
        mago_vectors: np.ndarray,
        mago_epoch: np.ndarray,
        magi_vectors: np.ndarray,
        magi_epoch: np.ndarray,
        quality_flag_threshold: float = np.inf,
    ) -> xr.Dataset:
        """
        Calculate the gradiometry offsets between MAGo and MAGi.

        This uses linear interpolation to align the MAGi data to the MAGo timestamps,
        then calculates the difference between the two sensors on each axis.

        All vectors must be in the DSRF frame before starting.

        Static method that can be used by i-ALiRT.

        Parameters
        ----------
        mago_vectors : np.ndarray
            The MAGo vectors, shape (N, 3).
        mago_epoch : np.ndarray
            The MAGo epoch values, shape (N,).
        magi_vectors : np.ndarray
            The MAGi vectors, shape (N, 3).
        magi_epoch : np.ndarray
            The MAGi epoch values, shape (N,).
        quality_flag_threshold : np.float64, optional
            Threshold for quality flags. If the magnitude of gradiometer offset
            exceeds this threshold, quality flag will be set. Default is np.inf
            (no quality flags set).

        Returns
        -------
        xr.Dataset
            The gradiometer offsets dataset, with variables:
            - epoch: the timestamp of the MAGo data
            - gradiometer_offsets: the offset values (MAGi - MAGo) for each axis
            - gradiometer_offset_magnitude: magnitude of the offset vector
            - quality_flags: quality flags (1 if magnitude > threshold, 0 otherwise)
        """
        # TODO: should this extrapolate or should non-overlapping data be removed?
        _, aligned_magi = linear(
            magi_vectors,
            magi_epoch,
            mago_epoch,
            extrapolate=True,
        )

        diff = aligned_magi - mago_vectors

        # Calculate magnitude of gradiometer offset for each vector
        magnitude = np.linalg.norm(diff, axis=1)

        # Set quality flags: 0 = good data (below threshold), 1 = bad data
        quality_flags = (magnitude > quality_flag_threshold).astype(int)

        grad_epoch = xr.DataArray(mago_epoch, dims=["epoch"])
        direction = xr.DataArray(["x", "y", "z"], dims=["axis"])
        grad_ds = xr.Dataset(coords={"epoch": grad_epoch, "direction": direction})
        grad_ds["gradiometer_offsets"] = xr.DataArray(diff, dims=["epoch", "direction"])
        grad_ds["gradiometer_offset_magnitude"] = xr.DataArray(
            magnitude, dims=["epoch"]
        )
        grad_ds["quality_flags"] = xr.DataArray(quality_flags, dims=["epoch"])

        return grad_ds

    @staticmethod
    def apply_gradiometry_offsets(
        gradiometry_offsets: xr.Dataset,
        vectors: np.ndarray,
        gradiometer_factor: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the gradiometry offsets to the input vectors.

        Gradiometry epoch and vectors epoch should align (i.e. the vectors should be
        from mago).

        The vectors should be in the DSRF frame.

        Parameters
        ----------
        gradiometry_offsets : xr.Dataset
            The gradiometry offsets dataset, as output by calculate_gradiometry_offsets.
        vectors : np.ndarray
            The input vectors to apply offsets to, shape (N, 3). Should be on the same
            epoch as the gradiometry offsets.
        gradiometer_factor : np.ndarray
            A (3,3) element matrix to scale and rotate the gradiometer offsets.

        Returns
        -------
        np.ndarray
            The output vectors with gradiometry offsets applied, shape (N, 3).
        """
        offset_value = gradiometry_offsets["gradiometer_offsets"].data
        offset_value = np.apply_along_axis(
            np.dot,
            1,
            offset_value,
            gradiometer_factor,
        )

        return vectors - offset_value
