"""Data classes for MAG L1D processing."""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from imap_processing.mag.constants import FILLVAL, DataMode
from imap_processing.mag.l2.mag_l2 import retrieve_matrix_from_l2_calibration
from imap_processing.mag.l2.mag_l2_data import MagL2L1dBase, ValidFrames
from imap_processing.spice import spin


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
    offsets : np.ndarray
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
    """

    offsets: np.ndarray
    mago_calibration: np.ndarray
    magi_calibration: np.ndarray
    spin_count_calibration: int
    quality_flag_threshold: np.float64
    spin_average_application_factor: np.float64
    gradiometer_factor: np.ndarray

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
        self.offsets = calibration_dataset.sel(epoch=day)["offsets"].data
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
    config : MagL1dConfiguration
        The configuration for L1d processing, including calibration matrices and
        offsets. This is generated from the input ancillary file and the
        MagL1dConfiguration class.
    spin_offsets : xr.Dataset, optional
        The spin offsets dataset, if already calculated. If not provided, it will be
        calculated during processing if in NORM mode.
    """

    # TODO magi epoch
    magi_vectors: np.ndarray
    magi_range: np.ndarray
    config: MagL1dConfiguration
    spin_offsets: xr.Dataset = None

    def __post_init__(self) -> None:
        """
        Run all processing steps to generate L1d data.

        This updates class variables to match L1D outputs.
        """
        self.vectors, self.magi_vectors = self._calibrate_and_offset_vectors(
            self.config.mago_calibration,
            self.config.magi_calibration,
            self.config.offsets,
        )

        # Before gradiometry mode, we want to convert to the spacecraft frame
        self.rotate_frame(ValidFrames.SRF)

        if self.spin_offsets is None and self.data_mode == DataMode.NORM:
            self.spin_offsets = self.calculate_spin_offsets()

        self.vectors = self.apply_spin_offsets(self.vectors)
        self.magi_vectors = self.apply_spin_offsets(self.magi_vectors)

        self.magnitude = MagL2L1dBase.calculate_magnitude(vectors=self.vectors)
        self.is_l1d = True

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
            func1d=self.offset_vector,
            axis=1,
            arr=mago_vectors,
            offsets=offsets,
            is_magi=False,
        )

        magi_vectors = np.apply_along_axis(
            func1d=self.offset_vector,
            axis=1,
            arr=magi_vectors,
            offsets=offsets,
            is_magi=True,
        )

        return mago_vectors[:, :3], magi_vectors[:, :3]

    @staticmethod
    def offset_vector(
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
        updated_vector = input_vector.copy().astype(np.int64)
        range = int(input_vector[3])
        x_y_z = input_vector[:3]
        updated_vector[:3] = x_y_z - offsets[int(is_magi), range, :]
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

        # TODO: get the spin numbers which correspond to the epoch values for output
        sc_spin_phase = spin.get_spacecraft_spin_phase(self.epoch)
        # mark vectors as nan where they are nan in sc_spin_phase
        vectors = self.vectors.copy().astype(np.float64)

        vectors[np.isnan(sc_spin_phase), :] = np.nan

        # TODO: currently fully skipping spins with no valid data (not including
        #  them in the averaging OR IN SPIN COUNTING!) is this correct?

        # first timestamp where spin phase is less than the previous value
        # this is when the spin crosses zero
        spin_starts = np.where(np.diff(sc_spin_phase) < 0)[0] + 1

        # if the value switches from nan to a number, that is also a spin start (for an
        # invalid spin)
        nan_to_number = (
            np.where(np.isnan(sc_spin_phase[:-1]) & ~np.isnan(sc_spin_phase[1:]))[0] + 1
        )

        # find the places spins start while skipping over invalid or missing data
        # (marked as nan by get_spacecraft_spin_phase)
        spin_starts = np.sort(np.concatenate((spin_starts, nan_to_number)))

        chunk_start = 0
        offset_epochs = []
        x_avg = []
        y_avg = []
        while chunk_start < len(spin_starts):
            # Take self.spin_count_calibration number of spins and put them into a chunk
            chunk_indices = spin_starts[
                chunk_start : chunk_start + self.config.spin_count_calibration + 1
            ]
            chunk_start = chunk_start + self.config.spin_count_calibration

            # If we are in the end of the chunk, just grab all remaining data
            if chunk_start >= len(spin_starts):
                chunk_indices = np.append(chunk_indices, len(self.epoch))

            chunk_vectors = self.vectors[chunk_indices[0] : chunk_indices[-1]]
            chunk_epoch = self.epoch[chunk_indices[0] : chunk_indices[-1]]

            # average the x and y axes (z is fixed, as the spin axis)
            # TODO: is z the correct axis here?
            avg_x = np.nanmean(chunk_vectors[:, 0])
            avg_y = np.nanmean(chunk_vectors[:, 1])

            if not np.isnan(avg_x) and not np.isnan(avg_y):
                offset_epochs.append(chunk_epoch[0])
                x_avg.append(avg_x)
                y_avg.append(avg_y)

        spin_epoch_dataarray = xr.DataArray(np.array(offset_epochs))

        spin_offsets = xr.Dataset(coords={"epoch": spin_epoch_dataarray})

        spin_offsets["x_offset"] = xr.DataArray(np.array(x_avg), dims=["epoch"])
        spin_offsets["y_offset"] = xr.DataArray(np.array(y_avg), dims=["epoch"])

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

    def apply_spin_offsets(self, vectors: np.ndarray) -> np.ndarray:
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
        vectors : np.ndarray
            The input vectors to apply offsets to, shape (N, 3). Can be Mago, magi,
            burst or norm. The same offsets file is applied to all.

        Returns
        -------
        np.ndarray
            The output vectors with spin offsets applied, shape (N, 3).
        """
        if self.spin_offsets is None:
            raise ValueError("No spin offsets calculated to apply.")

        output_vectors = np.full(vectors.shape, FILLVAL, dtype=np.int64)

        for index in range(self.spin_offsets["epoch"].data.shape[0] - 1):
            timestamp = self.spin_offsets["epoch"].data[index]
            # for the first timestamp, catch all the beginning vectors
            if index == 0:
                timestamp = self.epoch[0]

            end_timestamp = self.spin_offsets["epoch"].data[index + 1]

            # for the last timestamp, catch all the ending vectors
            if index + 2 >= len(self.spin_offsets["epoch"].data):
                end_timestamp = self.epoch[-1] + 1

            mask = (self.epoch >= timestamp) & (self.epoch < end_timestamp)

            mask = mask & (vectors[:, 0] != FILLVAL)

            if not np.any(mask):
                continue

            # TODO: should vectors be a float?
            x_offset = (
                self.spin_offsets["x_offset"].data[index]
                * self.config.spin_average_application_factor
            )
            y_offset = (
                self.spin_offsets["y_offset"].data[index]
                * self.config.spin_average_application_factor
            )

            output_vectors[mask, 0] = self.vectors[mask, 0] - x_offset
            output_vectors[mask, 1] = self.vectors[mask, 1] - y_offset

        output_vectors[:, 2] = vectors[:, 2]

        return output_vectors
