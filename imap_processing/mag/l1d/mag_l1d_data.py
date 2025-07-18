from dataclasses import InitVar, dataclass, field

import numpy as np

from imap_processing.mag.constants import DataMode, Sensor, FILLVAL
from imap_processing.mag.l2.mag_l2 import retrieve_matrix_from_l2_calibration
from imap_processing.mag.l2.mag_l2_data import MagL2L1dBase, ValidFrames
from imap_processing.spice import spin
import xarray as xr


@dataclass
class MagL1dConfiguration:
    offsets: np.ndarray
    mago_calibration: np.ndarray
    magi_calibration: np.ndarray
    spin_count_calibration: int
    quality_flag_threshold: np.float64
    spin_average_application_factor: np.float64
    gradiometer_factor: np.ndarray

    def __init__(self, calibration_dataset: xr.Dataset, day):
        """
        Create a MagL1dConfiguration from a calibration dataset and day.

        Parameters
        ----------
        calibration_dataset
        day
        """
        self.mago_calibration = retrieve_matrix_from_l2_calibration(
            calibration_dataset, day, use_mago=True
        )

        self.magi_calibration = retrieve_matrix_from_l2_calibration(
            calibration_dataset, day, use_mago=False
        )
        self.offsets = calibration_dataset.sel(epoch=day)["offsets"].data
        self.spin_count_calibration = calibration_dataset.sel(epoch=day)["number_of_spins"].data
        self.quality_flag_threshold = calibration_dataset.sel(epoch=day)["quality_flag_threshold"].data
        self.spin_average_application_factor = calibration_dataset.sel(epoch=day)["spin_average_application_factor"].data
        self.gradiometer_factor = calibration_dataset.sel(epoch=day)["gradiometer_factor"].data


@dataclass(kw_only=True)
class MagL1d(MagL2L1dBase):
    magi_vectors: np.ndarray
    magi_range: np.ndarray
    config: MagL1dConfiguration
    spin_offsets: xr.Dataset = None

    def __post_init__(self):
        self.vectors, self.magi_vectors = self._calibrate_and_offset_vectors(
            self.config.mago_calibration, self.config.magi_calibration, self.config.offsets
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
        self, mago_calibration, magi_calibration, offsets
    ) -> tuple[np.ndarray, np.ndarray]:
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
    def offset_vector(input_vector, offsets, is_magi=False) -> np.ndarray:
        # Offsets are in shape (sensor, range, axis)
        updated_vector = input_vector.copy().astype(np.int64)
        range = int(input_vector[3])
        x_y_z = input_vector[:3]
        updated_vector[:3] = x_y_z - offsets[int(is_magi), range, :]
        return updated_vector

    def calculate_spin_offsets(self):

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
        nan_to_number = np.where(np.isnan(sc_spin_phase[:-1]) & ~np.isnan(sc_spin_phase[1:]))[0] + 1

        # find the places spins start while skipping over invalid or missing data
        # (marked as nan by get_spacecraft_spin_phase)
        spin_starts = np.sort(np.concatenate(
            (spin_starts, nan_to_number)
        ))

        chunk_start = 0
        offset_epochs = []
        x_avg = []
        y_avg = []
        while chunk_start < len(spin_starts):
            # Take self.spin_count_calibration number of spins and put them into a chunk
            chunk_indices = spin_starts[chunk_start:chunk_start + self.config.spin_count_calibration + 1]
            chunk_start = chunk_start + self.config.spin_count_calibration

            # If we are in the end of the chunk, just grab all remaining data
            if chunk_start >= len(spin_starts):
                chunk_indices = np.append(chunk_indices, len(self.epoch))

            chunk_vectors = self.vectors[chunk_indices[0]:chunk_indices[-1]]
            chunk_epoch = self.epoch[chunk_indices[0]:chunk_indices[-1]]

            # average the x and y axes (z is fixed, as the spin axis)
            # TODO: is z the correct axis here?
            avg_x = np.nanmean(chunk_vectors[:, 0])
            avg_y = np.nanmean(chunk_vectors[:, 1])

            if avg_x is not np.nan and avg_y is not np.nan:
                offset_epochs.append(chunk_epoch[0])
                x_avg.append(avg_x)
                y_avg.append(avg_y)

        spin_epoch_dataarray = xr.DataArray(np.array(offset_epochs))

        spin_offsets = xr.Dataset(coords={"epoch": spin_epoch_dataarray})

        spin_offsets["x_offset"] = xr.DataArray(np.array(x_avg), dims=["epoch"])
        spin_offsets["y_offset"] = xr.DataArray(np.array(y_avg), dims=["epoch"])

        return spin_offsets

    def generate_spin_offset_dataset(self):
        """
        Output the spin offsets file as a dataset.

        Returns
        -------
        xr.Dataset
            The spin offsets dataset.
        """
        return self.spin_offsets

    def apply_spin_offsets(self, vectors) -> np.ndarray:
        if self.spin_offsets is None:
            raise ValueError("No spin offsets calculated to apply.")

        output_vectors = np.full(vectors.shape, FILLVAL, dtype=np.int64)

        for index, timestamp in enumerate(self.spin_offsets['epoch'].data[:-1]):
            # for the first timestamp, catch all the beginning vectors
            if index == 0:
                timestamp = self.epoch[0]

            end_timestamp = self.spin_offsets['epoch'].data[index + 1]

            # for the last timestamp, catch all the ending vectors
            if index + 2 >= len(self.spin_offsets['epoch'].data):
                end_timestamp = self.epoch[-1] + 1

            mask = (self.epoch >= timestamp) & (self.epoch < end_timestamp)

            mask = mask & (vectors[:, 0] != FILLVAL)

            if not np.any(mask):
                continue

            x_offset = self.spin_offsets['x_offset'].data[index] * self.config.spin_average_application_factor
            y_offset = self.spin_offsets['y_offset'].data[index] * self.config.spin_average_application_factor

            output_vectors[mask, 0] = self.vectors[mask, 0] - x_offset
            output_vectors[mask, 1] = self.vectors[mask, 1] - y_offset

        output_vectors[:, 2] = vectors[:, 2]

        return output_vectors







