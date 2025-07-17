from dataclasses import InitVar, dataclass, field

import numpy as np

from imap_processing.mag.constants import DataMode
from imap_processing.mag.l2.mag_l2_data import MagL2L1dBase, ValidFrames
from imap_processing.spice import geometry, spin


@dataclass(kw_only=True)
class MagL1d(MagL2L1dBase):
    magi_vectors: np.ndarray
    magi_range: np.ndarray
    offsets: InitVar[np.ndarray] = None
    mago_calibration: InitVar[np.ndarray] = None
    magi_calibration: InitVar[np.ndarray] = None
    sensor: int = field(init=False)
    spin_offsets = None

    def __post_init__(
            self,
            offsets: np.ndarray,
            mago_calibration: np.ndarray = None,
            magi_calibration: np.ndarray = None,
    ):
        self.vectors, self.magi_vectors = self._calibrate_and_offset_vectors(
            mago_calibration, magi_calibration, offsets)

        # Before gradiometry mode, we want to convert to the spacecraft frame
        self.rotate_frame(ValidFrames.SRF)

        if self.spin_offsets is None and self.data_mode == DataMode.NORM:
            self.spin_offsets = self.calculate_spin_offsets()

        # 0 for NORM, 1 for BURST
        self.sensor = int(self.data_mode == DataMode.NORM)
        self.magnitude = MagL2L1dBase.calculate_magnitude(vectors=self.vectors)
        self.is_l1d = True

    def _calibrate_and_offset_vectors(self, mago_calibration, magi_calibration,
                                      offsets) -> tuple[np.ndarray, np.ndarray]:
        vectors_plus_range_mago = np.concatenate(
            (self.vectors, self.range[:, np.newaxis]), axis=1
        )

        vectors_plus_range_magi = np.concatenate(
            (self.magi_vectors, self.magi_range[:, np.newaxis]), axis=1
        )

        mago_vectors = MagL2L1dBase.apply_calibration(vectors_plus_range_mago,
                                                      mago_calibration)
        magi_vectors = MagL2L1dBase.apply_calibration(vectors_plus_range_magi,
                                                      magi_calibration)

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
            raise ValueError("Spin offsets can only be calculated in NORM mode and SRF frame.")

        sc_spin_phase = spin.get_spacecraft_spin_phase(self.epoch)

        raise NotImplementedError
