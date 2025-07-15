from dataclasses import InitVar, dataclass, field

import numpy as np

from imap_processing.mag.constants import DataMode
from imap_processing.mag.l2.mag_l2_data import MagL2L1dBase


@dataclass(kw_only=True)
class MagL1d(MagL2L1dBase):
    magi_vectors: np.ndarray
    magi_range: np.ndarray
    offsets: InitVar[np.ndarray] = None
    mago_calibration: InitVar[np.ndarray] = None
    magi_calibration: InitVar[np.ndarray] = None
    sensor: int = field(init=False)


    def __post_init__(
        self,
        offsets: np.ndarray,
        mago_calibration: np.ndarray = None,
        magi_calibration: np.ndarray = None,
    ):
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

        self.vectors = mago_vectors
        # Before gradiometry mode, we want to convert to the spacecraft frame
        # 0 for NORM, 1 for BURST
        sensor = int(self.data_mode == DataMode.NORM)
        self.magnitude = MagL2L1dBase.calculate_magnitude(vectors=self.vectors)
        self.is_l1d = True

    def apply_offsets(self) -> np.ndarray:
        # Offsets are in shape (sensor, range, axis)
        return  []
