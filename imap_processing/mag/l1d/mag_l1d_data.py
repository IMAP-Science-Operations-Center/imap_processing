from dataclasses import InitVar, dataclass

import numpy as np

from imap_processing.mag.l2.mag_l2_data import MagL2L1DBase


@dataclass(kw_only=True)
class MagL1d(MagL2L1DBase):
    magi_vectors: np.ndarray
    mago_calibration: InitVar[np.ndarray] = None
    magi_calibration: InitVar[np.ndarray] = None
    offsets: InitVar[np.ndarray] = None
    timedelta: InitVar[np.ndarray] = None

    def __post_init__(
        self,
        mago_calibration: np.ndarray,
        magi_calibration: np.ndarray,
        offsets: np.ndarray,
        timedelta: np.ndarray,
    ):
        # Before gradiometry mode, we want to convert to the spacecraft frame

        offsets = self.calculate_offsets()

        self.is_l1d = True

    def calculate_offsets(self) -> np.ndarray:
        return None
