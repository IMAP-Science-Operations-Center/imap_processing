from abc import ABC, abstractmethod
from typing import Tuple, List


class BinScheme(ABC):

    def __init__(self):
        self._frame = "foo"

    @abstractmethod
    def bin_number(self, longitude: float, latitude: float) -> int:
        pass

    @abstractmethod
    def center_position(self, bin_number:int) -> Tuple[float, float]:
        pass

    @abstractmethod
    def nbins(self) -> int:
        pass

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = frame
