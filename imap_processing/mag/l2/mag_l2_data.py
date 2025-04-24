"""Data structures for MAG L2 and L1D processing."""

from dataclasses import InitVar, dataclass, field
from enum import Enum

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import DataMode


class ValidFrames(Enum):
    """SPICE reference frames for output."""

    dsrf = "dsrf"
    srf = "srf"
    rtn = "rtn"
    gse = "gse"


@dataclass
class MagL2:
    """
    Dataclass for MAG L2 data.

    Since L2 and L1D should have the same structure, this can be used for either level.

    Some of the methods are also static, so they can be used in i-ALiRT processing.

    Attributes
    ----------
    vectors: np.ndarray
        Magnetic field vectors of size (n, 3) where n is the number of vectors.
        Describes (x, y, z) components of the magnetic field.
    epoch: np.ndarray
        Time of each vector in J2000 seconds. Should be of length n.
    range: np.ndarray
        Range of each vector. Should be of length n.
    global_attributes: dict
        Any global attributes we want to carry forward into the output CDF file.
    quality_flags: np.ndarray
        Quality flags for each vector. Should be of length n.
    quality_bitmask: np.ndarray
        Quality bitmask for each vector. Should be of length n. Copied from offset
        file in L2, marked as good always in L1D.
    magnitude: np.ndarray
        Magnitude of each vector. Should be of length n. Calculated from L2 vectors.
    is_l1d: bool
        Flag to indicate if the data is L1D. Defaults to False.
    """

    vectors: np.ndarray
    epoch: np.ndarray
    range: np.ndarray
    global_attributes: dict
    quality_flags: np.ndarray
    quality_bitmask: np.ndarray
    data_mode: DataMode
    magnitude: np.ndarray = field(init=False)
    is_l1d: bool = False
    offsets: InitVar[np.ndarray] = None
    timedelta: InitVar[np.ndarray] = None

    def __post_init__(self, offsets: np.ndarray, timedelta: np.ndarray) -> None:
        """
        Calculate the magnitude of the vectors after initialization.

        Parameters
        ----------
        offsets : np.ndarray
            Offsets to apply to the vectors. Should be of shape (n, 3) where n is the
            number of vectors.
        timedelta : np.ndarray
            Time deltas to shift the timestamps by. Should be of length n.
            Given in seconds.
        """
        if offsets is not None:
            self.vectors = self.apply_offsets(self.vectors, offsets)
        if timedelta is not None:
            self.epoch = self.shift_timestamps(self.epoch, timedelta)

        self.magnitude = self.calculate_magnitude(self.vectors)

    @staticmethod
    def calculate_magnitude(
        vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Given a list of vectors (x, y, z), calculate the magnitude of each vector.

        For an input list of vectors of size (n, 3) returns a list of magnitudes of
        size (n,).

        Parameters
        ----------
        vectors : np.ndarray
            Array of vectors to calculate the magnitude of.

        Returns
        -------
        np.ndarray
            Array of magnitudes of the input vectors.
        """
        return np.zeros(vectors.shape[0])  # type: ignore

    @staticmethod
    def apply_offsets(vectors: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """
        Apply the offsets to the vectors by adding them together.

        These offsets are used to shift the vectors in the x, y, and z directions.
        They can either be provided through a custom offsets datafile, or calculated
        using a gradiometry algorithm.

        Parameters
        ----------
        vectors : np.ndarray
            Array of vectors to apply the offsets to. Should be of shape (n, 3) where n
            is the number of vectors.
        offsets : np.ndarray
            Array of offsets to apply to the vectors. Should be of shape (n, 3) where n
            is the number of vectors.

        Returns
        -------
        np.ndarray
            Array of vectors with offsets applied. Should be of shape (n, 3).
        """
        if vectors.shape[0] != offsets.shape[0]:
            raise ValueError("Vectors and offsets must have the same length.")

        offset_vectors: np.ndarray = vectors[:, :3] + offsets

        # TODO: CDF files don't have NaNs. Emailed MAG to ask what this will look like.
        # Any values where offsets is nan must also be nan
        offset_vectors[np.isnan(offsets).any(axis=1)] = np.nan

        return offset_vectors

    @staticmethod
    def shift_timestamps(epoch: np.ndarray, timedelta: np.ndarray) -> np.ndarray:
        """
        Shift the timestamps by the given timedelta.

        If timedelta is positive, the epochs are shifted forward in time.

        Parameters
        ----------
        epoch : np.ndarray
            Array of timestamps to shift. Should be of length n.
        timedelta : np.ndarray
            Array of time deltas to shift the timestamps by. Should be the same length
            as epoch. Given in seconds.

        Returns
        -------
        np.ndarray
            Shifted timestamps.
        """
        if epoch.shape[0] != timedelta.shape[0]:
            raise ValueError(
                "Input Epoch and offsets timedeltas must be the same length."
            )

        timedelta_ns = timedelta * 1e9
        shifted_timestamps = epoch + timedelta_ns
        return shifted_timestamps

    def truncate_to_24h(self, timestamp: str) -> None:
        """
        Truncate all data to a 24 hour period.

        24 hours is given by timestamp in the format YYYYmmdd.

        Parameters
        ----------
        timestamp : str
            Timestamp in the format YYYYMMDD.
        """
        pass

    def generate_dataset(
        self,
        attribute_manager: ImapCdfAttributes,
        frame: ValidFrames = ValidFrames.dsrf,
    ) -> xr.Dataset:
        """
        Generate an xarray dataset from the dataclass.

        This method can be used for L2 and L1D, since they have extremely similar
        output.

        Parameters
        ----------
        attribute_manager : ImapCdfAttributes
            CDF attributes object for the correct level.
        frame : ValidFrames
            SPICE reference frame to rotate the data into.

        Returns
        -------
        xr.Dataset
            Complete dataset ready to write to CDF file.
        """
        logical_source_id = f"imap_mag_l2_{self.data_mode.value.lower()}-{frame.name}"
        direction = xr.DataArray(
            np.arange(3),
            name="direction",
            dims=["direction"],
            attrs=attribute_manager.get_variable_attributes(
                "direction_attrs", check_schema=False
            ),
        )

        direction_label = xr.DataArray(
            direction.values.astype(str),
            name="direction_label",
            dims=["direction_label"],
            attrs=attribute_manager.get_variable_attributes(
                "direction_label", check_schema=False
            ),
        )

        epoch_time = xr.DataArray(
            self.epoch,
            name="epoch",
            dims=["epoch"],
            attrs=attribute_manager.get_variable_attributes("epoch"),
        )

        vectors = xr.DataArray(
            self.vectors,
            name="vectors",
            dims=["epoch", "direction"],
            attrs=attribute_manager.get_variable_attributes("vector_attrs"),
        )

        quality_flags = xr.DataArray(
            self.quality_flags,
            name="quality_flags",
            dims=["epoch"],
            attrs=attribute_manager.get_variable_attributes("compression"),
        )

        quality_bitmask = xr.DataArray(
            self.quality_flags,
            name="quality_flags",
            dims=["epoch"],
            attrs=attribute_manager.get_variable_attributes("compression"),
        )

        rng = xr.DataArray(
            self.range,
            name="range",
            dims=["epoch"],
            # TODO temp attrs
            attrs=attribute_manager.get_variable_attributes("compression_width"),
        )

        magnitude = xr.DataArray(
            self.magnitude,
            name="magnitude",
            dims=["epoch"],
            attrs=attribute_manager.get_variable_attributes("compression_width"),
        )

        global_attributes = (
            attribute_manager.get_global_attributes(logical_source_id)
            | self.global_attributes
        )

        output = xr.Dataset(
            coords={
                "epoch": epoch_time,
                "direction": direction,
                "direction_label": direction_label,
            },
            attrs=global_attributes,
        )

        output["vectors"] = vectors
        output["quality_flags"] = quality_flags
        output["quality_bitmask"] = quality_bitmask
        output["range"] = rng
        output["magnitude"] = magnitude

        return output
