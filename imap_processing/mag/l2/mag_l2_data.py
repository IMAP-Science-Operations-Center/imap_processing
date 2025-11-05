"""Data structures for MAG L2 and L1D processing."""

from dataclasses import InitVar, dataclass, field
from enum import Enum

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import FILLVAL, DataMode
from imap_processing.mag.l1b.mag_l1b import calibrate_vector
from imap_processing.spice.geometry import SpiceFrame, frame_transform
from imap_processing.spice.time import (
    et_to_ttj2000ns,
    str_to_et,
    ttj2000ns_to_et,
)


class ValidFrames(Enum):
    """SPICE reference frames for output."""

    MAGO = SpiceFrame.IMAP_MAG_O
    MAGI = SpiceFrame.IMAP_MAG_I
    DSRF = SpiceFrame.IMAP_DPS
    SRF = SpiceFrame.IMAP_SPACECRAFT
    GSE = SpiceFrame.IMAP_GSE
    GSM = SpiceFrame.IMAP_GSM
    RTN = SpiceFrame.IMAP_RTN


@dataclass(kw_only=True)
class MagL2L1dBase:
    """
    Base class for MAG L2 and L1D data.

    Since these two data levels output identical files, and share some methods, this
    superclass captures the tools in common, while allowing each subclass to define
    individual attributes and algorithms.

    May also be extended for I-ALiRT.

    Attributes
    ----------
    vectors: np.ndarray
        Magnetic field vectors of size (n, 3) where n is the number of vectors.
        Describes (x, y, z) components of the magnetic field. This field is the output
        vectors, which are nominally from the MAGo sensor.
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
    frame:
        The reference frame of the input vectors. Defaults to the MAGO instrument frame.
    epoch_et: np.ndarray
        The epoch timestamps converted to ET format. Used for frame transformations.
        Calculated on first use and then saved. Should not be passed in.
    """

    vectors: np.ndarray
    epoch: np.ndarray
    range: np.ndarray
    global_attributes: dict
    quality_flags: np.ndarray
    quality_bitmask: np.ndarray
    data_mode: DataMode
    magnitude: np.ndarray = field(init=False)
    frame: ValidFrames = ValidFrames.MAGO
    epoch_et: np.ndarray | None = field(init=False, default=None)

    def generate_dataset(
        self,
        attribute_manager: ImapCdfAttributes,
        day: np.datetime64,
    ) -> xr.Dataset:
        """
        Generate an xarray dataset from the dataclass.

        This method can be used for L2 and L1D, since they have extremely similar
        output.

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
        self.truncate_to_24h(day)

        logical_source_id = (
            f"imap_mag_l2_{self.data_mode.value.lower()}-{self.frame.name.lower()}"
        )
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
            attrs=attribute_manager.get_variable_attributes(
                "epoch", check_schema=False
            ),
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
            attrs=attribute_manager.get_variable_attributes("qf_bitmask"),
        )

        quality_bitmask = xr.DataArray(
            self.quality_bitmask,
            name="quality_bitmask",
            dims=["epoch"],
            attrs=attribute_manager.get_variable_attributes("qf"),
        )

        rng = xr.DataArray(
            self.range,
            name="range",
            dims=["epoch"],
            # TODO temp attrs
            attrs=attribute_manager.get_variable_attributes("fill"),
        )

        magnitude = xr.DataArray(
            self.magnitude,
            name="magnitude",
            dims=["epoch"],
            attrs=attribute_manager.get_variable_attributes("fill"),
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

    def truncate_to_24h(self, timestamp: np.datetime64) -> None:
        """
        Truncate all data to a 24 hour period.

        24 hours is given by timestamp in the format YYYYmmdd.

        Parameters
        ----------
        timestamp : str
            Timestamp in the format YYYYMMDD.
        """
        if self.epoch.shape[0] != self.vectors.shape[0]:
            raise ValueError("Timestamps and vectors are not the same shape!")
        start_timestamp_j2000 = et_to_ttj2000ns(str_to_et(str(timestamp)))
        end_timestamp_j2000 = et_to_ttj2000ns(
            str_to_et(str(timestamp + np.timedelta64(1, "D")))
        )

        day_start_index = np.searchsorted(self.epoch, start_timestamp_j2000)
        day_end_index = np.searchsorted(self.epoch, end_timestamp_j2000)

        self.epoch = self.epoch[day_start_index:day_end_index]
        self.vectors = self.vectors[day_start_index:day_end_index, :]
        self.range = self.range[day_start_index:day_end_index]
        self.magnitude = self.magnitude[day_start_index:day_end_index]
        self.quality_flags = self.quality_flags[day_start_index:day_end_index]
        self.quality_bitmask = self.quality_bitmask[day_start_index:day_end_index]

        if self.epoch_et is not None:
            self.epoch_et = self.epoch_et[day_start_index:day_end_index]

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
        return np.linalg.norm(vectors, axis=1)

    @staticmethod
    def apply_calibration(
        vectors: np.ndarray, calibration_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply the calibration matrix to the vectors.

        This works by repeatedly calling the function calibrate_vector on the vectors
        input.

        Parameters
        ----------
        vectors : np.ndarray
            Array of vectors to apply the calibration to, including x,y,z and range.
            Should be of shape (n, 4) where n is the number of vectors.
        calibration_matrix : np.ndarray
            Calibration matrix to apply to the vectors. Should be of shape (3, 3, 4).

        Returns
        -------
        np.ndarray
            Array of calibrated vectors. Should be of shape (n, 4).
        """
        calibrated_vectors = np.apply_along_axis(
            func1d=calibrate_vector,
            axis=1,
            arr=vectors,
            calibration_matrix=calibration_matrix,
        )

        return calibrated_vectors

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

    def rotate_frame(self, end_frame: ValidFrames) -> None:
        """
        Rotate the vector data in the class to the output frame.

        Parameters
        ----------
        end_frame : ValidFrames
            The frame to rotate the data to. Must be one of the ValidFrames enum
            values.
        """
        if self.epoch_et is None:
            self.epoch_et = ttj2000ns_to_et(self.epoch)
        self.vectors = frame_transform(
            self.epoch_et,
            self.vectors,
            from_frame=self.frame.value,
            to_frame=end_frame.value,
            allow_spice_noframeconnect=True,
        )
        self.frame = end_frame


@dataclass(kw_only=True)
class MagL2(MagL2L1dBase):
    """
    Dataclass for MAG L2 data.

    Since L2 and L1D should have the same structure, this can be used for either level.

    Some of the methods are also static, so they can be used in i-ALiRT processing.
    """

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

        offset_vectors: np.ndarray = vectors + offsets

        # Any values where offsets is FILLVAL must also be FILLVAL
        offset_vectors[(offsets == FILLVAL).any(axis=1), :] = FILLVAL
        return offset_vectors
