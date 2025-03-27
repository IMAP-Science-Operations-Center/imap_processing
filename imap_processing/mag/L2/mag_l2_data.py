from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


@dataclass
class MagL2:
    """
    Dataclass for MAG L2 data.

    Since L2 and L1D should have the same structure, this can be used for either level.

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
    magnitude: np.ndarray = field(init=False)
    is_l1d: bool = False

    def __post_init__(self):
        self.magnitude = self.calculate_magnitude(self.vectors)

    @staticmethod
    def calculate_magnitude(vectors: np.ndarray) -> np.ndarray:
        """
        Given a list of vectors (x, y, z), calculate the magnitude of each vector.

        For an input list of vectors of size (n, 3) returns a list of magnitudes of
        size (n,).

        Parameters
        ----------
        vectors

        Returns
        -------
        """
        return np.zeros(vectors.shape[0])

    def truncate_to_24h(self, timestamp: str):
        """
        Truncate all data to a 24 hour period.

        24 hours is given by timestamp in the format YYYYmmdd.
        """
        pass

    def generate_dataset(self, attribute_manager: ImapCdfAttributes):
        """
        Generate an xarray dataset from the dataclass.

        This method can be used for L2 and L1D, since they have extremely similar
        output.

        Parameters
        ----------
        attribute_manager : ImapCdfAttributes
            CDF attributes object for the correct level.

        Returns
        -------
        xr.Dataset
        Complete dataset ready to write to CDF file.
        """
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
            attrs=attribute_manager.get_variable_attributes("vectors"),
        )

        quality_flags = xr.DataArray(
            self.quality_flags,
            name="quality_flags",
            dims=["epoch"],
            attrs=attribute_manager.get_variable_attributes("mag_flag_attrs"),
        )

        quality_bitmask = xr.DataArray(
            self.quality_flags,
            name="quality_flags",
            dims=["epoch"],
            attrs=attribute_manager.get_variable_attributes("mag_flag_attrs"),
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
            attribute_manager.get_global_attributes() | self.global_attributes
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
