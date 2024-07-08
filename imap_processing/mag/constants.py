"""Collection of constant types or values for MAG."""

from dataclasses import dataclass
from enum import Enum


class DataMode(Enum):
    """Enum for MAG data modes: burst and normal (BURST + NORM)."""

    BURST = "BURST"
    NORM = "NORM"


class Sensor(Enum):
    """Enum for MAG sensors: raw, MAGo, and MAGi (RAW, MAGO, MAGI)."""

    MAGO = "MAGO"
    MAGI = "MAGI"
    RAW = "RAW"


@dataclass
class MagSensorMode:
    """
    Organize attributes for different kinds of L1A CDF files.

    Generation_date and input_files are added to the raw, mago, and magi attributes.
    The attributes are set based on the data_mode and sensor values.

    Attributes
    ----------
    data_mode : DataMode
        The data mode of the CDF file.
    sensor : Sensor
        The sensor type of the CDF file.

    Methods
    -------
    get_logical_id()
        Return the logical ID for the CDF file.
    """

    data_mode: DataMode
    sensor: Sensor

    def get_logical_id(self) -> str:
        """
        Return the logical ID for the CDF file.

        Returns
        -------
        string
            The logical ID for the CDF file.
        """
        return (
            f"imap_mag_l1a_{self.data_mode.value.lower()}-{self.sensor.value.lower()}"
        )
