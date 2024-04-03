"""Data classes for storing and processing MAG Level 1A data."""

from dataclasses import dataclass
from math import floor

import numpy as np

MAX_FINE_TIME = 65535  # maximum 16 bit unsigned int


@dataclass
class TimeTuple:
    """
    Class for storing fine time/coarse time for MAG data.

    Course time is mission SCLK in seconds. Fine time is 16bit unsigned sub-second
    counter.
    """

    coarse_time: int
    fine_time: int

    def __add__(self, seconds: float):
        """
        Add a number of seconds to the time tuple.

        Parameters
        ----------
        seconds : float
            Number of seconds to add

        Returns
        -------
        time : TimeTuple
            New time tuple with the current time tuple + seconds.
        """
        # Add whole seconds to coarse time
        coarse = self.coarse_time + floor(seconds)
        # fine time is 1/65535th of a second
        fine = self.fine_time + round((seconds % 1) * MAX_FINE_TIME)

        # If fine is larger than the max, move the excess into coarse time.
        if fine > MAX_FINE_TIME:
            coarse = coarse + floor(fine / MAX_FINE_TIME)
            fine = fine % MAX_FINE_TIME

        return TimeTuple(coarse, fine)


@dataclass
class Vector:
    """
    Data class for storing MAG vector data.

    Attributes
    ----------
    timestamp : TimeTuple
        Time of the vector sample
    vectors : tuple[int, int, int, int]
        Vector sample, containing x,y,z,range
    """

    # TODO: This timestamp should be in J2000/datetime64
    timestamp: TimeTuple
    x: int
    y: int
    z: int
    rng: int

    def __init__(self, vectors, time: TimeTuple):
        self.timestamp = time
        self.x = vectors[0]
        self.y = vectors[1]
        self.z = vectors[2]
        self.rng = vectors[3]


@dataclass
class MagL1a:
    """
    Data class for MAG Level 1A data.

    Attributes
    ----------
    is_mago : bool
        True if the data is from MagO, False if data is from MagI
    active : bool
        True if the sensor is active
    start_time : TimeTuple
        The coarse and fine time for the sensor
    vectors_per_second : int
        Number of vectors per second
    expected_vector_count : int
        Expected number of vectors (vectors_per_second * seconds_of_data)
    seconds_of_data : int
        Number of seconds of data
    SHCOARSE : int
        Mission elapsed time
    vectors : list[Vector]
        List of magnetic vector samples, starting at start_time

    """

    is_mago: bool
    active: bool
    start_time: TimeTuple
    vectors_per_second: int
    expected_vector_count: int
    seconds_of_data: int
    SHCOARSE: int
    vectors: list

    def __post_init__(self):
        """
        Convert the vector list to a vector list with timestamps associated.

        The first vector starts at start_time, then each subsequent vector time is
        computed by adding 1/vectors_per_second to the previous vector's time.

        This replaces self.vectors with a list of Vector objects.
        """
        sample_time_interval = 1 / self.vectors_per_second
        current_time = self.start_time
        for index, vector in enumerate(self.vectors):
            self.vectors[index] = Vector(vector, current_time)
            current_time = self.vectors[index].timestamp + sample_time_interval

    @staticmethod
    def process_vector_data(
        vector_data: np.ndarray, primary_count: int, secondary_count: int
    ) -> (list[tuple], list[tuple]):
        """
        Given raw packet data, process into Vectors.

        Vectors are grouped into primary sensor and secondary sensor, and returned as a
        tuple (primary sensor vectors, secondary sensor vectors)

        Written by MAG instrument team

        Parameters
        ----------
        vector_data : np.ndarray
            Raw vector data, in bytes. Contains both primary and secondary vector data
            (first primary, then secondary)
        primary_count : int
            Count of the number of primary vectors
        secondary_count : int
            Count of the number of secondary vectors

        Returns
        -------
        (primary, secondary)
            Two arrays, each containing tuples of (x, y, z, sample_range) for each
            vector sample.
        """

        # TODO: error handling
        def to_signed16(n):
            n = n & 0xFFFF
            return n | (-(n & 0x8000))

        pos = 0
        primary_vectors = []
        secondary_vectors = []

        # Since the vectors are stored as 50 bit chunks but accessed via hex (4 bit
        # chunks) there is some shifting required for processing the bytes.
        # However, from a bit processing perspective, the first 48 bits of each 50 bit
        # chunk corresponds to 3 16 bit signed integers. The last 2 bits are the sensor
        # range.

        for i in range(primary_count + secondary_count):  # 0..63 say
            x, y, z, rng = 0, 0, 0, 0
            if i % 4 == 0:  # start at bit 0, take 8 bits + 8bits
                # pos = 0, 25, 50...
                x = (
                    ((vector_data[pos + 0] & 0xFF) << 8)
                    | ((vector_data[pos + 1] & 0xFF) << 0)
                ) & 0xFFFF
                y = (
                    ((vector_data[pos + 2] & 0xFF) << 8)
                    | ((vector_data[pos + 3] & 0xFF) << 0)
                ) & 0xFFFF
                z = (
                    ((vector_data[pos + 4] & 0xFF) << 8)
                    | ((vector_data[pos + 5] & 0xFF) << 0)
                ) & 0xFFFF
                rng = (vector_data[pos + 6] >> 6) & 0x3
                pos += 6
            elif i % 4 == 1:  # start at bit 2, take 6 bits, 8 bit, 2 bits per vector
                # pos = 6, 31...
                x = (
                    ((vector_data[pos + 0] & 0x3F) << 10)
                    | ((vector_data[pos + 1] & 0xFF) << 2)
                    | ((vector_data[pos + 2] >> 6) & 0x03)
                ) & 0xFFFF
                y = (
                    ((vector_data[pos + 2] & 0x3F) << 10)
                    | ((vector_data[pos + 3] & 0xFF) << 2)
                    | ((vector_data[pos + 4] >> 6) & 0x03)
                ) & 0xFFFF
                z = (
                    ((vector_data[pos + 4] & 0x3F) << 10)
                    | ((vector_data[pos + 5] & 0xFF) << 2)
                    | ((vector_data[pos + 6] >> 6) & 0x03)
                ) & 0xFFFF
                rng = (vector_data[pos + 6] >> 4) & 0x3
                pos += 6
            elif i % 4 == 2:  # start at bit 4, take 4 bits, 8 bits, 4 bits per vector
                # pos = 12, 37...
                x = (
                    ((vector_data[pos + 0] & 0x0F) << 12)
                    | ((vector_data[pos + 1] & 0xFF) << 4)
                    | ((vector_data[pos + 2] >> 4) & 0x0F)
                ) & 0xFFFF
                y = (
                    ((vector_data[pos + 2] & 0x0F) << 12)
                    | ((vector_data[pos + 3] & 0xFF) << 4)
                    | ((vector_data[pos + 4] >> 4) & 0x0F)
                ) & 0xFFFF
                z = (
                    ((vector_data[pos + 4] & 0x0F) << 12)
                    | ((vector_data[pos + 5] & 0xFF) << 4)
                    | ((vector_data[pos + 6] >> 4) & 0x0F)
                ) & 0xFFFF
                rng = (vector_data[pos + 6] >> 2) & 0x3
                pos += 6
            elif i % 4 == 3:  # start at bit 6, take 2 bits, 8 bits, 6 bits per vector
                # pos = 18, 43...
                x = (
                    ((vector_data[pos + 0] & 0x03) << 14)
                    | ((vector_data[pos + 1] & 0xFF) << 6)
                    | ((vector_data[pos + 2] >> 2) & 0x3F)
                ) & 0xFFFF
                y = (
                    ((vector_data[pos + 2] & 0x03) << 14)
                    | ((vector_data[pos + 3] & 0xFF) << 6)
                    | ((vector_data[pos + 4] >> 2) & 0x3F)
                ) & 0xFFFF
                z = (
                    ((vector_data[pos + 4] & 0x03) << 14)
                    | ((vector_data[pos + 5] & 0xFF) << 6)
                    | ((vector_data[pos + 6] >> 2) & 0x3F)
                ) & 0xFFFF
                rng = (vector_data[pos + 6] >> 0) & 0x3
                pos += 7

            vector = (to_signed16(x), to_signed16(y), to_signed16(z), rng)
            if i < primary_count:
                primary_vectors.append(vector)
            else:
                secondary_vectors.append(vector)

        return (primary_vectors, secondary_vectors)
