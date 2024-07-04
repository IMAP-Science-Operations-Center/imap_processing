"""Data classes for storing and processing MAG Level 1A data."""

from dataclasses import dataclass
from math import floor

import numpy as np

from imap_processing.cdf.utils import met_to_j2000ns

MAX_FINE_TIME = 65535  # maximum 16 bit unsigned int


@dataclass
class TimeTuple:
    """
    Class for storing fine time/coarse time for MAG data.

    Course time is mission SCLK in seconds. Fine time is 16bit unsigned sub-second
    counter.

    Attributes
    ----------
    coarse_time : int
        Coarse time in seconds.
    fine_time : int
        Subsecond.

    Methods
    -------
    to_seconds()
    """

    coarse_time: int
    fine_time: int

    def __add__(self, seconds: float):  # type: ignore[no-untyped-def]
        # ruff is saying TimeTuple is undefined for this usage.
        """
        Add a number of seconds to the time tuple.

        Parameters
        ----------
        seconds : float
            Number of seconds to add.

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

    def to_seconds(self) -> float:
        """
        Convert time tuple into seconds (float).

        Returns
        -------
        seconds : float
            Time in seconds.
        """
        return self.coarse_time + self.fine_time / MAX_FINE_TIME


@dataclass
class MagL1a:
    """
    Data class for MAG Level 1A data.

    TODO: This object should end up having 1 days worth of data.

    One MAG L1A object corresponds to part of one MAG L0 packet, which corresponds to
    one packet of data from the MAG instrument. Each L0 packet consists of data from
    two sensors, MAGO (outboard) and MAGI (inboard). One of these sensors is designated
    as the primary sensor (first part of data stream), and one as the secondary.

    We expect the primary sensor to be MAGO, and the secondary to be MAGI, but this is
    not guaranteed. Each MagL1A object contains data from one sensor. The
    primary/secondary construct is only used to sort the vectors into MAGo and MAGi
    data, and therefore is not used at higher levels.

    Attributes
    ----------
    is_mago : bool
        True if the data is from MagO, False if data is from MagI.
    active : bool
        True if the sensor is active.
    SHCOARSE : int
        Mission elapsed time.
    vectors : numpy.ndarray
        List of magnetic vector samples, starting at start_time. [x, y, z, range, time],
        where time is np.datetime64[ns].

    Methods
    -------
    calculate_vector_time(vectors, vectors_per_second, start_time)
    process_vector_data(vector_data, primary_count, secondary_count)
    """

    is_mago: bool
    active: bool
    SHCOARSE: int
    vectors: np.ndarray

    @staticmethod
    def calculate_vector_time(
        vectors: np.ndarray, vectors_per_second: int, start_time: TimeTuple
    ) -> np.ndarray:
        """
        Add timestamps to the vector list, turning the shape from (n, 4) to (n, 5).

        The first vector starts at start_time, then each subsequent vector time is
        computed by adding 1/vectors_per_second to the previous vector's time.

        Parameters
        ----------
        vectors : numpy.ndarray
            List of magnetic vector samples, starting at start_time. Shape of (n, 4).
        vectors_per_second : int
            Number of vectors per second.
        start_time : TimeTuple
            The coarse and fine time for the sensor.

        Returns
        -------
        vector_objects : numpy.ndarray
            Vectors with timestamps added in seconds, calculated from
            cdf.utils.met_to_j2000ns.
        """
        # TODO: Move timestamps to J2000
        timedelta = np.timedelta64(int(1 / vectors_per_second * 1e9), "ns")

        start_time_ns = met_to_j2000ns(start_time.to_seconds())

        # Calculate time skips for each vector in ns
        times = np.reshape(
            np.arange(
                start_time_ns,
                start_time_ns + timedelta * vectors.shape[0],
                timedelta,
                dtype=np.int64,
                like=vectors,
            ),
            (vectors.shape[0], -1),
        )
        vector_objects = np.concatenate([vectors, times], axis=1, dtype=np.int64)
        return vector_objects

    @staticmethod
    def process_vector_data(
        vector_data: np.ndarray, primary_count: int, secondary_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given raw packet data, process into Vectors.

        Vectors are grouped into primary sensor and secondary sensor, and returned as a
        tuple (primary sensor vectors, secondary sensor vectors).

        Written by MAG instrument team.

        Parameters
        ----------
        vector_data : numpy.ndarray
            Raw vector data, in bytes. Contains both primary and secondary vector data
            (first primary, then secondary).
        primary_count : int
            Count of the number of primary vectors.
        secondary_count : int
            Count of the number of secondary vectors.

        Returns
        -------
        (primary, secondary): (numpy.ndarray, numpy.ndarray)
            Two arrays, each containing tuples of (x, y, z, sample_range) for each
            vector sample.
        """

        # TODO: error handling
        def to_signed16(n: int) -> int:
            """
            Convert an integer to a signed 16-bit integer.

            Parameters
            ----------
            n : int
                The integer to be converted.

            Returns
            -------
            int
                Converted integer.
            """
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

        return (
            np.array(primary_vectors, dtype=np.int64),
            np.array(secondary_vectors, dtype=np.int64),
        )
