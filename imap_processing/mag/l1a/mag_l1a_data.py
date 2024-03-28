"""Data classes for storing and processing MAG Level 1A data."""
from dataclasses import dataclass
from math import floor

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

    def __add__(self, seconds: int):
        """
        Add a number of seconds to the time tuple.

        Parameters
        ----------
        seconds : int
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
    vectors: tuple[int, int, int, int]

    def __init__(self, vectors, previous_time, time_step):
        self.vectors = vectors
        self.timestamp = previous_time + time_step


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
    vector_samples : list[Vector]
        List of magnetic vector samples, starting at start_time

    """

    is_mago: bool
    active: bool
    start_time: TimeTuple
    vectors_per_second: int
    expected_vector_count: int
    seconds_of_data: int
    SHCOARSE: int
    vector_samples: list

    # def __post_init__(self):
    # TODO: Assign timestamps to vectors, convert list to Vector type
