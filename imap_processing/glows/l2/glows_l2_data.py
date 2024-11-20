"""Module containing the class definition for the HistogramL2 class."""

from dataclasses import InitVar, dataclass, field

import numpy as np


@dataclass
class HistogramL2:
    """
    Dataclass describing Histogram L2 data variables and methods.

    This class collects multiple HistogramL1B classes into one L2 per observational day.

    flight_software_version : str
    number_of_good_l1b_inputs : int
        number of good-time Level-1B times used for generation of Level-2 data
    total_l1b_inputs : int
      number of all Level-1B times for observational day
    identifier : int
        unique Level-2 histogram identifier
    start_time : numpy.double
        UTC start time of a given observational day
    end_time : numpy.double
        UTC end time of a given observational day
    daily_lightcurve : numpy.ndarray
        arrays for observational-day-accumulated lightcurve
    filter_temperature_average : numpy.ndarray
        observational-day-averaged filter temperature [Celsius deg]
    filter_temperature_variance : numpy.ndarray
        standard deviation for filter temperature [Celsius deg]
    hv_voltage_average : numpy.ndarray
        observational-day-averaged channeltron voltage [volt]
    hv_voltage_variance : numpy.ndarray
        standard deviation for channeltron voltage [volt]
    spin_period_average : numpy.ndarray
        observational-day-averaged spin period [s] (onboard value)
    spin_period_variance : numpy.ndarray
        a standard deviation for spin period [s]
    pulse_length_average : numpy.ndarray
        observational-day-averaged pulse length [μs]
    pulse_length_variance : numpy.ndarray
        standard deviation for pulse length [μs]
    spin_period_ground_average : numpy.ndarray
        observational-day-averaged spin period [s] (ground value)
    spin_period_ground_variance : numpy.ndarray
        a standard deviation for spin period [s]
    position_angle_offset_average : numpy.ndarray
        observational-day-averaged GLOWS angular offset [deg]
    position_angle_offset_variance : numpy.ndarray
        standard deviation for GLOWS angular offset [seg]
    spin_axis_orientation_variance : numpy.ndarray
        standard deviation for spin-axis longitude and latitude [deg]
    spacecraft_location_variance : numpy.ndarray
        standard deviation for ecliptic coordinates [km] of IMAP
    spacecraft_velocity_variance : numpy.ndarray
        standard deviation for IMAP velocity components [km/s]
    spin_axis_orientation_average : numpy.ndarray
        observational-day-averaged spin-axis ecliptic longitude and latitude [deg]
    spacecraft_location_average : numpy.ndarray
        observational-day-averaged Cartesian ecliptic coordinates ⟨X⟩, ⟨Y ⟩, ⟨Z⟩ [km]
        of IMAP
    spacecraft_velocity_average : numpy.ndarray
        observational-day-averaged values ⟨VX ⟩, ⟨VY ⟩, ⟨VZ ⟩ of IMAP velocity
        components [km/s] (Cartesian ecliptic frame)
    bad_time_flag_occurrences : numpy.ndarray
        numbers of occurrences of blocks for each bad-time flag during observational day
    """

    flight_software_version: str
    number_of_good_l1b_inputs: int
    total_l1b_inputs: int
    # identifier: int  # comes from unique_block_identifier
    start_time: np.double
    end_time: np.double
    daily_lightcurve: np.ndarray = field(init=False)
    filter_temperature_average: np.ndarray[np.double]
    filter_temperature_std_dev: np.ndarray[np.double]
    hv_voltage_average: np.ndarray[np.double]
    hv_voltage_std_dev: np.ndarray[np.double]
    spin_period_average: np.ndarray[np.double]
    spin_period_std_dev: np.ndarray[np.double]
    pulse_length_average: np.ndarray[np.double]
    pulse_length_std_dev: np.ndarray[np.double]
    spin_period_ground_average: np.ndarray[np.double]
    spin_period_ground_std_dev: np.ndarray[np.double]
    position_angle_offset_average: np.ndarray[np.double]
    position_angle_offset_std_dev: np.ndarray[np.double]
    spin_axis_orientation_std_dev: np.ndarray[np.double]
    spacecraft_location_std_dev: np.ndarray[np.double]
    spacecraft_velocity_std_dev: np.ndarray[np.double]
    spin_axis_orientation_average: np.ndarray[np.double]
    spacecraft_location_average: np.ndarray[np.double]
    spacecraft_velocity_average: np.ndarray[np.double]
    bad_time_flag_occurrences: np.ndarray
    histogram: InitVar[np.ndarray]

    def __post_init__(self, histogram: np.ndarray) -> None:
        """
        Post-initialization method to generate the daily light curve from one histogram.

        Parameters
        ----------
        histogram : numpy.ndarray
            Histogram data from L1B, of shape (bins,) where bins is nominally 3600.
        """
        self.daily_lightcurve = self.generate_lightcurve(histogram)

    def generate_lightcurve(self, histogram: np.ndarray) -> np.ndarray:
        """
        Given an array of (n, bins) histograms, generate one lightcurve of size (bins).

        Parameters
        ----------
        histogram : numpy.ndarray
            Histogram data from L1B, of shape (bins,) where bins is nominally 3600.

        Returns
        -------
        numpy.ndarray
            Lightcurve of size (bins).
        """
        return np.zeros(3600)  # type: ignore[no-any-return]
