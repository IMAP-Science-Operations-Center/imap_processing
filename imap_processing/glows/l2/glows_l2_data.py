"""Module containing the class definition for the HistogramL2 class."""

from dataclasses import InitVar, dataclass, field

import numpy as np
import xarray as xr
from numpy.typing import NDArray


@dataclass
class DailyLightcurve:
    """
    Dataclass describing the daily lightcurve.

    Used inside the HistogramL2 class.

    Attributes
    ----------
    spin_angle : numpy.ndarray
        values of spin angle [deg] for bin centers (measured from the north)
    photon_flux : numpy.ndarray
        observational-day-averaged photon flux [Rayleigh]
    raw_histograms : numpy.ndarray
        sum of histograms across all timestamps
    exposure_times : numpy.ndarray
        exposure times for bins [s]
    flux_uncertainties : numpy.ndarray
        statistical uncertainties for photon flux [Rayleigh]
    histogram_flag_array : numpy.ndarray
        flags for bins
    ecliptic_lon : numpy.ndarray
        ecliptic longitude of bin centers [deg]
    ecliptic_lat : numpy.ndarray
        ecliptic latitude of bin centers [deg]
    number_of_bins : int
        number of bins in lightcurve
    l1b_data : xarray.Dataset
        L1B data filtered by good times, good angles, and good bins.
    """

    # All variables should have n_bin elements
    spin_angle: np.ndarray = field(init=False)
    photon_flux: np.ndarray = field(init=False)
    raw_histograms: np.ndarray = field(init=False)
    exposure_times: np.ndarray = field(init=False)
    flux_uncertainties: np.ndarray = field(init=False)
    # TODO: flag array
    histogram_flag_array: np.ndarray = field(init=False)
    # TODO: ecliptic coordinates
    ecliptic_lon: np.ndarray = field(init=False)
    ecliptic_lat: np.ndarray = field(init=False)
    number_of_bins: int = field(init=False)
    l1b_data: InitVar[xr.Dataset]

    def __post_init__(self, l1b_data: xr.Dataset) -> None:
        """
        Compute all the daily lightcurve variables from L1B data.

        Parameters
        ----------
        l1b_data : xarray.Dataset
            L1B data filtered by good times, good angles, and good bins for one
            observation day.
        """
        self.raw_histograms = self.calculate_histogram_sums(l1b_data["histogram"].data)

        exposure_times_per_timestamp = (
            l1b_data["spin_period_average"]
            * l1b_data["number_of_spins_per_block"]
            / 3600
        )

        self.exposure_times = self.calculate_exposure_times(
            l1b_data, exposure_times_per_timestamp
        )
        raw_uncertainties = np.sqrt(self.raw_histograms)
        self.photon_flux = np.zeros(len(self.raw_histograms))
        self.flux_uncertainties = np.zeros(len(self.raw_histograms))

        # TODO: Only where exposure counts != 0
        if len(self.exposure_times) != 0:
            self.photon_flux = self.raw_histograms / self.exposure_times
            self.flux_uncertainties = raw_uncertainties / self.exposure_times

        # TODO: Average this, or should they all be the same?
        self.spin_angle = np.average(l1b_data["imap_spin_angle_bin_cntr"].data, axis=0)

        # TODO: is the first number here ok? Would it change mid-obs day?
        self.number_of_bins = len(self.spin_angle)

        self.histogram_flag_array = np.zeros(self.number_of_bins)
        self.ecliptic_lon = np.zeros(self.number_of_bins)
        self.ecliptic_lat = np.zeros(self.number_of_bins)

    @staticmethod
    def calculate_exposure_times(
        good_times: xr.Dataset, exposure_count: xr.DataArray
    ) -> NDArray:
        """
        Calculate exposure times for each bin across all the timestamps.

        Parameters
        ----------
        good_times : xarray.Dataset
            Dataset with only good times.
        exposure_count : float
            Exposure count for each valid timestamp and bin.

        Returns
        -------
        numpy.ndarray
            Array of summed exposure times for each bin.
        """
        weighted_sum = (good_times["histogram"].data != -1) * exposure_count.data[
            :, np.newaxis
        ]
        return np.sum(weighted_sum, axis=0)

    @staticmethod
    def calculate_histogram_sums(histograms: NDArray) -> NDArray:
        """
        Calculate the sum of histograms across all timestamps.

        Parameters
        ----------
        histograms : numpy.ndarray
            Array of histograms across all timestamps.

        Returns
        -------
        numpy.ndarray
            Sum of valid histograms across all timestamps.
        """
        histograms[histograms == -1] = 0
        return np.sum(histograms, axis=0, dtype=np.int64)


@dataclass
class HistogramL2:
    """
    Dataclass describing Histogram L2 data variables and methods.

    This class collects multiple HistogramL1B classes into one L2 per observational day.

    Attributes
    ----------
    number_of_good_l1b_inputs : int
        number of good-time Level-1B times used for generation of Level-2 data.
    total_l1b_inputs : int
      number of all Level-1B times for observational day.
    identifier : int
        unique Level-2 histogram identifier
    start_time : numpy.double
        J2000 start time of a given observational day
    end_time : numpy.double
        J2000 end time of a given observational day
    daily_lightcurve : numpy.ndarray
        arrays for observational-day-accumulated lightcurve
    filter_temperature_average : numpy.ndarray
        observational-day-averaged filter temperature [Celsius deg]
    filter_temperature_std_dev : numpy.ndarray
        standard deviation for filter temperature [Celsius deg]
    hv_voltage_average : numpy.ndarray
        observational-day-averaged channeltron voltage [volt]
    hv_voltage_std_dev : numpy.ndarray
        standard deviation for channeltron voltage [volt]
    spin_period_average : numpy.ndarray
        observational-day-averaged spin period [s] (onboard value)
    spin_period_std_dev : numpy.ndarray
        a standard deviation for spin period [s]
    pulse_length_average : numpy.ndarray
        observational-day-averaged pulse length [μs]
    pulse_length_std_dev : numpy.ndarray
        standard deviation for pulse length [μs]
    spin_period_ground_average : numpy.ndarray
        observational-day-averaged spin period [s] (ground value)
    spin_period_ground_std_dev : numpy.ndarray
        a standard deviation for spin period [s]
    position_angle_offset_average : numpy.ndarray
        observational-day-averaged GLOWS angular offset [deg]
    position_angle_offset_std_dev : numpy.ndarray
        standard deviation for GLOWS angular offset [seg]
    spin_axis_orientation_std_dev : numpy.ndarray
        standard deviation for spin-axis longitude and latitude [deg]
    spacecraft_location_average : numpy.ndarray
        observational-day-averaged Cartesian ecliptic coordinates ⟨X⟩, ⟨Y ⟩, ⟨Z⟩ [km]
        of IMAP
    spacecraft_location_std_dev : numpy.ndarray
        standard deviation for ecliptic coordinates [km] of IMAP
    spacecraft_velocity_average : numpy.ndarray
        observational-day-averaged values ⟨VX ⟩, ⟨VY ⟩, ⟨VZ ⟩ of IMAP velocity
        components [km/s] (Cartesian ecliptic frame)
    spacecraft_velocity_std_dev : numpy.ndarray
        standard deviation for IMAP velocity components [km/s]
    spin_axis_orientation_average : numpy.ndarray
        observational-day-averaged spin-axis ecliptic longitude and latitude [deg]
    bad_time_flag_occurrences : numpy.ndarray
        numbers of occurrences of blocks for each bad-time flag during observational day
    """

    number_of_good_l1b_inputs: int
    total_l1b_inputs: int
    identifier: int  # TODO: Should be the official pointing number
    start_time: np.double
    end_time: np.double
    daily_lightcurve: DailyLightcurve
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
    spacecraft_location_average: np.ndarray[np.double]
    spacecraft_location_std_dev: np.ndarray[np.double]
    spacecraft_velocity_average: np.ndarray[np.double]
    spacecraft_velocity_std_dev: np.ndarray[np.double]
    spin_axis_orientation_average: np.ndarray[np.double]
    bad_time_flag_occurrences: np.ndarray
