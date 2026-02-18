"""Module containing the class definition for the HistogramL2 class."""

from dataclasses import InitVar, dataclass, field

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.glows import FLAG_LENGTH
from imap_processing.glows.l1b.glows_l1b_data import PipelineSettings


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

        self.number_of_bins = l1b_data["histogram"].shape[1]

        exposure_per_epoch = (
            l1b_data["spin_period_average"].data
            * l1b_data["number_of_spins_per_block"].data
            / self.number_of_bins
        )

        # Exposure is uniform across bins; sum over all good-time epochs
        self.exposure_times = np.full(self.number_of_bins, np.sum(exposure_per_epoch))

        raw_uncertainties = np.sqrt(self.raw_histograms)
        self.photon_flux = np.zeros(len(self.raw_histograms))
        self.flux_uncertainties = np.zeros(len(self.raw_histograms))

        # TODO: Only where exposure counts != 0
        if len(self.exposure_times) != 0:
            self.photon_flux = self.raw_histograms / self.exposure_times
            self.flux_uncertainties = raw_uncertainties / self.exposure_times

        # TODO: Average this, or should they all be the same?
        self.spin_angle = np.average(l1b_data["imap_spin_angle_bin_cntr"].data, axis=0)

        self.histogram_flag_array = np.zeros(self.number_of_bins)
        self.ecliptic_lon = np.zeros(self.number_of_bins)
        self.ecliptic_lat = np.zeros(self.number_of_bins)

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
        histograms = histograms.copy()
        histograms[histograms == -1] = 0
        return np.sum(histograms, axis=0, dtype=np.int64)


@dataclass
class HistogramL2:
    """
    Dataclass describing Histogram L2 data variables and methods.

    This class collects multiple HistogramL1B classes into one L2 per observational day.

    Parameters
    ----------
        l1b_dataset : xr.Dataset
            GLOWS histogram L1B dataset, as produced by glows_l1b.py.
        pipeline_settings : PipelineSettings
            Pipeline settings object read from ancillary file.

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

    def __init__(self, l1b_dataset: xr.Dataset, pipeline_settings: PipelineSettings):
        """
        Given an L1B dataset, process data into an output HistogramL2 object.

        Parameters
        ----------
        l1b_dataset : xr.Dataset
            GLOWS histogram L1B dataset, as produced by glows_l1b.py.
        pipeline_settings : PipelineSettings
            Pipeline settings object read from ancillary file.
        """
        active_flags = np.array(pipeline_settings.active_bad_time_flags, dtype=float)

        # Select the good blocks (i.e. epoch values) according to the flags. Drop any
        # bad blocks before processing.
        good_data = l1b_dataset.isel(
            epoch=self.return_good_times(l1b_dataset["flags"], active_flags)
        )
        # todo: bad angle filter
        # TODO filter bad bins out. Needs to happen here while everything is still
        # per-timestamp.

        self.daily_lightcurve = DailyLightcurve(good_data)

        self.total_l1b_inputs = len(good_data["epoch"])
        self.number_of_good_l1b_inputs = len(good_data["epoch"])
        self.identifier = -1  # TODO: retrieve from spin table
        # TODO fill this in
        self.bad_time_flag_occurrences = np.zeros((1, FLAG_LENGTH))

        if len(good_data["epoch"]) != 0:
            # Generate outputs that are passed in directly from L1B
            self.start_time = good_data["epoch"].data[0]
            self.end_time = good_data["epoch"].data[-1]
        else:
            # No good times in the file
            self.start_time = l1b_dataset["imap_start_time"].data[0]
            self.end_time = (
                l1b_dataset["imap_start_time"].data[0]
                + l1b_dataset["imap_time_offset"].data[0]
            )

        self.filter_temperature_average = (
            good_data["filter_temperature_average"]
            .mean(dim="epoch", keepdims=True)
            .data
        )
        self.filter_temperature_std_dev = (
            good_data["filter_temperature_average"].std(dim="epoch", keepdims=True).data
        )
        self.hv_voltage_average = (
            good_data["hv_voltage_average"].mean(dim="epoch", keepdims=True).data
        )
        self.hv_voltage_std_dev = (
            good_data["hv_voltage_average"].std(dim="epoch", keepdims=True).data
        )
        self.spin_period_average = (
            good_data["spin_period_average"].mean(dim="epoch", keepdims=True).data
        )
        self.spin_period_std_dev = (
            good_data["spin_period_average"].std(dim="epoch", keepdims=True).data
        )
        self.pulse_length_average = (
            good_data["pulse_length_average"].mean(dim="epoch", keepdims=True).data
        )
        self.pulse_length_std_dev = (
            good_data["pulse_length_average"].std(dim="epoch", keepdims=True).data
        )
        self.spin_period_ground_average = (
            good_data["spin_period_ground_average"]
            .mean(dim="epoch", keepdims=True)
            .data
        )
        self.spin_period_ground_std_dev = (
            good_data["spin_period_ground_average"].std(dim="epoch", keepdims=True).data
        )
        self.position_angle_offset_average = (
            good_data["position_angle_offset_average"]
            .mean(dim="epoch", keepdims=True)
            .data
        )
        self.position_angle_offset_std_dev = (
            good_data["position_angle_offset_average"]
            .std(dim="epoch", keepdims=True)
            .data
        )
        self.spacecraft_location_average = (
            good_data["spacecraft_location_average"]
            .mean(dim="epoch")
            .data[np.newaxis, :]
        )
        self.spacecraft_location_std_dev = (
            good_data["spacecraft_location_average"]
            .std(dim="epoch")
            .data[np.newaxis, :]
        )
        self.spacecraft_velocity_average = (
            good_data["spacecraft_velocity_average"]
            .mean(dim="epoch")
            .data[np.newaxis, :]
        )
        self.spacecraft_velocity_std_dev = (
            good_data["spacecraft_velocity_average"]
            .std(dim="epoch")
            .data[np.newaxis, :]
        )
        self.spin_axis_orientation_average = (
            good_data["spin_axis_orientation_average"]
            .mean(dim="epoch")
            .data[np.newaxis, :]
        )
        self.spin_axis_orientation_std_dev = (
            good_data["spin_axis_orientation_average"]
            .std(dim="epoch")
            .data[np.newaxis, :]
        )

    def filter_bad_bins(self, histograms: NDArray, bin_exclusions: NDArray) -> NDArray:
        """
        Filter out bad bins from the histogram.

        Parameters
        ----------
        histograms : numpy.ndarray
            Histogram data, with shape (n_timestamps, n_bins).
        bin_exclusions : numpy.ndarray
            Array of bin exclusions. This 2d array has a timestamp and bin filter array
            pair. The bin filter array indicates "1" if a bin is to be excluded.

        Returns
        -------
        numpy.ndarray
            Histogram data with bad bins marked with -1.
        """
        # TODO: will need ancillary file imap_glows_exclusions_by_instr_team
        # TODO: complete once unique_block_identifier is implemented
        # file contains timestamp & bin filter array pairs. For the timestamp, the
        # filter should be applied such that 1 excludes the bin.

        # excluded bins can be marked with -1
        return histograms

    @staticmethod
    def return_good_times(flags: xr.DataArray, active_flags: NDArray) -> NDArray:
        """
        Return the good times based on the input flags.

        Parameters
        ----------
        flags : xarray.DataArray
            Flags dataset with shape (n_timestamps, n_flags). If a flag is active and
             set to 1, the timestamp is considered good.

        active_flags : numpy.ndarray
            Array of active flags. If the flag is set to 1, it is considered active.

        Returns
        -------
        numpy.ndarray
            An array of indices for good times.
        """
        if len(active_flags) != flags.shape[1]:
            print("Active flags don't matched expected length")

        # A good time is where all the active flags are equal to one.
        # Here, we mask the active indices using active_flags, and then return the times
        # where all the active indices == 1.
        good_times = np.where(np.all(flags[:, active_flags == 1] == 1, axis=1))[0]
        return good_times
