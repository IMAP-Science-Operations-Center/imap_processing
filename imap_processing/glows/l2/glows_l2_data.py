"""Module containing the class definition for the HistogramL2 class."""

from dataclasses import InitVar, dataclass, field

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.glows import FLAG_LENGTH
from imap_processing.glows.l1b.glows_l1b_data import PipelineSettings
from imap_processing.glows.utils.constants import GlowsConstants
from imap_processing.spice.geometry import (
    SpiceFrame,
    frame_transform_az_el,
    get_instrument_mounting_az_el,
)
from imap_processing.spice.time import (
    et_to_datetime64,
    met_to_sclkticks,
    sct_to_et,
    ttj2000ns_to_et,
)


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
    calibration_factor : float | None
        Rayleigh calibration factor used for flux calculations.
    """

    # All variables should have n_bin elements
    spin_angle: np.ndarray = field(init=False)
    photon_flux: np.ndarray = field(init=False)
    raw_histograms: np.ndarray = field(init=False)
    exposure_times: np.ndarray = field(init=False)
    flux_uncertainties: np.ndarray = field(init=False)
    histogram_flag_array: np.ndarray = field(init=False)
    ecliptic_lon: np.ndarray = field(init=False)
    ecliptic_lat: np.ndarray = field(init=False)
    number_of_bins: int = field(init=False)
    l1b_data: InitVar[xr.Dataset]
    position_angle: InitVar[float]
    calibration_factor: InitVar[float | None]

    def __post_init__(
        self,
        l1b_data: xr.Dataset,
        position_angle: float,
        calibration_factor: float | None,
    ) -> None:
        """
        Compute all the daily lightcurve variables from L1B data.

        Parameters
        ----------
        l1b_data : xarray.Dataset
            L1B data filtered by good times, good angles, and good bins for one
            observation day.
        position_angle : float
            The offset angle of the GLOWS instrument from the north spin point - this
            is used in spin angle calculations.
        calibration_factor : float
            Calibration factor used for flux calculations, in units of counts per second
            per Rayleigh. This is used to convert from raw histograms and exposure times
            to physical photon flux units.
        """
        # number_of_bins_per_histogram is the count of valid (non-FILLVAL) bins.
        # Histogram arrays from L1B are always GlowsConstants.STANDARD_BIN_COUNT
        # (3600) long, with unused bins filled with GlowsConstants.HISTOGRAM_FILLVAL.
        # All bin-dimensioned arrays here are chopped to number_of_bins so that
        # computations only operate on valid data. glows_l2.py is responsible for
        # re-expanding these arrays back to STANDARD_BIN_COUNT, filling unused bins
        # with the appropriate CDF FILLVAL before writing to output.

        self.number_of_bins = (
            l1b_data["number_of_bins_per_histogram"].data[0]
            if len(l1b_data["number_of_bins_per_histogram"].data) != 0
            else 0
        )

        self.raw_histograms = self.calculate_histogram_sums(l1b_data["histogram"].data)[
            : self.number_of_bins
        ]

        exposure_per_epoch = (
            l1b_data["spin_period_average"].data
            * l1b_data["number_of_spins_per_block"].data
            / self.number_of_bins
        )

        # Exposure is uniform across bins; sum over all good-time epochs
        self.exposure_times = np.full(self.number_of_bins, np.sum(exposure_per_epoch))

        raw_uncertainties = np.sqrt(self.raw_histograms)
        self.photon_flux = np.zeros(self.number_of_bins)
        self.flux_uncertainties = np.zeros(self.number_of_bins)

        if (
            self.number_of_bins > 0
            and self.exposure_times[0] > 0
            and calibration_factor
        ):
            self.photon_flux = (
                self.raw_histograms / self.exposure_times
            ) / calibration_factor
            self.flux_uncertainties = (
                raw_uncertainties / self.exposure_times
            ) / calibration_factor

        self.spin_angle = np.zeros(0)

        # Apply 'OR' operation to histogram_flag_array across all
        # good-time L1B blocks per bin.
        # Per Section 12.3.4: a flag is True in L2 if it is True in any L1B block.
        # flags shape: (n_epochs, 4, n_bins)
        flags = l1b_data["histogram_flag_array"].data
        if flags.size > 0:
            # Flatten epochs and flag rows into one axis: (n_epochs * 4, n_bins)
            flags_2d = flags.reshape(-1, self.number_of_bins)
            # Apply binary 'OR' operation across all rows per bin: (n_bins,)
            self.histogram_flag_array = np.bitwise_or.reduce(flags_2d, axis=0).astype(
                np.uint8
            )
        else:
            self.histogram_flag_array = np.zeros(self.number_of_bins, dtype=np.uint8)

        if self.number_of_bins:
            # imap_spin_angle_bin_cntr is the raw IMAP spin angle ψ (0 - 360°,
            # bin midpoints).
            spin_angle_bin_cntr = l1b_data["imap_spin_angle_bin_cntr"].data[0][
                : self.number_of_bins
            ]
            # Convert ψ → ψPA (Eq. 29): position angle measured from the
            # northernmost point of the scanning circle.
            self.spin_angle = (spin_angle_bin_cntr - position_angle + 360.0) % 360.0

            # Roll all bin arrays so bin 0 corresponds to the northernmost
            # point (minimum ψPA).
            roll = -np.argmin(self.spin_angle)
            self.spin_angle = np.roll(self.spin_angle, roll)
            self.raw_histograms = np.roll(self.raw_histograms, roll)
            self.photon_flux = np.roll(self.photon_flux, roll)
            self.exposure_times = np.roll(self.exposure_times, roll)
            self.flux_uncertainties = np.roll(self.flux_uncertainties, roll)
            self.histogram_flag_array = np.roll(self.histogram_flag_array, roll)

            # Get the midpoint start time covered by repointing kernels
            # needed to compute ecliptic coordinates
            mid_idx = len(l1b_data["imap_start_time"]) // 2
            pointing_midpoint_time_et = sct_to_et(
                met_to_sclkticks(l1b_data["imap_start_time"][mid_idx].data)
            )
            self.ecliptic_lon, self.ecliptic_lat = (
                self.compute_ecliptic_coords_of_bin_centers(
                    pointing_midpoint_time_et, self.spin_angle
                )
            )

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
        # Zero out areas where HISTOGRAM_FILLVAL (i.e. unused bins)
        histograms[histograms == GlowsConstants.HISTOGRAM_FILLVAL] = 0
        return np.sum(histograms, axis=0, dtype=np.int64)

    @staticmethod
    def compute_ecliptic_coords_of_bin_centers(
        data_time_et: float, spin_angle_bin_centers: NDArray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the ecliptic coordinates of the histogram bin centers.

        This method transforms the instrument pointing direction for each bin
        center from the IMAP Pointing frame (IMAP_DPS) to the ECLIPJ2000 frame.

        Parameters
        ----------
        data_time_et : float
            Ephemeris time corresponding to the midpoint of the histogram accumulation.

        spin_angle_bin_centers : numpy.ndarray
            Spin angle bin centers for the histogram bins, measured in the IMAP frame,
            with shape (n_bins,), and already corrected for the northernmost point of
            the scanning circle.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Longitude and latitudes in the ECLIPJ2000 frame representing the pointing
            direction of each histogram bin center, with shape (n_bins,).
        """
        # In the IMAP frame, the azimuth corresponds to the spin angle bin centers
        azimuth = spin_angle_bin_centers

        # Get elevation from instrument pointing direction in the DPS frame.
        az_el_instrument_mounting = get_instrument_mounting_az_el(SpiceFrame.IMAP_GLOWS)
        elevation = az_el_instrument_mounting[1]

        # Create array of azimuth, elevation coordinates in the DPS frame (n_bins, 2)
        az_el = np.stack((azimuth, np.full_like(azimuth, elevation)), axis=-1)

        # Transform coordinates to ECLIPJ2000 frame using SPICE transformations.
        ecliptic_coords = frame_transform_az_el(
            data_time_et,
            az_el,
            SpiceFrame.IMAP_DPS,
            SpiceFrame.ECLIPJ2000,
        )

        # Return ecliptic longitudes and latitudes as separate arrays.
        return ecliptic_coords[:, 0], ecliptic_coords[:, 1]


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
        calibration_dataset : xr.Dataset
            The cps-to-Rayleigh calibration dataset needed for flux calculations.

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
    identifier: int
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
    position_angle_offset_average: np.double
    position_angle_offset_std_dev: np.double
    spin_axis_orientation_std_dev: np.ndarray[np.double]
    spacecraft_location_average: np.ndarray[np.double]
    spacecraft_location_std_dev: np.ndarray[np.double]
    spacecraft_velocity_average: np.ndarray[np.double]
    spacecraft_velocity_std_dev: np.ndarray[np.double]
    spin_axis_orientation_average: np.ndarray[np.double]
    bad_time_flag_occurrences: np.ndarray

    def __init__(
        self,
        l1b_dataset: xr.Dataset,
        pipeline_settings: PipelineSettings,
        calibration_dataset: xr.Dataset,
    ) -> None:
        """
        Given an L1B dataset, process data into an output HistogramL2 object.

        Parameters
        ----------
        l1b_dataset : xr.Dataset
            GLOWS histogram L1B dataset, as produced by glows_l1b.py.
        pipeline_settings : PipelineSettings
            Pipeline settings object read from ancillary file.
        calibration_dataset : xr.Dataset
            cps-to-Rayleigh calibration dataset used for flux calculations.
            coords: start_time_utc,  data_vars: cps_per_r
        """
        active_flags = np.array(pipeline_settings.active_bad_time_flags, dtype=float)

        # Apply sunrise/sunset offsets to extend the night region around
        # is_night transitions before selecting good blocks.
        flags = self.apply_is_night_offsets(
            l1b_dataset["flags"].data,
            is_night_idx=GlowsConstants.IS_NIGHT_FLAG_IDX,
            sunrise_offset=int(pipeline_settings.sunrise_offset),
            sunset_offset=int(pipeline_settings.sunset_offset),
        )
        flags_da = xr.DataArray(flags, dims=l1b_dataset["flags"].dims)

        # Select the good blocks (i.e. epoch values) according to the flags. Drop any
        # bad blocks before processing.
        good_data = l1b_dataset.isel(
            epoch=self.return_good_times(flags_da, active_flags)
        )
        # TODO: bad angle filter
        # TODO: filter bad bins out. Needs to happen here while everything is still
        #       per-timestamp.

        self.total_l1b_inputs = len(l1b_dataset["epoch"])
        self.number_of_good_l1b_inputs = len(good_data["epoch"])
        repointing = l1b_dataset.attrs.get("Repointing")
        self.identifier = int(repointing.replace("repoint", ""))
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

        position_angle = self.compute_position_angle()
        self.position_angle_offset_average: np.double = np.double(position_angle)

        # Always zero - per algorithm doc 10.6
        self.position_angle_offset_std_dev = np.double(0.0)
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

        # Select calibration factor corresponding to the mid-epoch in the L1B data.
        if len(good_data["epoch"].data) != 0:
            calibration_factor = self.get_calibration_factor(
                good_data["epoch"].data, calibration_dataset
            )
        else:
            calibration_factor = None  # No good data available. Still proceed

        self.daily_lightcurve = DailyLightcurve(
            good_data, position_angle, calibration_factor
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
            print("Active flags don't match expected length")

        # A good time is where all the active flags are equal to one.
        # Here, we mask the active indices using active_flags, and then return the times
        # where all the active indices == 1.
        good_times = np.where(np.all(flags[:, active_flags == 1] == 1, axis=1))[0]
        return good_times

    @staticmethod
    def apply_is_night_offsets(
        flags: np.ndarray,
        is_night_idx: int,
        sunrise_offset: int,
        sunset_offset: int,
    ) -> np.ndarray:
        """
        Apply sunrise/sunset offsets to is_night transitions.

        Per algorithm doc v4.4.7, Sec. 3.9.1, item 2 (raw is_night: 1=night, 0=day):

        sunset_offset applies at both transitions:
          >0: night shortens by N at each end (first N night epochs at sunset become
              day; last N night epochs before sunrise become day)
          <0: night extends by |N| at each end

        sunrise_offset is an additional adjustment at sunrise (is_night 1->0) only:
          >0: night extends N histograms past the raw sunrise transition
          <0: night shortens by |N| before the raw sunrise transition

        In the processed flags array: 0 = bad (night), 1 = good (day).

        Parameters
        ----------
        flags : numpy.ndarray
            Flags array with shape (n_epochs, FLAG_LENGTH), 0=bad, 1=good.
        is_night_idx : int
            Column index of the is_night flag in the flags array.
        sunrise_offset : int
            Additional histogram shift at the sunrise (is_night 1->0) transition.
        sunset_offset : int
            Histogram shift applied at both the sunset and sunrise transitions.

        Returns
        -------
        numpy.ndarray
            Returns the original flags array if no offsets are applied,
            otherwise returns a modified copy.

        Notes
        -----
        Algorithm doc v4.4.7, Sec. 3.9.1, item 2
        is_night: 1 = daytime (good), 0 = night (bad)
        """
        # If sunrise_offset=0 and sunset_offset=0 then no corrections are needed
        # relative to is_night transition set onboard.
        if sunrise_offset == 0 and sunset_offset == 0:
            return flags

        flags_with_offsets = flags.copy()

        is_night_col = flags[:, is_night_idx]
        n = flags.shape[0]
        diff = np.diff(is_night_col.astype(int))
        sunset_index = np.where(diff == -1)[0]
        sunrise_index = np.where(diff == 1)[0]

        if sunrise_offset > 0:
            # Night (flag = 0) extends by sunrise_offset relative
            # to is_night 0 -> 1 transition.
            for i in sunrise_index:
                flags_with_offsets[
                    i + 1 : min(n, i + 1 + sunrise_offset), is_night_idx
                ] = 0

        elif sunrise_offset < 0:
            # Night (flag = 0) shortens by sunrise_offset relative
            # to is_night 0 -> 1 transition.
            for i in sunrise_index:
                flags_with_offsets[
                    max(0, i + 1 + sunrise_offset) : i + 1, is_night_idx
                ] = 1

        if sunset_offset > 0:
            # Night (flag = 0) shortens by sunset_offset relative
            # to is_night 1 -> 0 transition.
            for i in sunset_index:
                flags_with_offsets[
                    i + 1 : min(n, i + 1 + sunset_offset), is_night_idx
                ] = 1

        elif sunset_offset < 0:
            # Night (flag = 0) extends by sunset_offset relative
            # to is_night 1 -> 0 transition.
            for i in sunset_index:
                flags_with_offsets[
                    max(0, i + 1 + sunset_offset) : i + 1, is_night_idx
                ] = 0

        return flags_with_offsets

    def compute_position_angle(self) -> float:
        """
        Compute the position angle based on the instrument mounting.

        This number is not expected to change significantly. It is the same for all L1B
        blocks (epoch values).

        Returns
        -------
        float
            The GLOWS mounting position angle.
        """
        # Calculation described in algorithm doc 10.6 (Eq. 30):
        # psi_G_eff = 360 - psi_GLOWS
        # where psi_GLOWS is the azimuth of the GLOWS boresight in the
        # IMAP spacecraft frame, measured from the spacecraft x-axis.
        # This angle does not change with time, as it is in the spinning IMAP frame.
        # It basically defines the angle between x=0 in the IMAP frame and x=0 in the
        # GLOWS instrument frame, and is defined by the physical mounting location of
        # the instrument.
        # delta_psi_G_eff is assumed to be 0 per instrument team decision (aka this
        # doesn't move from the SPICE determined mounting angle.
        glows_mounting_azimuth, _ = get_instrument_mounting_az_el(SpiceFrame.IMAP_GLOWS)
        return (360.0 - glows_mounting_azimuth) % 360.0

    @staticmethod
    def get_calibration_factor(
        epoch_values: np.ndarray, calibration_dataset: xr.Dataset
    ) -> float:
        """
        Select calibration factor for an observational day.

        The calibration factor is needed to compute flux in Rayleigh units.
        There is a strong assumption that the calibration is constant for
        a given observational day.

        Parameters
        ----------
        epoch_values : np.ndarray
            Array of epoch values from the L1B dataset, in TT J2000 nanoseconds.
        calibration_dataset : xr.Dataset
            Dataset containing calibration data with the following structure:
                Coords: epoch (datetime64[s])
                Dims: epoch, cps_per_r_dim_0, start_time_utc_dim_0
                Data vars: "cps_per_r" and "start_time_utc" are 2D (epoch, *_dim_0)

                Note: epoch and start_time_utc do not necessarily match in size or
                      values
                    - epoch contains timestamps in the calibration data up to a defined
                      day buffer and start_time_utc are the timestamps for all the
                      calibration data entries.
                    - epoch is used for selecting the time block, and start_time_utc is
                      used for selecting the calibration value within that block.

        Returns
        -------
        float
            The calibration factor needed to compute flux in Rayleigh units.
        """
        # Use the midpoint epoch for the observation day
        mid_idx = len(epoch_values) // 2
        mid_epoch_utc = et_to_datetime64(ttj2000ns_to_et(epoch_values[mid_idx].item()))

        # Select calibration data before or equal to mid_epoch_utc using "pad" to find
        # the nearest preceding entry in the calibration dataset's epoch
        # coordinate which is in UTC datetime64 format.
        cal_at_epoch = calibration_dataset.sel(epoch=mid_epoch_utc, method="pad")

        # start_time_utc is a data variable with its own index dimension.
        # Use searchsorted to find the last entry whose start_time_utc <= mid_epoch_utc.
        start_times = np.array(
            cal_at_epoch["start_time_utc"].values, dtype="datetime64[ns]"
        )
        nearest_idx = np.searchsorted(start_times, mid_epoch_utc, side="right") - 1

        # Select the calibration value at the nearest index.
        return float(cal_at_epoch["cps_per_r"].isel(cps_per_r_dim_0=nearest_idx))
