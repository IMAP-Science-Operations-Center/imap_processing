"""Module for GLOWS L1B data products."""

import dataclasses
import json
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from imap_processing.glows import FLAG_LENGTH
from imap_processing.glows.utils.constants import TimeTuple


class AncillaryParameters:
    """
    GLOWS L1B Ancillary Parameters for decoding ancillary histogram data points.

    This class reads from a JSON file input which defines ancillary parameters.
    It validates to ensure the input file has all the required parameters.

    Parameters
    ----------
    input_table : dict
        Dictionary generated from input JSON file.

    Attributes
    ----------
    version: str
        version of the ancillary file
    filter_temperature: dict
        dictionary of filter temperature parameters, with keys ["min", "max", "n_bits",
        "p01", "p02", "p03", "p04"]
    hv_voltage: dict
        dictionary of CEM voltage parameters, with keys ["min", "max", "n_bits",
        "p01", "p02", "p03", "p04"]
    spin_period: dict
        dictionary of spin period parameters, with keys ["min", "max", "n_bits"]
    spin_phase: dict
        dictionary of spin phase parameters, with keys ["min", "max", "n_bits"]
    pulse_length: dict
        dictionary of pulse length parameters, with keys ["min", "max", "n_bits",
        "p01", "p02", "p03", "p04"]
    """

    def __init__(self, input_table: dict):
        """
        Generate ancillary parameters from the given input.

        Validates parameters and will throw a KeyError if input data is incorrect.
        """
        full_keys = ["min", "max", "n_bits", "p01", "p02", "p03", "p04"]
        spin_keys = ["min", "max", "n_bits"]

        try:
            self.version = input_table["version"]
            self.filter_temperature = input_table["filter_temperature"]
            if any([key not in full_keys for key in self.filter_temperature.keys()]):
                raise KeyError("Filter temperature parameters are incorrect.")
            self.hv_voltage = input_table["hv_voltage"]
            if any([key not in full_keys for key in self.hv_voltage.keys()]):
                raise KeyError("HV voltage parameters are incorrect.")
            self.spin_period = input_table["spin_period"]
            if any([key not in spin_keys for key in self.spin_period.keys()]):
                raise KeyError("Spin period parameters are incorrect.")
            self.spin_phase = input_table["spin_phase"]
            if any([key not in spin_keys for key in self.spin_phase.keys()]):
                raise KeyError("Spin phase parameters are incorrect.")
            self.pulse_length = input_table["pulse_length"]
            if any([key not in full_keys for key in self.pulse_length.keys()]):
                raise KeyError("Pulse length parameters are incorrect.")

        except KeyError as e:
            raise KeyError(
                "GLOWS L1B Ancillary input_table does not conform to expected format."
            ) from e

    def decode(self, param_key: str, encoded_value: np.double) -> np.double:
        """
        Decode parameters using the algorithm defined in section -.

        The output parameter T_d is defined as:
        T_d = (T_e - B) / A

        where T_e is the encoded value and A and B are:
        A = (2^n - 1) / (max - min)
        B = -min * A

        Max, min, and n are defined in an ancillary data file defined by
        AncillaryParameters.

        Parameters
        ----------
        param_key : str
            The parameter to use for decoding. Should be one of "filter_temperature",
            "hv_voltage", "spin_period", "spin_phase", or "pulse_length".
        encoded_value : np.double
            The encoded value to decode.

        Returns
        -------
        decoded_value : np.double
            The decoded value.
        """
        params = getattr(self, param_key)
        # compute parameters a and b:
        param_a = (2 ** params["n_bits"] - 1) / (params["max"] - params["min"])
        param_b = -params["min"] * param_a

        return np.double((encoded_value - param_b) / param_a)

    def decode_std_dev(self, param_key: str, encoded_value: np.double) -> np.double:
        """
        Decode an encoded variance variable and compute the standard deviation.

        The decoded value of encoded_value is given by:
        variance = encoded_value / (param_a**2)

        where param_a is defined as:
        param_a = (2^n - 1) / (max - min)

        The standard deviation is then the square root of the variance.

        Parameters
        ----------
        param_key : str
            The parameter to use for decoding. Should be one of "filter_temperature",
            "hv_voltage", "spin_period", "spin_phase", or "pulse_length".
        encoded_value : np.double
            The encoded variance to decode.

        Returns
        -------
        std_dev : np.double
            The standard deviation of the encoded value.
        """
        params = getattr(self, param_key)
        # compute parameters a and b:
        param_a = (2 ** params["n_bits"] - 1) / (params["max"] - params["min"])

        variance = encoded_value / (param_a**2)

        return np.double(np.sqrt(variance))


@dataclass
class DirectEventL1B:
    """
    GLOWS L1B direct event data product.

    This class uses dataclass "InitVar" types which are only used to create the
    output dataclass and not used beyond the __post_init__ function. These attributes
    represent data variables that are present in L1A but not passed on in the same form
    to L1B.

    Attributes
    ----------
    direct_events: np.ndarray
        4d array consisting of [seconds, subseconds, pulse_length, is_multi_event],
        which is the DirectEvent structure from L1A. This is used to generate
        direct_event_glows_times and direct_event_pulse_lengths.
    seq_count_in_pkts_file: int
        Sequence count in the input file, passed from L1A
    unique_identifier: str
        YYYY-MM-DDThh:mm:ss based on IMAP UTC time
    number_of_de_packets: InitVar[np.double]
        Number of DE packets in the block, passed in from L1A.
        TODO: Missing from algorithm document, double check that this should be in L1B
    imap_time_last_pps: np.double
        Last PPS in IMAP clock format. Copied from imap_sclk_last_pps in L1A,
         In seconds.
    glows_time_last_pps: np.double
        Last PPS in GLOWS clock format. Created from glows_sclk_last_pps and
        glows_ssclk_last_pps in L1A. In seconds, with subseconds as decimal.
    glows_ssclk_last_pps: InitVar[np.double]
        Subseconds of the last PPS in GLOWS clock format. Used to update
        glows_time_last_pps.
    imap_time_next_pps: np.double
        Next PPS in IMAP clock format. Copied from imap_slck_next_pps in L1A.
        In seconds.
    catbed_heater_active: InitVar[np.double]
        Flag for catbed heater
    spin_period_valid: InitVar[np.double]
        Flag for valid spin period
    spin_phase_at_next_pps_valid: InitVar[np.double]
        Flag for valid spin phase at next PPS
    spin_period_source: InitVar[np.double]
        Source of spin period flag
    spin_period: np.double
        Spin period in seconds, decoded from ancillary data
    spin_phase_at_next_pps: np.double
        Spin phase at the next PPS in degrees, decoded from ancillary data
    number_of_completed_spins: int
        Number of completed spins in the block, passed from L1A
    filter_temperature: np.double
        Filter temperature in Celsius degrees, decoded from ancillary data
    hv_voltage: np.double
        CEM voltage in volts, decoded from ancillary data
    glows_time_on_pps_valid: InitVar[np.double]
        Flag for valid GLOWS time on PPS, ends up in flags array
    time_status_valid: InitVar[np.double]
        Flag for valid time status, ends up in flags array
    housekeeping_valid: InitVar[np.double]
        Flag for valid housekeeping, ends up in flags array
    is_pps_autogenerated: InitVar[np.double]
        Flag for autogenerated PPS, ends up in flags array
    hv_test_in_progress: InitVar[np.double]
        Flag for HV test in progress, ends up in flags array
    pulse_test_in_progress: InitVar[np.double]
        Flag for pulse test in progress, ends up in flags array
    memory_error_detected: InitVar[np.double]
        Flag for memory error detected, ends up in flags array
    flags: ndarray
        array of flags for extra information, per histogram. This is assembled from
        L1A variables.
    direct_event_glows_times: ndarray
        array of times for direct events, GLOWS clock, subseconds as decimal part of
        float. From direct_events.
    direct_event_pulse_lengths: ndarray
        array of pulse lengths [μs] for direct events. From direct_events
    """

    direct_events: InitVar[np.ndarray]
    seq_count_in_pkts_file: np.double  # Passed from L1A
    # unique_identifier: str = field(init=False)
    number_of_de_packets: np.double  # TODO Is this required in L1B?
    imap_time_last_pps: np.double
    glows_time_last_pps: np.double
    # Added to the end of glows_time_last_pps as subseconds
    glows_ssclk_last_pps: InitVar[int]
    imap_time_next_pps: np.double
    catbed_heater_active: InitVar[np.double]
    spin_period_valid: InitVar[np.double]
    spin_phase_at_next_pps_valid: InitVar[np.double]
    spin_period_source: InitVar[np.double]
    spin_period: np.double
    spin_phase_at_next_pps: np.double
    number_of_completed_spins: np.double
    filter_temperature: np.double
    hv_voltage: np.double
    glows_time_on_pps_valid: InitVar[np.double]
    time_status_valid: InitVar[np.double]
    housekeeping_valid: InitVar[np.double]
    is_pps_autogenerated: InitVar[np.double]
    hv_test_in_progress: InitVar[np.double]
    pulse_test_in_progress: InitVar[np.double]
    memory_error_detected: InitVar[np.double]
    # The following variables are created from the InitVar data
    de_flags: Optional[np.ndarray] = field(init=False, default=None)
    # TODO: First two values of DE are sec/subsec
    direct_event_glows_times: Optional[np.ndarray] = field(init=False, default=None)
    # 3rd value is pulse length
    direct_event_pulse_lengths: Optional[np.ndarray] = field(init=False, default=None)
    # TODO: where does the multi-event flag go?

    def __post_init__(
        self,
        direct_events: np.ndarray,
        glows_ssclk_last_pps: int,
        catbed_heater_active: np.double,
        spin_period_valid: np.double,
        spin_phase_at_next_pps_valid: np.double,
        spin_period_source: np.double,
        glows_time_on_pps_valid: np.double,
        time_status_valid: np.double,
        housekeeping_valid: np.double,
        is_pps_autogenerated: np.double,
        hv_test_in_progress: np.double,
        pulse_test_in_progress: np.double,
        memory_error_detected: np.double,
    ) -> None:
        """
        Generate the L1B data for direct events using the inputs from InitVar.

        Parameters
        ----------
        direct_events : np.ndarray
            Direct events.
        glows_ssclk_last_pps : int
            Glows subsecond clock for the last PPS.
        catbed_heater_active : np.double
            Flag if the catbed heater is active.
        spin_period_valid : np.double
            Valid spin period.
        spin_phase_at_next_pps_valid : np.double
            Flag indicating if the next spin phase is valid.
        spin_period_source : np.double
            Spin period source.
        glows_time_on_pps_valid : np.double
            Flag indicating if the glows time is valid.
        time_status_valid : np.double
            Flag indicating if time status is valid.
        housekeeping_valid : np.double
            Flag indicating if housekeeping is valid.
        is_pps_autogenerated : np.double
            Flag indicating if the PPS is autogenerated.
        hv_test_in_progress : np.double
            Flag indicating if a HV (high voltage) test is in progress.
        pulse_test_in_progress : np.double
           Flag indicating if a pulse test is in progress.
        memory_error_detected : np.double
            Flag indicating if a memory error is detected.
        """
        self.direct_event_glows_times, self.direct_event_pulse_lengths = (
            self.process_direct_events(direct_events)
        )

        # TODO: double check that this time is in unix time and is the correct variable
        # TODO: This cannot be in the data because it's a string, put it in the
        #  attributes
        # self.unique_identifier = np.datetime_as_string(
        #     np.datetime64(int(self.imap_time_last_pps), "ns"), "s"
        # )
        self.glows_time_last_pps = TimeTuple(
            int(self.glows_time_last_pps), glows_ssclk_last_pps
        ).to_seconds()

        with open(
            Path(__file__).parents[1] / "ancillary" / "l1b_conversion_table_v001.json"
        ) as f:
            self.ancillary_parameters = AncillaryParameters(json.loads(f.read()))

        self.filter_temperature = self.ancillary_parameters.decode(
            "filter_temperature", self.filter_temperature
        )
        self.hv_voltage = self.ancillary_parameters.decode(
            "hv_voltage", self.hv_voltage
        )
        self.spin_period = self.ancillary_parameters.decode(
            "spin_period", self.spin_period
        )

        self.spin_phase_at_next_pps = self.ancillary_parameters.decode(
            "spin_phase", self.spin_phase_at_next_pps
        )

        self.de_flags = np.array(
            [
                catbed_heater_active,
                spin_period_valid,
                spin_phase_at_next_pps_valid,
                spin_period_source,
                glows_time_on_pps_valid,
                time_status_valid,
                housekeeping_valid,
                is_pps_autogenerated,
                hv_test_in_progress,
                pulse_test_in_progress,
                memory_error_detected,
            ]
        )

    @staticmethod
    def process_direct_events(direct_events: np.ndarray) -> tuple:
        """
        Will process direct events data, separating out the time flags and pulse length.

        Parameters
        ----------
        direct_events : np.ndarray
            Direct event data from L1A, with shape (n, 4) where n is the number of
            direct events.

        Returns
        -------
        (times, pulse_lengths) : tuple
            Tuple of two np.ndarrays, the first being the times of the direct events
            and the second being the pulse lengths. Both of shape (n,).
        """
        times = np.zeros((direct_events.shape[0],))
        pulse_lengths = np.zeros((direct_events.shape[0],))
        for index, de in enumerate(direct_events):
            times[index] = TimeTuple(de[0], de[1]).to_seconds()
            pulse_lengths[index] = de[2]

        return times, pulse_lengths


@dataclass
class HistogramL1B:
    """
    GLOWS L1B histogram data product, generated from GLOWS L1A histogram data product.

    All the spice attributes come from the SPICE kernels and are not initialized.
    Other variables are initialized as their encoded or unprocessed values, and then
    decoded or processed in the __post_init__ method.

    IMPORTANT: The order of the fields inherited from L1A must match the order of the
    fields in the DataSet created in decom_glows.py.

    Attributes
    ----------
    histogram
        array of block-accumulated count numbers
    flight_software_version: str
    seq_count_in_pkts_file: int
    first_spin_id: int
        The start ID
    last_spin_id: int
        The ID of the previous spin
    flags_set_onboard: int
    is_generated_on_ground: int
    number_of_spins_per_block
        nblock
    unique_block_identifier
        YYYY-MM-DDThh:mm:ss based on IMAP UTC time
    number_of_bins_per_histogram
        nbin
    number_of_events
        total number of events/counts in histogram
    imap_spin_angle_bin_cntr
        IMAP spin angle ψ for bin centers
    filter_temperature_average
        block-averaged value, decoded to Celsius degrees using Eq. (47)
    filter_temperature_std_dev
        standard deviation (1 sigma), decoded to Celsius degrees using Eq. (51)
    hv_voltage_average
        block-averaged value, decoded to volts using Eq. (47)
    hv_voltage_std_dev
        standard deviation (1 sigma), decoded to volts using Eq. (51)
    spin_period_average
        block-averaged onboard value, decoded to seconds using Eq. (47)
    spin_period_std_dev
        standard deviation (1 sigma), decoded to seconds using Eq. (51)
    pulse_length_average
        block-averaged value, decoded to μs using Eq. (47)
    pulse_length_std_dev
    standard deviation (1 sigma), decoded to μs using Eq. (51)
    glows_start_time
        GLOWS clock, subseconds as decimal part of float
    glows_time_offset
        GLOWS clock, subseconds as decimal part of float
    imap_start_time
        IMAP clock, subseconds as decimal part of float
    imap_time_offset
        IMAP clock, subseconds as decimal part of float
    histogram_flag_array
        flags for bad-time information per bin, consisting of [is_close_to_uv_source,
        is_inside_excluded_region, is_excluded_by_instr_team, is_suspected_transient]
    spin_period_ground_average
        block-averaged value computed on ground
    spin_period_ground_std_dev
        standard deviation (1 sigma)
    position_angle_offset_average
        block-averaged value in degrees
    position_angle_offset_std_dev
        standard deviation (1 sigma)
    spin_axis_orientation_std_dev
        standard deviation( 1 sigma): ∆λ, ∆φ for ⟨λ⟩, ⟨φ⟩
    spin_axis_orientation_average
        block-averaged spin-axis ecliptic longitude ⟨λ⟩ and latitude ⟨φ⟩ in degrees
    spacecraft_location_average
        block-averaged Cartesian ecliptic coordinates ⟨X⟩, ⟨Y ⟩, ⟨Z⟩ [km] of IMAP
    spacecraft_location_std_dev
        standard deviations (1 sigma) ∆X, ∆Y , ∆Z for ⟨X⟩, ⟨Y ⟩, ⟨Z⟩
    spacecraft_velocity_average
        block-averaged values ⟨VX⟩, ⟨VY⟩, ⟨VZ⟩ [km/s] of IMAP velocity components
        (Cartesian ecliptic frame)
    spacecraft_velocity_std_dev
        standard deviations (1 sigma) ∆VX , ∆VY , ∆VZ for ⟨VX ⟩, ⟨VY ⟩, ⟨VZ ⟩
    flags
        flags for extra information, per histogram. This should be a human-readable
        structure.
    """

    histogram: np.ndarray
    flight_software_version: str
    seq_count_in_pkts_file: int
    # ancillary_data_files: np.ndarray TODO Add this
    first_spin_id: int
    last_spin_id: int
    flags_set_onboard: int  # TODO: this should be renamed in L1B
    is_generated_on_ground: int
    number_of_spins_per_block: int
    number_of_bins_per_histogram: int
    number_of_events: int
    filter_temperature_average: np.double
    filter_temperature_variance: InitVar[np.double]
    filter_temperature_std_dev: np.double = field(init=False)
    hv_voltage_average: np.double
    hv_voltage_variance: InitVar[np.double]
    hv_voltage_std_dev: np.double = field(init=False)
    spin_period_average: np.double
    spin_period_variance: InitVar[np.double]
    spin_period_std_dev: np.double = field(init=False)
    pulse_length_average: np.double
    pulse_length_variance: InitVar[np.double]
    pulse_length_std_dev: np.double = field(init=False)
    imap_start_time: np.double  # No conversion needed from l1a->l1b
    imap_time_offset: np.double  # No conversion needed from l1a->l1b
    glows_start_time: np.double  # No conversion needed from l1a->l1b
    glows_time_offset: np.double  # No conversion needed from l1a->l1b
    # unique_block_identifier: str = field(
    #     init=False
    # )  # Could be datetime TODO: Can't put a string in data
    imap_spin_angle_bin_cntr: np.ndarray = field(init=False)  # Same size as bins
    histogram_flag_array: np.ndarray = field(init=False)
    spin_period_ground_average: np.double = field(init=False)  # retrieved from SPICE?
    spin_period_ground_std_dev: np.double = field(init=False)  # retrieved from SPICE?
    position_angle_offset_average: np.double = field(init=False)  # retrieved from SPICE
    position_angle_offset_std_dev: np.double = field(init=False)  # from SPICE
    spin_axis_orientation_std_dev: np.double = field(init=False)  # from SPICE
    spin_axis_orientation_average: np.double = field(init=False)  # retrieved from SPICE
    spacecraft_location_average: np.ndarray = field(init=False)  # retrieved from SPIC
    spacecraft_location_std_dev: np.ndarray = field(init=False)  # retrieved from SPIC
    spacecraft_velocity_average: np.ndarray = field(init=False)  # retrieved from SPIC
    spacecraft_velocity_std_dev: np.ndarray = field(init=False)  # retrieved from SPIC
    flags: np.ndarray = field(init=False)
    # TODO:
    # - Determine a good way to output flags as "human readable"
    # - Add spice pieces
    # - also unique identifiers
    # - Bad angle algorithm using SPICE locations
    # - Move ancillary file to AWS

    def __post_init__(
        self,
        filter_temperature_variance: np.double,
        hv_voltage_variance: np.double,
        spin_period_variance: np.double,
        pulse_length_variance: np.double,
    ) -> None:
        """
        Will process data.

        The input variance values are used to calculate the output standard deviation.

        Parameters
        ----------
        filter_temperature_variance : numpy.double
            Encoded filter temperature variance.
        hv_voltage_variance : numpy.double
            Encoded HV voltage variance.
        spin_period_variance : numpy.double
            Encoded spin period variance.
        pulse_length_variance : numpy.double
            Encoded pulse length variance.
        """
        # self.histogram_flag_array = np.zeros((2,))

        # TODO: These pieces will need to be filled in from SPICE kernels. For now,
        #  they are placeholders. GLOWS example code has better placeholders if needed.
        self.spin_period_ground_average = np.double(-999.9)
        self.spin_period_ground_std_dev = np.double(-999.9)
        self.position_angle_offset_average = np.double(-999.9)
        self.position_angle_offset_std_dev = np.double(-999.9)
        self.spin_axis_orientation_std_dev = np.double(-999.9)
        self.spin_axis_orientation_average = np.double(-999.9)
        self.spacecraft_location_average = np.array([-999.9, -999.9, -999.9])
        self.spacecraft_location_std_dev = np.array([-999.9, -999.9, -999.9])
        self.spacecraft_velocity_average = np.array([-999.9, -999.9, -999.9])
        self.spacecraft_velocity_std_dev = np.array([-999.9, -999.9, -999.9])
        # Will require some additional inputs
        self.imap_spin_angle_bin_cntr = np.zeros((3600,))

        # TODO: This should probably be an AWS file
        # TODO Pass in AncillaryParameters object instead of reading here.
        with open(
            Path(__file__).parents[1] / "ancillary" / "l1b_conversion_table_v001.json"
        ) as f:
            self.ancillary_parameters = AncillaryParameters(json.loads(f.read()))

        self.filter_temperature_average = self.ancillary_parameters.decode(
            "filter_temperature", self.filter_temperature_average
        )
        self.filter_temperature_std_dev = self.ancillary_parameters.decode_std_dev(
            "filter_temperature", filter_temperature_variance
        )

        self.hv_voltage_average = self.ancillary_parameters.decode(
            "hv_voltage", self.hv_voltage_average
        )
        self.hv_voltage_std_dev = self.ancillary_parameters.decode_std_dev(
            "hv_voltage", hv_voltage_variance
        )
        self.spin_period_average = self.ancillary_parameters.decode(
            "spin_period", self.spin_period_average
        )
        self.spin_period_std_dev = self.ancillary_parameters.decode_std_dev(
            "spin_period", spin_period_variance
        )
        self.pulse_length_average = self.ancillary_parameters.decode(
            "pulse_length", self.pulse_length_average
        )
        self.pulse_length_std_dev = self.ancillary_parameters.decode_std_dev(
            "pulse_length", pulse_length_variance
        )

        self.histogram_flag_array = np.zeros((4, 3600), dtype=np.uint8)
        # self.unique_block_identifier = np.datetime_as_string(
        #     np.datetime64(int(self.imap_start_time), "ns"), "s"
        # )
        self.flags = np.ones((FLAG_LENGTH,), dtype=np.uint8)

    def output_data(self) -> tuple:
        """
        Output the L1B DataArrays as a tuple.

        It is faster to return the values like this than to use to_dict() from
        dataclasses.

        Returns
        -------
        tuple
            A tuple containing each attribute value in the class.
        """
        return tuple(getattr(self, out.name) for out in dataclasses.fields(self))

    @staticmethod
    def deserialize_flags(raw: int) -> np.ndarray[int]:
        """
        Deserialize the flags into a list.

        Parameters
        ----------
        raw : int
            16 bit integer containing the on-board flags to deserialize.

        Returns
        -------
        flags : np.ndarray
            Array of flags as a boolean.
        """
        # there are only 10 flags in the on-board flag array, additional flags are added
        # later.
        flags: np.ndarray[bool] = np.array(
            [bool((raw >> i) & 1) for i in range(10)], dtype=bool
        )

        return flags
