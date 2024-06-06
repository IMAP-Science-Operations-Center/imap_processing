"""Module for GLOWS L1B data products."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from imap_processing.glows.l1a.glows_l1a_data import DirectEventL1A


class AncillaryParameters:
    """
    GLOWS L1B Ancillary Parameters for decoding ancillary histogram data points.

    This class reads from a JSON file input which defines ancillary parameters.
    It validates to ensure the input file has all the required parameters.

    Attributes
    ----------
    version: str
        version of the ancillary file
    filter_temp: dict
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

        Parameters
        ----------
        table: dict
            Dictionary generated from input JSON file
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
                "GLOWS L1B Ancillary input_table does not conform to "
                "expected format."
            ) from e


class DirectEventL1B:
    """GLOWS L1B direct event data product."""

    def __init__(self, l1a: DirectEventL1A):
        pass


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
    histograms
        array of block-accumulated count numbers
    flight_software_version: str
    seq_count_in_pkts_file: int
    flags_set_onboard: int
    is_generated_on_ground: int
    number_of_spins_per_block
        nblock
    block_header
        header, see Tab. 12
    unique_block_identifier
        YYYY-MM-DDThh:mm:ss based on IMAP UTC time
    number_of_bins_per_histogram
        nbin
    number_of_events
        total number of events/counts in histogram
    imap_spin_angle_bin_cntr
        IMAP spin angle ψ for bin centers, see Sec. -
    histogram_flag_array
        array of bad-angle flags for histogram bins, see Tab. 14
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
        GLOWS clock, subseconds as decimal part of float, see Sec. -.1
    glows_end_time_offset
        GLOWS clock, subseconds as decimal part of float, see Sec. -.1
    imap_start_time
        IMAP clock, subseconds as decimal part of float, see Sec. -.1
    imap_end_time_offset
        IMAP clock, subseconds as decimal part of float, see Sec. -.1
    spin_period_ground_average
        block-averaged value computed on ground, see Sec. -.1
    spin_period_ground_std_dev
        standard deviation (1 sigma), see Sec. -.1
    position_angle_offset_average
        block-averaged value in degrees, see Sec. - and -.1
    position_angle_offset_std_dev
        standard deviation (1 sigma), see Sec. - and -.1
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

    histograms: np.ndarray
    flight_software_version: str
    # pkts_file_name: str TODO: add this in L0
    seq_count_in_pkts_file: int
    # l1a_file_name: str TODO: add this
    # ancillary_data_files: np.ndarray TODO Add this
    last_spin_id: int
    flags_set_onboard: int  # TODO: this should be renamed in L1B
    is_generated_on_ground: int
    number_of_spins_per_block: int
    number_of_bins_per_histogram: int
    number_of_events: int
    filter_temperature_average: np.single
    filter_temperature_std_dev: np.single
    hv_voltage_average: np.single
    hv_voltage_std_dev: np.single
    spin_period_average: np.single
    spin_period_std_dev: np.single
    pulse_length_average: np.single
    pulse_length_std_dev: np.single
    imap_start_time: np.single  # No conversion needed from l1a->l1b
    imap_end_time_offset: np.single  # No conversion needed from l1a->l1b
    glows_start_time: np.single  # No conversion needed from l1a->l1b
    glows_end_time_offset: np.single  # No conversion needed from l1a->l1b
    unique_block_identifier: str = field(
        init=False
    )  # Could be datetime TODO: Missing from values in L1A
    imap_spin_angle_bin_cntr: np.ndarray = field(
        init=False
    )  # Same size as bins TODO add dims
    histogram_flag_array: np.ndarray = field(init=False)  # TODO add dims
    spin_period_ground_average: np.single = field(init=False)  # retrieved from SPICE?
    spin_period_ground_std_dev: np.single = field(init=False)  # retrieved from SPICE?
    position_angle_offset_average: np.single = field(init=False)  # retrieved from SPICE
    position_angle_offset_std_dev: np.single = field(init=False)  # retrieved from SPICE
    spin_axis_orientation_std_dev: np.single = field(init=False)  # retrieved from SPICE
    spin_axis_orientation_average: np.single = field(init=False)  # retrieved from SPICE
    spacecraft_location_average: np.ndarray = field(init=False)  # retrieved from SPIC
    spacecraft_location_std_dev: np.ndarray = field(init=False)  # retrieved from SPIC
    spacecraft_velocity_average: np.ndarray = field(init=False)  # retrieved from SPIC
    spacecraft_velocity_std_dev: np.ndarray = field(init=False)  # retrieved from SPIC
    # TODO make these human - readable
    # flags: np.ndarray = field(init=False) # Generated per-histogram

    # TODO:
    # - Determine a good way to output flags as "human readable"
    # - Add spice pieces
    # - add in the filenames for the input files
    # - Bad angle algorithm using SPICE locations
    # - Move ancillary file to AWS

    def __post_init__(self):
        """Process data."""
        # self.histogram_flag_array = np.zeros((2,))

        # TODO: These pieces will need to be filled in from SPICE kernels. For now,
        #  they are placeholders. GLOWS example code has better placeholders if needed.
        self.spin_period_ground_average = np.single(-999.9)
        self.spin_period_ground_std_dev = np.single(-999.9)
        self.position_angle_offset_average = np.single(-999.9)
        self.position_angle_offset_std_dev = np.single(-999.9)
        self.spin_axis_orientation_std_dev = np.single(-999.9)
        self.spin_axis_orientation_average = np.single(-999.9)
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

        self.filter_temperature_average = self.decode_ancillary_data(
            self.ancillary_parameters.filter_temperature,
            self.filter_temperature_average,
        )
        self.filter_temperature_std_dev = self.decode_ancillary_data(
            self.ancillary_parameters.filter_temperature,
            self.filter_temperature_std_dev,
        )

        self.histogram_flag_array = np.zeros((17, 3600))
        self.unique_block_identifier = np.datetime_as_string(
            np.datetime64(int(self.imap_start_time), "ns"), "s"
        )

    @staticmethod
    def decode_ancillary_data(params: dict, encoded_value: np.single) -> np.single:
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
        params : dict
            A dictionary of parameters for decoding the ancillary data. Consists of
            keys ['n_bits', 'max', 'min'].
        encoded_value : float
            The encoded value to decode.

        Returns
        -------
        decoded_value : float
            The decoded value.
        """
        # compute parameters a and b:
        param_a = (2 ** params["n_bits"] - 1) / (params["max"] - params["min"])
        param_b = -params["min"] * param_a

        decoded_value: np.single = (encoded_value - param_b) / param_a

        return decoded_value
