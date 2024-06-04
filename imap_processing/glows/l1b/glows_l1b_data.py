"""Module for GLOWS L1B data products."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imap_processing.glows.l1a.glows_l1a_data import DirectEventL1A, HistogramL1A


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

    Attributes
    ----------
    block_header
        header, see Tab. 12
    unique_block_identifier
        YYYY-MM-DDThh:mm:ss based on IMAP UTC time
    glows_start_time
        GLOWS clock, subseconds as decimal part of float, see Sec. 12.2.1
    glows_end_time_offset
        GLOWS clock, subseconds as decimal part of float, see Sec. 12.2.1
    imap_start_time
        IMAP clock, subseconds as decimal part of float, see Sec. 12.2.1
    imap_end_time_offset
        IMAP clock, subseconds as decimal part of float, see Sec. 12.2.1
    number_of_spins_per_block
        nblock
    number_of_bins_per_histogram
        nbin
    histogram
        array of block-accumulated count numbers
    number_of_events
        total number of events/counts in histogram
    imap_spin_angle_bin_cntr
        IMAP spin angle ψ for bin centers, see Sec. 10.6
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
    spin_period_ground_average
        block-averaged value computed on ground, see Sec. 12.6.1
    spin_period_ground_std_dev
        standard deviation (1 sigma), see Sec. 12.6.1
    position_angle_offset_average
        block-averaged value in degrees, see Sec. 10.6 and 12.6.1
    position_angle_offset_std_dev
        standard deviation (1 sigma), see Sec. 10.6 and 12.6.1
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
        flags for extra information see Tab. 13 see Sec. 12.3
    """

    # TODO: map-reduce style for creating xarray -> L1B data -> xarray
    block_header: dict  # could be a new type
    unique_block_identifier: str  # Could be datetime
    glows_start_time: np.single
    glows_end_time_offset: np.single
    imap_start_time: np.single
    imap_end_time_offset: np.single
    number_of_spins_per_block: int
    number_of_bins_per_histogram: int
    histogram: np.ndarray
    number_of_events: int
    imap_spin_angle_bin_cntr: np.single
    histogram_flag_array: np.ndarray
    filter_temperature_average: np.single
    filter_temperature_std_dev: np.single
    hv_voltage_average: np.single
    hv_voltage_std_dev: np.single
    spin_period_average: np.single
    spin_period_std_dev: np.single
    pulse_length_average: np.single
    pulse_length_std_dev: np.single
    spin_period_ground_average: np.single  # retrieved from SPICE?
    spin_period_ground_std_dev: np.single  # retrieved from SPICE?
    position_angle_offset_average: np.single  # retrieved from SPICE
    position_angle_offset_std_dev: np.single  # retrieved from SPICE
    spin_axis_orientation_std_dev: np.single  # retrieved from SPICE
    spin_axis_orientation_average: np.single  # retrieved from SPICE
    spacecraft_location_average: np.ndarray  # retrieved from SPICE
    spacecraft_location_std_dev: np.ndarray  # retrieved from SPICE
    spacecraft_velocity_average: np.ndarray  # retrieved from SPICE
    spacecraft_velocity_std_dev: np.ndarray  # retrieved from SPICE
    flags: np.ndarray
    conversion_table: dict

    # TODO:
    # - unique block identifier for individual histograms
    # - Decode ancillary parameters to physical units
    # - flags
    # - maybe: bat angle flags? (section 12.3.3)

    def __init__(self, l1a: HistogramL1A):
        self.imap_start_time = l1a.imap_start_time.to_float()
        self.imap_time_offset = l1a.imap_time_offset.to_float()
        self.glows_start_time = l1a.glows_start_time.to_float()
        self.glows_time_offset = l1a.glows_time_offset.to_float()

        # TODO: These pieces will need to be filled in from SPICE kernels. For now,
        #  they are placeholders. GLOWS example code has better placeholders if needed.
        self.spin_period_ground_average = -999.0
        self.spin_period_ground_std_dev = -999.0
        self.position_angle_offset_average = -999.0
        self.position_angle_offset_std_dev = -999.0
        self.spin_axis_orientation_std_dev = -999.0
        self.spin_axis_orientation_average = -999.0
        self.spacecraft_location_average = np.array([-999.0, -999.0, -999.0])
        self.spacecraft_location_std_dev = np.array([-999.0, -999.0, -999.0])
        self.spacecraft_velocity_average = np.array([-999.0, -999.0, -999.0])
        self.spacecraft_velocity_std_dev = np.array([-999.0, -999.0, -999.0])

        # TODO: This should probably be an AWS file
        self.ancillary_parameters = AncillaryParameters(
            json.loads(
                str(Path(__file__).parents[1] / "l1b_conversion_table_v001.json")
            )
        )

        self.filter_temperature_average = self.decode_ancillary_data(
            self.ancillary_parameters.filter_temperature, l1a.filter_temperature_average
        )
        self.filter_temperature_std_dev = self.decode_ancillary_data(
            self.ancillary_parameters.filter_temperature, l1a.filter_temperature_std_dev
        )

    @staticmethod
    def decode_ancillary_data(params: dict, encoded_value: float) -> np.single:
        """
        Decode parameters using the algorithm defined in section 11.4.

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

        decoded_value = (encoded_value - param_b) / param_a

        return np.single(decoded_value)
