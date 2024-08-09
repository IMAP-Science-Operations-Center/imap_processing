"""
Perform IDEX l2 Processing.

This module processes decommutated IDEX packets and creates l2 data products.
"""

import logging
from pathlib import Path

import lmfit
import numpy as np
import xarray as xr
from cdflib.xarray import xarray_to_cdf
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, find_peaks
from scipy.special import erfc

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.idex import constants

# from . import idex_cdf_attrs


def get_idex_attrs(data_version: str) -> ImapCdfAttributes:
    """
    Load in CDF attributes for IDEX l2 files.

    Parameters
    ----------
    data_version : str
        Data version for CDF filename, in the format "vXXX".

    Returns
    -------
    idex_attrs : ImapCdfAttributes
        Imap object with l1a attribute files loaded in.
    """
    idex_attrs = ImapCdfAttributes()
    idex_attrs.add_instrument_global_attrs("idex")
    idex_attrs.add_instrument_variable_attrs("idex", "l2")
    idex_attrs.add_global_attribute("Data_version", data_version)
    return idex_attrs


class L2Processor:
    """
    Example
    -------
    from imap_processing.idex.idex_packet_parser import PacketParser
    from imap_processing.idex.l2_processing import L2Processor

    l0_file = "imap_processing/idex/tests/imap_idex_l0_20230725_v01-00.pkts"
    l1_data = PacketParser(l0_file)
    l1_data.write_cdf_file("20230725")

    l2_data = L2Processor('imap_idex_l1_20230725_v01.cdf')
    l2_data.write_l2_cdf()
    """

    def __init__(self, l1_file: str, data_version: str):
        """
        Function/method description.

        Parameters
        ----------
        l1_file: str
            File path to l1 file being processed to l2 data
        data_version : str
            The version of the data product being created.
        """
        self.l1_file = l1_file

        # Switched from cdf_to_xarray to load_cdf function
        self.l1_data = load_cdf(Path(l1_file))

        target_signal_model_dataset = self.model_fitter(
            "Target_High", constants.TARGET_HIGH_CALIBRATION, butterworth_filter=False
        )
        ion_grid_model_dataset = self.model_fitter(
            "Ion_Grid", constants.TARGET_HIGH_CALIBRATION, butterworth_filter=True
        )
        tof_model_dataset = self.fit_tof_model(
            "TOF_Low", peak_prominence=constants.TOF_Low_Peak_Prominence
        )

        self.l2_data = xr.merge(
            [
                self.l1_data,
                target_signal_model_dataset,
                ion_grid_model_dataset,
                tof_model_dataset,
            ]
        )

        # TODO: Perhaps I want to make this just code inside the
        #  init rather than a function?
        self.l2_attrs = get_idex_attrs(data_version)

    def model_fitter(
        self,
        variable: str,
        amplitude_calibration: float,
        butterworth_filter: bool = False,
    ):
        """
        Function/method description.

        Parameters
        ----------
        variable: str
            Something

        amplitude_calibration: float
            Something

        butterworth_filter: bool
            Something

        Returns
        -------
        xr.concat
            Something

        """
        model_fit_list = []
        for impact in self.l1_data[variable]:
            epoch_xr = xr.DataArray(
                name="epoch",
                # TODO: What should the impact time be? in RawDustEvents
                data=[impact["epoch"].data],
                dims=("epoch"),
                attrs=self.l2_attrs.get_variable_attributes("epoch"),
            )

            x = impact[impact.attrs["DEPEND_1"]].data
            y = impact.data - np.mean(impact.data[0:10])
            try:
                model = lmfit.Model(self.idex_response_function)
                params = model.make_params(
                    time_of_impact=constants.time_of_impact_init,
                    constant_offset=constants.constant_offset_init,
                    amplitude=max(y),
                    rise_rime=constants.rise_time_init,
                    discharge_time=constants.discharge_time_init,
                )
                params["rise_time"].min = 5
                params["rise_time"].max = 10000

                if butterworth_filter:
                    y = self.butter_lowpass_filter(y, x)

                result = model.fit(y, params, x=x)

                param = result.best_values

                _, param_cov, _, _, _ = curve_fit(self.idex_response_function, x, y)
                fit_uncertainty = np.linalg.det(param_cov)

                time_of_impact_fit = param["time_of_impact"]
                constant_offset_fit = param["constant_offset"]
                amplitude_fit = amplitude_calibration * param["amplitude"]
                rise_time_fit = param["rise_time"]
                discharge_time_fit = param["discharge_time"]

            except Exception as e:
                logging.warning(
                    "Error fitting Models, resorting to FILLVALs: " + str(e)
                )
                time_of_impact_fit = constants.FILLVAL
                constant_offset_fit = constants.FILLVAL
                amplitude_fit = constants.FILLVAL
                rise_time_fit = constants.FILLVAL
                discharge_time_fit = constants.FILLVAL
                fit_uncertainty = constants.FILLVAL

            time_of_impact_fit_xr = xr.DataArray(
                name=f"{variable}_Model_Time_Of_Impact",
                data=[time_of_impact_fit],
                dims=("epoch"),
                # TODO: attrs
            )

            constant_offset_fit_xr = xr.DataArray(
                name=f"{variable}_Model_Constant_Offset",
                data=[constant_offset_fit],
                dims=("epoch"),
                # TODO: attrs
            )

            amplitude_fit_xr = xr.DataArray(
                name=f"{variable}_Model_Amplitude",
                data=[amplitude_fit],
                dims=("epoch"),
                # TODO: attrs
            )

            rise_time_fit_xr = xr.DataArray(
                name=f"{variable}_Model_Rise_time",
                data=[rise_time_fit],
                dims=("epoch"),
                # TODO: attrs
            )

            discharge_time_xr = xr.DataArray(
                name=f"{variable}_Model_Discharge_time",
                data=[discharge_time_fit],
                dims=("epoch"),
                # TODO: attrs
            )

            fit_uncertainty_xr = xr.DataArray(
                name=f"{variable}_Model_Uncertainty",
                data=[fit_uncertainty],
                dims=("epoch"),
                # TODO: attrs
            )

            model_fit_list.append(
                xr.Dataset(
                    data_vars={
                        f"{variable}_Model_Time_Of_Impact": time_of_impact_fit_xr,
                        f"{variable}_Model_Constant_Offset": constant_offset_fit_xr,
                        f"{variable}_Model_Amplitude": amplitude_fit_xr,
                        f"{variable}_Model_Rise_Time": rise_time_fit_xr,
                        f"{variable}_Model_Discharge_Time": discharge_time_xr,
                        f"{variable}_Model_Uncertainty": fit_uncertainty_xr,
                    },
                    coords={"epoch": epoch_xr},
                )
            )

        return xr.concat(model_fit_list, dim="epoch")

    @staticmethod
    def idex_response_function(
        x, time_of_impact, constant_offset, amplitude, rise_time, discharge_time
    ):
        """
        Docstring.
        """
        heaviside = np.heaviside(x - time_of_impact, 0)
        exponent_1 = 1.0 - np.exp(-(x - time_of_impact) / rise_time)
        exponent_2 = np.exp(-(x - time_of_impact) / discharge_time)
        return constant_offset + (heaviside * amplitude * exponent_1 * exponent_2)

    # fmt: skip

    # Create a model for exponentially modified Gaussian
    @staticmethod
    def expgaussian(x, amplitude, center, sigma, gamma):
        """
        Docstring.
        """
        dx = center - x
        return amplitude * np.exp(gamma * dx) * erfc(dx / (np.sqrt(2) * sigma))

    @staticmethod
    def butter_lowpass_filter(data, time):
        """
        Docstring.
        """
        # Filter requirements.
        t = time[1] - time[0]  # |\Sample Period (s)
        fs = (time[-1] - time[0]) / t  # ||sample rate, Hz
        cutoff = 10  # ||desired cutoff frequency of the filter, Hz
        nyq = 0.5 * fs  # ||Nyquist Frequency
        order = 2  # ||sine wave can be approx represented as quadratic
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a, _ = butter(order, normal_cutoff, btype="low", analog=False)
        y = filtfilt(b, a, data)
        return y

    def fit_tof_model(self, variable, peak_prominence):
        """
        Function/method description.

        Parameters
        ----------
        variable: str
            Something

        peak_prominence
            Something
        """
        mass_number_xr = xr.DataArray(
            name="mass_number",
            data=np.linspace(1, 50, 50),
            dims=("mass_number"),
            # TODO: Attrs
            # attrs = self.l2_attrs.get_variable_attributes((mass_number_attrs))
        )

        tof_model_parameters_list = []
        for impact in self.l1_data[variable]:
            epoch_xr = xr.DataArray(
                name="epoch",
                # TODO: What should the impact time be? in RawDustEvents
                data=[impact["epoch"].data],
                dims=("epoch"),
                attrs=self.l2_attrs.get_variable_attributes("epoch"),
            )

            mass_amplitudes = np.full(50, constants.FILLVAL)
            mass_centers = np.full(50, constants.FILLVAL)
            mass_sigmas = np.full(50, constants.FILLVAL)
            mass_gammas = np.full(50, constants.FILLVAL)

            x = impact[impact.attrs["DEPEND_1"]].data
            y = impact.data

            peaks, _ = find_peaks(y, prominence=peak_prominence)
            i = 0
            for peak in peaks:
                try:
                    i += 1
                    fit_params = self.fit_expgaussian(
                        x[peak - 10 : peak + 10], y[peak - 10 : peak + 10]
                    )
                    (
                        mass_amplitudes[i],
                        mass_centers[i],
                        mass_sigmas[i],
                        mass_gammas[i],
                    ) = tuple(fit_params.values())
                except Exception as e:
                    logging.warning(
                        "Error fitting TOF Model.  Defaulting to FILLVALS. " + str(e)
                    )

            amplitude_xr = xr.DataArray(
                name=f"{variable}_model_masses_amplitude",
                data=[mass_amplitudes],
                dims=("epoch", "mass_number"),
                # TODO: Attrs
            )

            center_xr = xr.DataArray(
                name=f"{variable}_model_masses_center",
                data=[mass_centers],
                dims=("epoch", "mass_number"),
                # TODO: Attrs
            )

            sigma_xr = xr.DataArray(
                name=f"{variable}_model_masses_sigma",
                data=[mass_sigmas],
                dims=("epoch", "mass_number"),
                # TODO: Attrs
            )

            gamma_xr = xr.DataArray(
                name=f"{variable}_model_masses_gamma",
                data=[mass_gammas],
                dims=("epoch", "mass_number"),
                # TODO: Attrs
            )

            tof_model_parameters_list.append(
                xr.Dataset(
                    data_vars={
                        f"{variable}_model_masses_amplitude": amplitude_xr,
                        f"{variable}_model_masses_center": center_xr,
                        f"{variable}_model_masses_sigma": sigma_xr,
                        f"{variable}_model_Mmsses_gamma": gamma_xr,
                    },
                    coords={"epoch": epoch_xr, "mass_number": mass_number_xr},
                )
            )

        return xr.concat(tof_model_parameters_list, dim="epoch")

    # Fit the exponentially modified Gaussian
    def fit_expgaussian(self, x, y):
        """
        Function/method description.
        """
        model = lmfit.Model(self.expgaussian)
        params = model.make_params(
            amplitude=max(y), center=x[np.argmax(y)], sigma=10.0, gamma=10.0
        )
        result = model.fit(y, params, x=x)
        return result.best_values

    def write_l2_cdf(self):
        """
        Function/method description.
        """
        # TODO: Do I need a get_global_attributes line here?
        # self.l2_data.attrs = idex_cdf_attrs.idex_l2_global_attrs

        for var in self.l2_data:
            if "_model_amplitude" in var:
                self.l2_data[var].attrs = {
                    "CATDESC": var,
                    "FIELDNAM": var,
                    "LABLAXIS": var,
                    "VAR_NOTES": f"The amplitude of the response for "
                    f"{var.replace('_model_amplitude', '')}",
                }
                # | idex_cdf_attrs.model_amplitude_base
                # self.l2_attrs.get_variable_attributes("model_amplitude_base")

            if "_model_uncertainty" in var:
                self.l2_data[var].attrs = {
                    "CATDESC": var,
                    "FIELDNAM": var,
                    "LABLAXIS": var,
                    "VAR_NOTES": f"The uncertainty in the model of the response for "
                    f"{var.replace('_model_amplitude', '')}",
                }
                # | idex_cdf_attrs.model_dimensionless_base
                # self.l2_attrs.get_variable_attributes("model_dimensionless_base")

            if "_model_constant_offset" in var:
                self.l2_data[var].attrs = {
                    "CATDESC": var,
                    "FIELDNAM": var,
                    "VAR_NOTES": f"The constant offset of the response for "
                    f"{var.replace('_model_constant_offset', '')}",
                }
                # | idex_cdf_attrs.model_amplitude_base
                # self.l2_attrs.get_variable_attributes("model_amplitude_base")

            if "_model_time_of_impact" in var:
                self.l2_data[var].attrs = {
                    "CATDESC": var,
                    "FIELDNAM": var,
                    "LABLAXIS": var,
                    "VAR_NOTES": f"The time of impact for "
                    f"{var.replace('_model_time_of_impact', '')}",
                }
                # | idex_cdf_attrs.model_time_base
                # self.l2_attrs.get_variable_attributes("model_time_base")

            if "_model_rise_time" in var:
                self.l2_data[var].attrs = {
                    "CATDESC": var,
                    "FIELDNAM": var,
                    "LABLAXIS": var,
                    "VAR_NOTES": f"The rise time of the response for "
                    f"{var.replace('_model_rise_time', '')}",
                }
                # | idex_cdf_attrs.model_time_base
                # self.l2_attrs.get_variable_attributes("model_time_base")

            if "_model_discharge_time" in var:
                self.l2_data[var].attrs = {
                    "CATDESC": var,
                    "FIELDNAM": var,
                    "LABLAXIS": var,
                    "VAR_NOTES": f"The discharge time of the response for "
                    f"{var.replace('_model_discharge_time', '')}",
                }
                # | idex_cdf_attrs.model_time_base
                # self.l2_attrs.get_variable_attributes("model_time_base")
            if "_model_masses_amplitude" in var:
                self.l2_data[var].attrs = {
                    "CATDESC": var,
                    "FIELDNAM": var,
                    "LABLAXIS": var,
                    "VAR_NOTES": f"The amplitude of the first 50 peaks in "
                    f"{var.replace('_Model_Masses_Amplitude', '')}",
                }
                # | idex_cdf_attrs.tof_model_amplitude_base
                # self.l2_attrs.get_variable_attributes("tof_model_amplitude_base")

            if "_model_masses_center" in var:
                self.l2_data[var].attrs = {
                    "CATDESC": var,
                    "FIELDNAM": var,
                    "LABLAXIS": var,
                    "VAR_NOTES": f"The center of the first 50 peaks in "
                    f"{var.replace('_model_masses_center', '')}",
                }
                # | idex_cdf_attrs.tof_model_dimensionless_base
                # self.l2_attrs.get_variable_attributes("tof_model_amplitude_base")

            if "_model_masses_sigma" in var:
                self.l2_data[var].attrs = {
                    "CATDESC": var,
                    "FIELDNAM": var,
                    "LABLAXIS": var,
                    "VAR_NOTES": f"The sigma of the fitted exponentially modified "
                    f"gaussian to the first 50 peaks in {var.replace('_model_masses_sigma', '')}",
                }
                # | idex_cdf_attrs.tof_model_dimensionless_base
                # self.l2_attrs.get_variable_attributes("tof_model_dimensionless_base")

            if "_model_masses_gamma" in var:
                self.l2_data[var].attrs = {
                    "CATDESC": var,
                    "FIELDNAM": var,
                    "LABLAXIS": var,
                    "VAR_NOTES": f"The gamma of the fitted exponentially modified "
                    f"gaussian to the first 50 peaks in {var.replace('_model_masses_gamma', '')}",
                }
                # | idex_cdf_attrs.tof_model_dimensionless_base
                # self.l2_attrs.get_variable_attributes("tof_model_dimensionless_base")

        l2_file_name = self.l1_file.replace("_l1_", "_l2_")

        xarray_to_cdf(self.l2_data, l2_file_name)

        return l2_file_name

    def process_idex_l2(self, l1_file: str, data_version: str):
        """
        Function/method description.

        Parameters
        ----------
        l1_file: str
            Something

        data_version: str
            Something

        Returns
        -------
        l2_cdf_file_name

        Notes
        -----
        Example usage ->
            from imap_processing.idex.idex_packet_parser import PacketParser
            from imap_processing.idex.l2_processing import L2Processor

            l0_file = "imap_processing/idex/tests/imap_idex_l0_20230725_v01-00.pkts"
            l1_data = PacketParser(l0_file)
            l1_data.write_cdf_file("20230725")

            l2_data = L2Processor('imap_idex_l1_20230725_v01.cdf')
            l2_data.write_l2_cdf()

        """
        l2_data = L2Processor(l1_file, data_version)

        l2_cdf_file_name = l2_data.write_l2_cdf()

        return l2_cdf_file_name
