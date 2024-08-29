"""
Perform IDEX l2 Processing.

TODO Finish Docstring.
"""

# ruff: noqa: PLR0913

import logging
from pathlib import Path

import lmfit
import numpy as np
import xarray as xr

# from cdflib.xarray import xarray_to_cdf
from lmfit.model import ModelResult
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, find_peaks
from scipy.special import erfc

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.idex import constants


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
    L2 processing class.

    Parameters
    ----------
    l1_file : str
        File path to l1 file being processed to l2 data.
    data_version : str
        The version of the data product being created.
    """

    def __init__(self, l1_file: str, data_version: str):
        """
        Function/method description.

        Parameters
        ----------
        l1_file : str
            File path to l1 file being processed to l2 data.
        data_version : str
            The version of the data product being created.
        """
        self.l1_file = l1_file

        # Switched from cdf_to_xarray to load_cdf function
        # Now l1 data is stored in a xarray.
        self.l1_data = load_cdf(Path(l1_file))

        # TODO: Perhaps I want to make this just code inside the
        #  init rather than a function?
        self.l2_attrs = get_idex_attrs(data_version)

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

    def model_fitter(
        self,
        variable: str,
        amplitude_calibration: float,
        butterworth_filter: bool = False,
    ) -> xr:
        """
        Function/method description.

        Parameters
        ----------
        variable : str
            Something.

        amplitude_calibration : float
            Something.

        butterworth_filter : bool
            Something.

        Returns
        -------
        xr.concat : xarray
            Something.
        """
        model_fit_list = []
        for impact in self.l1_data[variable]:
            epoch_xr = xr.DataArray(
                name="epoch",
                # TODO: What should the impact time be? in RawDustEvents
                data=[impact["epoch"].data],
                dims="epoch",
                # TODO: Double check that this is correct
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

            variable_lower = variable.lower()

            time_of_impact_fit_xr = xr.DataArray(
                # Target_High_model_time_of_impact
                # Ion_Grid_model_time_of_impact
                name=f"{variable}_model_time_of_impact",
                data=[time_of_impact_fit],
                dims="epoch",
                # TODO: attrs
                attrs=self.l2_attrs.get_variable_attributes(
                    f"{variable_lower}_model_time_of_impact"
                ),
            )

            constant_offset_fit_xr = xr.DataArray(
                # Target_High_model_constant_offset
                # Ion_Grid_model_time_of_impact
                name=f"{variable}_model_constant_offset",
                data=[constant_offset_fit],
                dims="epoch",
                # TODO: attrs
                attrs=self.l2_attrs.get_variable_attributes(
                    f"{variable_lower}_model_constant_offset"
                ),
            )

            amplitude_fit_xr = xr.DataArray(
                # Target_High_model_amplitude
                # Ion_Grid_model_amplitude
                name=f"{variable}_model_amplitude",
                data=[amplitude_fit],
                dims="epoch",
                # TODO: attrs
                attrs=self.l2_attrs.get_variable_attributes(
                    f"{variable_lower}_model_amplitude"
                ),
            )

            rise_time_fit_xr = xr.DataArray(
                # Target_High_model_rise_time
                # Ion_Grid_model_rise_time
                name=f"{variable}_model_rise_time",
                data=[rise_time_fit],
                dims="epoch",
                # TODO: attrs
                attrs=self.l2_attrs.get_variable_attributes(
                    f"{variable_lower}_model_rise_time"
                ),
            )

            discharge_time_xr = xr.DataArray(
                # Target_High_model_discharge_time
                # Ion_Grid_model_discharge_time
                name=f"{variable}_model_discharge_time",
                data=[discharge_time_fit],
                dims="epoch",
                # TODO: attrs
                attrs=self.l2_attrs.get_variable_attributes(
                    f"{variable_lower}_model_discharge_time"
                ),
            )

            fit_uncertainty_xr = xr.DataArray(
                # Target_High_model_uncertainty
                # Ion_Grid_model_uncertainty
                name=f"{variable}_model_uncertainty",
                data=[fit_uncertainty],
                dims="epoch",
                # TODO: attrs
                attrs=self.l2_attrs.get_variable_attributes(
                    f"{variable_lower}_model_uncertainty"
                ),
            )

            model_fit_list.append(
                xr.Dataset(
                    data_vars={
                        f"{variable}_model_time_Of_impact": time_of_impact_fit_xr,
                        f"{variable}_model_constant_offset": constant_offset_fit_xr,
                        f"{variable}_model_amplitude": amplitude_fit_xr,
                        f"{variable}_model_rise_time": rise_time_fit_xr,
                        f"{variable}_model_discharge_time": discharge_time_xr,
                        f"{variable}_model_uncertainty": fit_uncertainty_xr,
                    },
                    coords={"epoch": epoch_xr},
                )
            )

        return xr.concat(model_fit_list, dim="epoch")

    @staticmethod
    def idex_response_function(
        x: int,
        time_of_impact: int,
        constant_offset: int,
        amplitude: int,
        rise_time: int,
        discharge_time: int,
    ) -> float:
        """
        Function/method description.

        Parameters
        ----------
        x : int
            Something.

        time_of_impact : int
            Something.

        constant_offset : int
            Something.

        amplitude : int
            Something.

        rise_time : int
            Something.

        discharge_time : int
            Something.

        Returns
        -------
        result : Union[int, float]
            Something.
        """
        heaviside = np.heaviside(x - time_of_impact, 0)
        exponent_1 = 1.0 - np.exp(-(x - time_of_impact) / rise_time)
        exponent_2 = np.exp(-(x - time_of_impact) / discharge_time)
        result: float = constant_offset + (
            heaviside * amplitude * exponent_1 * exponent_2
        )
        return result

    # fmt: skip

    # Create a model for exponentially modified Gaussian
    @staticmethod
    def expgaussian(
        x: int, amplitude: int, center: int, sigma: int, gamma: int
    ) -> float:
        """
        Function/method description.

        Parameters
        ----------
        x : int
            Something.

        amplitude : int
            Something.

        center : int
            Something.

        sigma : int
            Something.

        gamma : int
            Something.

        Returns
        -------
        result : float
            Something.
        """
        dx = center - x

        result: float = amplitude * np.exp(gamma * dx) * erfc(dx / (np.sqrt(2) * sigma))
        return result

    @staticmethod
    def butter_lowpass_filter(data: str, time: list) -> np.ndarray:
        """
        Function/method description.

        Parameters
        ----------
        data : str
            Something.

        time : list
            Something.

        Returns
        -------
        y : np.ndarray
            Something.
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
        y: np.ndarray = filtfilt(b, a, data)
        return y

    def fit_tof_model(self, variable: str, peak_prominence: int) -> xr:
        """
        Function/method description.

        Parameters
        ----------
        variable : str
            Something.

        peak_prominence : int
            Something.

        Returns
        -------
        xr.concat : xarray
            Something.
        """
        mass_number_xr = xr.DataArray(
            name="mass_number",
            data=np.linspace(1, 50, 50),
            dims="mass_number",
            # TODO: check this is correct
            attrs=self.l2_attrs.get_variable_attributes("mass_number_attrs"),
        )

        tof_model_parameters_list = []
        for impact in self.l1_data[variable]:
            epoch_xr = xr.DataArray(
                name="epoch",
                # TODO: What should the impact time be? in RawDustEvents
                data=[impact["epoch"].data],
                dims="epoch",
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
                    ).best_values
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

            variable_lower = variable.lower()

            amplitude_xr = xr.DataArray(
                # TOF_Low_model_masses_amplitude
                name=f"{variable}_model_masses_amplitude",
                data=[mass_amplitudes],
                dims=("epoch", "mass_number"),
                # TODO: Attrs
                attrs=self.l2_attrs.get_variable_attributes(
                    f"{variable_lower}_model_masses_amplitude"
                ),
            )

            center_xr = xr.DataArray(
                # TOF_Low_model_masses_center
                name=f"{variable}_model_masses_center",
                data=[mass_centers],
                dims=("epoch", "mass_number"),
                # TODO: Attrs
                attrs=self.l2_attrs.get_variable_attributes(
                    f"{variable_lower}_model_masses_center"
                ),
            )

            sigma_xr = xr.DataArray(
                # TOF_Low_model_masses_sigma
                name=f"{variable}_model_masses_sigma",
                data=[mass_sigmas],
                dims=("epoch", "mass_number"),
                # TODO: Attrs
                attrs=self.l2_attrs.get_variable_attributes(
                    f"{variable_lower}_model_masses_sigma"
                ),
            )

            gamma_xr = xr.DataArray(
                # TOF_Low_model_masses_gamma
                name=f"{variable}_model_masses_gamma",
                data=[mass_gammas],
                dims=("epoch", "mass_number"),
                # TODO: Attrs
                attrs=self.l2_attrs.get_variable_attributes(
                    f"{variable_lower}_model_masses_gamma"
                ),
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
    def fit_expgaussian(self, x: str, y: str) -> ModelResult:
        """
        Function/method description.

        Parameters
        ----------
        x : str
            Something.

        y : str
            Something.

        Returns
        -------
        result.best_value : dict
            Something.
        """
        model = lmfit.Model(self.expgaussian)
        params = model.make_params(
            amplitude=max(y), center=x[np.argmax(y)], sigma=10.0, gamma=10.0
        )
        result = model.fit(y, params, x=x)
        return result

    # def write_l2_cdf(self) -> Path:
    #     """
    #     Function/method description.
    #
    #     Returns
    #     -------
    #     l2_file_name : Path
    #         The file name of the l2 file.
    #     """
    #
    #     return write_cdf(self.l2_data)

    def process_idex_l2(self, l1_file: str, data_version: str) -> str:
        """
        Function/method description.

        Parameters
        ----------
        l1_file : str
            Something.

        data_version : str
            Something.

        Returns
        -------
        l2_cdf_file_name : str
            The file name of the l2 cdf.

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

        return str(write_cdf(l2_data.l2_data))
