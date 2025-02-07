"""
SWE L2 processing module.

This module contains functions to process L1B data to L2 data products.
"""

import numpy as np
import numpy.typing as npt
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.spin import get_instrument_spin_phase, get_spin_angle
from imap_processing.swe.utils.swe_utils import (
    ESA_VOLTAGE_ROW_INDEX_DICT,
    read_lookup_table,
)

# TODO: add these to instrument status summary
ENERGY_CONVERSION_FACTOR = 4.75
# 7 CEMs geometric factors in cm^2 sr eV/eV units.
GEOMETRIC_FACTORS = np.array(
    [
        435e-6,
        599e-6,
        808e-6,
        781e-6,
        876e-6,
        548e-6,
        432e-6,
    ]
)
ELECTRON_MASS = 9.10938356e-31  # kg

# See doc string of calculate_phase_space_density() for more details.
VELOCITY_CONVERSION_FACTOR = 1.237e31
# See doc string of calculate_flux() for more details.
FLUX_CONVERSION_FACTOR = 6.187e30

CEM_DETECTORS_ANGLE = np.array([-63, -42, -21, 0, 21, 42, 63])


def get_particle_energy() -> npt.NDArray:
    """
    Get particle energy.

    Calculate particle energy and add to the lookup table.
    To convert Volts to Energy, multiply ESA voltage in Volts by
    energy conversion factor to get electron energy in eV.

    Returns
    -------
    lookup_table : pandas.DataFrame
        Lookup table with energy column added.
    """
    # The lookup table gives voltage applied to analyzers.
    lookup_table = read_lookup_table()

    # Convert voltage to electron energy in eV by apply conversion factor.
    lookup_table["energy"] = lookup_table["esa_v"].values * ENERGY_CONVERSION_FACTOR
    return lookup_table


def calculate_phase_space_density(l1b_dataset: xr.Dataset) -> xr.Dataset:
    """
    Convert counts to phase space density.

    Calculate phase space density is represented by this symbol, fv.
    Its unit is s^3/ (cm^6 * ster).

    The formula to calculate phase space density,

    Where:
        C / tau = corrected count rate which in the input L1B science data.
        G = geometric factor, in (cm^2 * ster). 7 CEMs geometric factor value.
        eV = eV in electron-volts, calculated by get_particle_energy().
        E = Energy in Joules. eV * 1.60219e-19(J/eV).
        m = mass of electron (9.10938356e-31 kg).
        s = second.
        v = sqrt(2 * E / m). Electron speed, computed from energy. In cm/s.
        J = kg * m^2 / s^2. J for joules.
        fv = phase space density.

    v   = sqrt(2 * E / m)
        = sqrt(2 * eV * 1.60219e-19(J/eV) / 9.10938e-31 kg)
        = sqrt(2 * 1.60219 * 10e−19 m^2/s^2 * eV / 9.10938e-31)
        = sqrt(2 * 1.60219 * 10e−19 * 10e4 cm^2/s^2 * eV / 9.10938e-31)
        = sqrt(3.20438 * 10e-15 * eV / 9.10938e-31) cm/s
        = sqrt((3.20438 * 10e-15 / 9.10938e-31) * eV) cm/s

    fv  = 2 * (C/tau) / (G * v^4)
        = 2 * (C/tau) / (G * (sqrt( (3.20438 * 10e-15 / 9.10938e-31) * eV ))^4)
        = 2 * (C/tau) / (G * (sqrt(3.5176e16)^4 * eV^2)
        = 2 * (C/tau) / (G * 1.237e31 * eV^2)
        Ruth Skoug also got the same result, 1.237e31.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The L1B dataset to process.

    Returns
    -------
    phase_space_density_dataset : xarray.Dataset
        Phase space density. We need to call this phase space density because
        there will be density in L3 processing.
    """
    # Get esa_table_num for each full sweep.
    esa_table_nums = l1b_dataset["esa_table_num"].values[:, 0]
    # Get energy values from lookup table.
    particle_energy = get_particle_energy()
    # Get 720 (24 energy steps x 30 angle) particle energy for each full
    # sweep data.
    particle_energy_data = np.array(
        [
            particle_energy[particle_energy["table_index"] == val]["energy"].tolist()
            for val in esa_table_nums
        ]
    )
    particle_energy_data = particle_energy_data.reshape(-1, 24, 30)

    # Calculate phase space density using formula:
    #   2 * (C/tau) / (G * 1.237e31 * eV^2)
    # See doc string for more details.
    density = (2 * l1b_dataset["science_data"]) / (
        GEOMETRIC_FACTORS[np.newaxis, np.newaxis, np.newaxis, :]
        * VELOCITY_CONVERSION_FACTOR
        * particle_energy_data[:, :, :, np.newaxis] ** 2
    )

    # Return density as xr.dataset with phase space density and
    # energy in eV value that flux calculation can use.
    phase_space_density_dataset = xr.Dataset(
        {
            "phase_space_density": (
                ["epoch", "esa_step", "spin_sector", "cem_id"],
                density.data,
            ),
            "energy_in_eV": (
                ["epoch", "esa_step", "spin_sector"],
                particle_energy_data,
            ),
        },
        coords=l1b_dataset.coords,
    )

    return phase_space_density_dataset


def calculate_flux(l1b_dataset: xr.Dataset) -> npt.NDArray:
    """
    Calculate flux.

    Flux is represented by this symbol, j. Its unit is
    1 / (2 * eV * cm^2 * s * ster).

    The formula to calculate flux,

    Where:
        fv = the phase space density of solar wind electrons
            given by calculate_phase_space_density() result.
        eV = Energy in electron-volts, calculated by get_particle_energy().
        E  = Energy in Joules. eV * 1.60219e-19(J/eV).
        v  = sqrt( (3.20438 * 10e-15 / 9.10938e-31) * eV ) cm/s. See
            calculate_phase_space_density() for this calculation.
        j  = flux factor.

    Flux units workout:
    j   = (fv * v^4) / (2 * eV)
        = ((s^3 / (cm^6 * ster)) * (cm^4/s^4)) / (2 * eV)
        = ((s^3 * cm^4) / (cm^6 * s^4 * ster)) / (2 * eV)
        = (1 / (cm^2 * s * ster)) / (2 * eV)
        = 1 / (2 * eV * cm^2 * s * ster)

    Flux conversion factor workout:
    j   = (fv * v^4) / (2 * eV)
        = ( fv * (sqrt( (3.20438 * 10e-15 / 9.10938e-31) * eV )^4) ) / (2 * eV)
        = ( fv * ((3.20438 * 10e-15 / 9.10938e-31) * eV)^1/2) ^ 4 ) / (2 * eV)
        = ( fv * (3.20438 * 10e-15 / 9.10938e-31)^2 * eV^2) ) / (2 * eV)
        = ( fv * 1.237e31 * eV^2) ) / (2 * eV)
        = ( fv * 1.237e31 * eV ) / 2
        = (fv * 6.187e30 * eV)
        Ruth Skoug confirmed this factor, 6.187e30.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The L1B dataset to process.

    Returns
    -------
    flux : numpy.ndarray
        Flux values.
    """
    phase_space_density_ds = calculate_phase_space_density(l1b_dataset)
    flux = (
        FLUX_CONVERSION_FACTOR
        * phase_space_density_ds["energy_in_eV"].data[:, :, :, np.newaxis]
        * phase_space_density_ds["phase_space_density"].data
    )
    return flux


def swe_l2(l1b_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Will process data to L2.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The L1B dataset to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    data : xarray.Dataset
        Processed data to L2.
    """
    cdf_attributes = ImapCdfAttributes()
    cdf_attributes.add_instrument_global_attrs("swe")
    cdf_attributes.add_instrument_variable_attrs("swe", "l2")
    cdf_attributes.add_global_attribute("Data_version", data_version)

    # Energy values in eV.
    energy_xr = xr.DataArray(
        np.array(list(ESA_VOLTAGE_ROW_INDEX_DICT.keys())) * ENERGY_CONVERSION_FACTOR,
        name="energy",
        dims=["energy"],
        attrs=cdf_attributes.get_variable_attributes("energy"),
    )

    # Angle of each CEM detectors.
    inst_el_xr = xr.DataArray(
        CEM_DETECTORS_ANGLE,
        name="inst_el",
        dims=["inst_el"],
        attrs=cdf_attributes.get_variable_attributes("inst_el"),
    )

    # Spin Angle bins storing bin center values.
    inst_az_xr = xr.DataArray(
        np.arange(6, 360, 12),
        name="inst_az",
        dims=["inst_az"],
        attrs=cdf_attributes.get_variable_attributes("inst_az"),
    )

    dataset = xr.Dataset(
        coords={
            "epoch": l1b_dataset["epoch"],
            "esa_step": l1b_dataset["esa_step"],
            "energy": energy_xr,
            "spin_sector": l1b_dataset["spin_sector"],
            "inst_az": inst_az_xr,
            "cem_id": l1b_dataset["cem_id"],
            "inst_el": inst_el_xr,
            "esa_step_label": l1b_dataset["esa_step_label"],
            "spin_sector_label": l1b_dataset["spin_sector_label"],
            "cem_id_label": l1b_dataset["cem_id_label"],
        },
        attrs=cdf_attributes.get_global_attributes("imap_swe_l2_sci"),
    )

    ############################################################
    # Calculate phase space density and flux. Store data in shape
    # (epoch, esa_step, spin_sector, cem_id). This is for L3 purposes.
    ############################################################
    phase_space_density = calculate_phase_space_density(l1b_dataset)[
        "phase_space_density"
    ]
    dataset["phase_space_density_spin_sector"] = xr.DataArray(
        phase_space_density,
        name="phase_space_density_spin_sector",
        dims=["epoch", "esa_step", "spin_sector", "cem_id"],
        attrs=cdf_attributes.get_variable_attributes("phase_space_density_spin_sector"),
    )

    flux = calculate_flux(l1b_dataset)
    dataset["flux_spin_sector"] = xr.DataArray(
        flux,
        name="flux_spin_sector",
        dims=["epoch", "esa_step", "spin_sector", "cem_id"],
        attrs=cdf_attributes.get_variable_attributes("flux_spin_sector"),
    )

    # Carry over acquisition times for L3 purposes.
    dataset["acquisition_time"] = l1b_dataset["acquisition_time"]

    # Calculate spin phase using SWE acquisition_time calculated in l1b.
    # L1B dataset stores it by (epoch, esa_step, spin_sector).
    # To calculate center time of data acquisition time, we will add
    #   acquisition_time + (acq_duration / 1000) / 2
    # acq_duration is in milliseconds and is stored in L1B dataset by
    # (epoch, cycle). acq_duration should be same for all esa_steps in
    # a full sweep. We will take the first acq_duration value for each
    # full sweep. This center time calculation is done to get the center
    # angle of the data.
    acq_duration = l1b_dataset["acq_duration"].data[:, 0] / 2000
    data_acq_time = (
        l1b_dataset["acquisition_time"].data + acq_duration[:, np.newaxis, np.newaxis]
    )

    # calculate spin phase
    inst_spin_phase = get_instrument_spin_phase(
        query_met_times=data_acq_time.flatten(),
        instrument=SpiceFrame.IMAP_SWE,
    )

    inst_spin_angle = get_spin_angle(inst_spin_phase, degrees=True).reshape(-1, 24, 30)
    # Spin angle bins range is little different from spin angle bins.
    # Spin angle bins are centered like:
    #   [ 6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174,
    #   186, 198, 210, 222, 234, 246, 258, 270, 282, 294, 306, 318, 330,
    #   342, 354]
    # Where does an input angle goes into which bins is determined by the
    # following logic:
    #   phi_begin <= center - 6
    #   phi_center = 6
    #   phi_end < center + 6
    # For example, if input_angle is 8.4, we would put in spin angle bin 6.
    # To make binning easier, we will use following bin ranges:
    #   [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168,
    #   180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324,
    #   336, 348]
    # SWE want to use right side of np.searchsorted, a[i-1] <= v < a[i].
    # Index of spin angle bins and spin angle range should match.
    # For example,
    #   np.searchsorted(x, [6], side="right") -> [1]. Bin center test.
    #   np.searchsorted(x, [8.4], side="right"] -> [1]. Bin center edge test.
    #   np.searchsorted(x, [12], side="right") -> [2]. Bin end test.
    #   np.searchsorted(x, [0], side="right") -> [1]. Bin start test.
    # [i-1] gives the correct bin index for all the above tests.
    spin_angle_bins_range = np.arange(0, 360, 12)
    spin_angle_bins_indices = np.searchsorted(
        spin_angle_bins_range, inst_spin_angle, side="right"
    )
    spin_angle_bins_indices = spin_angle_bins_indices - 1

    # Now, take flux data and put it in its spin angle bins using the indices.
    # TODO: do this

    # print(pd.DataFrame(inst_spin_angle[0], columns=np.arange(30)
    # ).to_csv("spin_angle.csv"))
    # print(pd.DataFrame(spin_angle_bins_indices[0], columns=np.arange(30)
    # ).to_csv("spin_angle_bins.csv"))
    return dataset
