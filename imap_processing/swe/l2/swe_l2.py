"""
SWE L2 processing module.

This module contains functions to process L1B data to L2 data products.
"""

import numpy as np
import numpy.typing as npt
import xarray as xr

from imap_processing.spice.geometry import get_spacecraft_spin_phase
from imap_processing.swe.utils.swe_utils import read_lookup_table

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
            "phase_space_density": (["epoch", "energy", "angle", "cem"], density.data),
            "energy_in_eV": (["epoch", "energy", "angle"], particle_energy_data),
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
    flux = calculate_flux(l1b_dataset)

    # Calculate spin phase using SWE sci_step_acq_time_sec calculated in l1b.
    # L1B dataset stores it by (epoch, energy, angle, cem).
    data_acq_time = l1b_dataset["sci_step_acq_time_sec"].data.flatten()

    # calculate spin phase
    spin_phase = get_spacecraft_spin_phase(
        query_met_times=data_acq_time,
    ).reshape(-1, 24, 30)
    # TODO: organize flux data by energy and spin_phase.
    # My understanding from conversation with Ruth is that this is the hardest part
    # and last part of the L2 processing.

    # TODO: Correct return value. This is just a placeholder.
    return xr.Dataset(
        {
            "flux": (["epoch", "energy", "spin_phase", "cem"], flux),
            "spin_phase": (["epoch", "energy", "spin_phase"], spin_phase),
        },
        coords=l1b_dataset.coords,
    )
