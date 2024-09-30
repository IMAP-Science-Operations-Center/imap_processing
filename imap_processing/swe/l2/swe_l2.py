"""SWE L2 processing module."""

import numpy.typing as npt

from imap_processing.swe.l1b.swe_l1b_science import read_lookup_table

# TODO: add this to instrument status summary
ENERGY_CONVERSION_FACTOR = 4.75


def calculate_particle_energy(esa_table_num: int) -> npt.NDArray:
    """
    Calculate particle energy.

    To convert Volts to Energy, multiply ESA voltage in Volts by 4.75 to get
    electron energy in eV.

    Parameters
    ----------
    esa_table_num : int
        ESA table number.

    Returns
    -------
    energy : np.ndarray
        720 step energy values.
    """
    # The lookup table gives voltage applied to analyzers.
    esa_table = read_lookup_table(table_index_value=esa_table_num)

    # Convert voltage to electron energy in eV by apply conversion factor.
    energy = esa_table["esa_v"] * ENERGY_CONVERSION_FACTOR
    return energy
