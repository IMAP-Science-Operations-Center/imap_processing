import numpy as np

# from imap_processing.ultra.l1b.ultra_l1b_extended import the energy equation here
from imap_processing.spice.geometry import get_spacecraft_spin_phase, get_spin_data
from imap_processing.quality_flags import ImapUltraFlags


def get_spin(met) -> list:

    # TODO: error handling for out of range spins
    spin_df = get_spin_data()

    last_spin_indices = (
        np.searchsorted(spin_df["spin_start_time"], met, side="right") - 1
    )
    spin = spin_df["spin_number"].values[last_spin_indices]
    spin_start_time = spin_df["spin_start_time"].values[last_spin_indices]
    spin_duration = spin_df["spin_period_sec"].values[last_spin_indices]

    return spin# TODO, spin_start_time, spin_duration


def get_energy_histogram(spin, energy):
    """
    Compute a 3D histogram of the particle data.

    Parameters
    ----------
    v : tuple[np.ndarray, np.ndarray, np.ndarray]
        The x,y,z-components of the velocity vector.
    energy : np.ndarray
        The particle energy.
    az_bin_edges : np.ndarray
        Array of azimuth bin boundary values.
    el_bin_edges : np.ndarray
        Array of elevation bin boundary values.
    energy_bin_edges : np.ndarray
        Array of energy bin edges.

    Returns
    -------
    hist : np.ndarray
        A 3D histogram array.
    """

    spin_edges = np.unique(spin)
    spin_edges = np.append(spin_edges, spin_edges[-1] + 1)
    energy_bin_edges = [-1e5, 0, 10, 20, 1e5]

    # 2D binning.
    hist, _ = np.histogramdd(sample=(energy, spin),
                          bins=[energy_bin_edges, spin_edges])

    return hist, spin_edges, energy_bin_edges


def flag_spin(l1b_de_dataset):

    quality_flags_data = np.zeros(len(l1b_de_dataset["de_event_met"]), np.uint16)
    quality_flags_data[l1b_de_dataset["energy"] < 0] |= ImapUltraFlags.NEG.value

    spin = get_spin(l1b_de_dataset["de_event_met"])
    hist, spin_edges, energy_bin_edges = get_energy_histogram(spin, l1b_de_dataset["energy"])

    energy_bin_idx = np.digitize(l1b_de_dataset["energy"], bins=energy_bin_edges) - 1
    spin_bin_idx = np.digitize(spin, bins=spin_edges) - 1

    for spin_idx, energy_idx in np.ndindex(hist.shape):
        if hist[energy_idx, spin_idx] > 100:  # Threshold for quality flag
            # Find all data points corresponding to this bin
            mask = (energy_bin_idx == energy_idx) & (spin_bin_idx == spin_idx)
            quality_flags_data[mask] |= ImapUltraFlags.BADSPIN.value
