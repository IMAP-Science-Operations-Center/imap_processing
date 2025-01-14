import numpy as np

# from imap_processing.ultra.l1b.ultra_l1b_extended import the energy equation here


def get_spin(met, energy) -> list:
    for time in met:
        # Get the nearest spin start and duration prior to the event
        spin_start = aux_spin_starts[aux_spin_starts <= time][-1]
        duration = np.array(decom_aux["DURATION"])[aux_spin_starts <= time][-1]

        # Find the events
        event_indices = np.where(np.array(decom_events["SHCOARSE"]) == time)

        for event_index in event_indices[0]:
            phase_angle = decom_events["PHASE_ANGLE"][event_index]

            durations.append(duration)
            spin_starts.append(spin_start)

            # If there were no events, the time is set to 'SHCOARSE'
            if decom_events["COUNT"][event_index] == 0:
                event_times.append(decom_events["SHCOARSE"][event_index])
            else:
                event_times.append(spin_start + (duration / 1000) * (phase_angle / 720))

    decom_events["DURATION"] = durations
    decom_events["TIMESPINSTART"] = spin_starts
    decom_events["EVENTTIMES"] = event_times

    return (
        energy,
        counts,
    )


def get_energy_histogram(
    energy: np.ndarray,
) -> NDArray:
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
    energy_bin_edges = [0, 10, 20]
    spin_times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 2D binning.
    hist, _ = np.histogramdd(sample=(energy), bins=[energy_bin_edges])

    return hist
