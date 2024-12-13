"""Calculate Annotated Direct Events."""

import numpy as np
import xarray as xr

from imap_processing.cdf.utils import parse_filename_like
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.ultra.l1b.ultra_l1b_annotated import (
    get_annotated_particle_velocity,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    StopType,
    determine_species_pulse_height,
    determine_species_ssd,
    get_coincidence_positions,
    get_ctof,
    get_energy_pulse_height,
    get_energy_ssd,
    get_front_x_position,
    get_front_y_position,
    get_path_length,
    get_ph_tof_and_back_positions,
    get_ssd_back_position_and_tof_offset,
    get_ssd_tof,
    get_unit_vector,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_de(de_dataset: xr.Dataset, name: str) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Direct Event Data.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        L1a dataset containing direct event data.
    name : str
        Name of the l1a dataset.

    Returns
    -------
    dataset : xarray.Dataset
        L1b de dataset.
    """
    de_dict = {}
    sensor = parse_filename_like(name)["sensor"][0:2]

    # Instantiate arrays
    yf = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float32)
    xb = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float32)
    yb = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float32)
    xc = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float32)
    d = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float64)
    r = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float32)
    tof = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float32)
    etof = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float32)
    ctof = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float32)
    energy = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float32)
    # TODO: uint8 fills with zeros instead of nans.
    #  Confirm with Ultra team what fill values and dtype we want.
    species_bin = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.uint8)
    t2 = np.full(len(de_dataset["epoch"]), np.nan, dtype=np.float32)

    # Drop events with invalid start type.
    de_dataset = de_dataset.where(
        de_dataset["START_TYPE"] != np.iinfo(np.int64).min, drop=True
    )
    # Define epoch.
    de_dict["epoch"] = de_dataset["epoch"]

    xf = get_front_x_position(
        de_dataset["START_TYPE"].data,
        de_dataset["START_POS_TDC"].data,
    )

    # Pulse height
    ph_indices = np.nonzero(
        np.isin(de_dataset["STOP_TYPE"], [StopType.Top.value, StopType.Bottom.value])
    )[0]
    tof[ph_indices], t2[ph_indices], xb[ph_indices], yb[ph_indices] = (
        get_ph_tof_and_back_positions(de_dataset, xf, f"ultra{sensor}")
    )
    d[ph_indices], yf[ph_indices] = get_front_y_position(
        de_dataset["START_TYPE"].data[ph_indices], yb[ph_indices]
    )
    energy[ph_indices] = get_energy_pulse_height(
        de_dataset["STOP_TYPE"].data[ph_indices],
        de_dataset["ENERGY_PH"].data[ph_indices],
        xb[ph_indices],
        yb[ph_indices],
    )
    r[ph_indices] = get_path_length(
        (xf[ph_indices], yf[ph_indices]),
        (xb[ph_indices], yb[ph_indices]),
        d[ph_indices],
    )
    species_bin[ph_indices] = determine_species_pulse_height(
        energy[ph_indices], tof[ph_indices], r[ph_indices]
    )
    etof[ph_indices], xc[ph_indices] = get_coincidence_positions(
        de_dataset.isel(epoch=ph_indices), t2[ph_indices], f"ultra{sensor}"
    )
    ctof[ph_indices] = get_ctof(tof[ph_indices], r[ph_indices], "PH")

    # SSD
    ssd_indices = np.nonzero(np.isin(de_dataset["STOP_TYPE"], StopType.SSD.value))[0]
    tof[ssd_indices] = get_ssd_tof(de_dataset, xf)
    yb[ssd_indices], _, ssd_number = get_ssd_back_position_and_tof_offset(de_dataset)
    xc[ssd_indices] = np.zeros(len(ssd_indices))
    xb[ssd_indices] = np.zeros(len(ssd_indices))
    etof[ssd_indices] = np.zeros(len(ssd_indices))
    d[ssd_indices], yf[ssd_indices] = get_front_y_position(
        de_dataset["START_TYPE"].data[ssd_indices], yb[ssd_indices]
    )
    energy[ssd_indices] = get_energy_ssd(de_dataset, ssd_number)
    r[ssd_indices] = get_path_length(
        (xf[ssd_indices], yf[ssd_indices]),
        (xb[ssd_indices], yb[ssd_indices]),
        d[ssd_indices],
    )
    species_bin[ssd_indices] = determine_species_ssd(
        energy[ssd_indices],
        tof[ssd_indices],
        r[ssd_indices],
    )
    ctof[ssd_indices] = get_ctof(tof[ssd_indices], r[ssd_indices], "SSD")

    # Combine ph_yb and ssd_yb along with their indices
    de_dict["x_front"] = xf.astype(np.float32)
    de_dict["y_front"] = yf
    de_dict["x_back"] = xb
    de_dict["y_back"] = yb
    de_dict["x_coin"] = xc
    de_dict["tof_start_stop"] = tof
    de_dict["tof_stop_coin"] = etof
    de_dict["tof_corrected"] = ctof
    de_dict["front_back_distance"] = d
    de_dict["path_length"] = r

    keys = [
        "coincidence_type",
        "start_type",
        "event_type",
        "de_event_met",
    ]
    dataset_keys = ["COIN_TYPE", "START_TYPE", "STOP_TYPE", "SHCOARSE"]

    de_dict.update(
        {key: de_dataset[dataset_key] for key, dataset_key in zip(keys, dataset_keys)}
    )

    vx_ultra, vy_ultra, vz_ultra = get_unit_vector(
        (de_dict["x_front"], de_dict["y_front"]),
        (de_dict["x_back"], de_dict["y_back"]),
        de_dict["front_back_distance"],
        de_dict["tof_start_stop"],
    )

    de_dict["vx_ultra"] = vx_ultra.astype(np.float32)
    de_dict["vy_ultra"] = vy_ultra.astype(np.float32)
    de_dict["vz_ultra"] = vz_ultra.astype(np.float32)
    de_dict["energy"] = energy
    de_dict["species"] = species_bin

    # Annotated Events.
    position = np.stack(
        (de_dict["vx_ultra"], de_dict["vy_ultra"], de_dict["vz_ultra"]), axis=-1
    )

    ultra_frame = getattr(SpiceFrame, f"IMAP_ULTRA_{sensor}")
    sc_velocity, sc_dps_velocity, helio_velocity = get_annotated_particle_velocity(
        de_dataset.data_vars["EVENTTIMES"],
        position,
        ultra_frame,
        SpiceFrame.IMAP_DPS,
        SpiceFrame.IMAP_SPACECRAFT,
    )

    de_dict["vx_sc"], de_dict["vy_sc"], de_dict["vz_sc"] = (
        sc_velocity[:, 0],
        sc_velocity[:, 1],
        sc_velocity[:, 2],
    )
    de_dict["vx_dps_sc"], de_dict["vy_dps_sc"], de_dict["vz_dps_sc"] = (
        sc_dps_velocity[:, 0],
        sc_dps_velocity[:, 1],
        sc_dps_velocity[:, 2],
    )
    de_dict["vx_dps_helio"], de_dict["vy_dps_helio"], de_dict["vz_dps_helio"] = (
        helio_velocity[:, 0],
        helio_velocity[:, 1],
        helio_velocity[:, 2],
    )

    # TODO: TBD.
    de_dict["event_efficiency"] = np.full(
        len(de_dataset["epoch"]), np.nan, dtype=np.float32
    )

    dataset = create_dataset(de_dict, name, "l1b")

    return dataset
