"""Creates xarray based on structure of queried DynamoDB."""

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from cdflib.epochs import CDFepoch

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice.constants import (
    HI_IALIRT_ELEVATION_ANGLE,
)
from imap_processing.ialirt.utils.constants import (
    IALIRT_DIMS,
    IALIRT_DTYPES,
    codice_hi_energy_center,
    codice_hi_energy_minus,
    codice_hi_energy_plus,
    hit_restricted_fields,
    swe_energy,
)


def create_xarray_from_records(records: list[dict]) -> xr.Dataset:  # noqa: PLR0912
    """
    Create dataset from a list of records.

    Parameters
    ----------
    records : list of dict
       Output of querying DynamoDB.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset in standard format.
    """
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_global_attrs("ialirt")
    cdf_manager.add_instrument_variable_attrs("ialirt", "l1")

    one_epoch = {"codice_lo", "hit", "swapi", "swe", "spacecraft", "mag"}
    multi_epoch = {"codice_hi"}

    epochs: dict[str, list[int]] = {inst: [] for inst in (one_epoch | multi_epoch)}
    by_inst: dict[str, list[dict]] = defaultdict(list)

    # Get the start and end ttj2000ns.
    date = records[0]["time_utc"]  # e.g. "2025-06-20T08:00:00Z"

    # Parse as UTC
    dt = datetime.fromisoformat(date)

    # Start and end of that UTC day
    start_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = start_dt + timedelta(days=1)

    start_ttj2000 = int(
        CDFepoch.compute_tt2000(
            [
                start_dt.year,
                start_dt.month,
                start_dt.day,
                start_dt.hour,
                start_dt.minute,
                start_dt.second,
                0,
                0,
                0,  # ms, us, ns
            ]
        )
    )
    end_ttj2000 = int(
        CDFepoch.compute_tt2000(
            [
                end_dt.year,
                end_dt.month,
                end_dt.day,
                end_dt.hour,
                end_dt.minute,
                end_dt.second,
                0,
                0,
                0,  # ms, us, ns
            ]
        )
    )

    for record in records:
        inst = record.get("instrument")
        by_inst[record["instrument"]].append(record)
        if inst == "spacecraft":
            epochs[inst].append(record["ttj2000ns"])
        elif inst in one_epoch:
            epochs[inst].append(record[f"{inst}_epoch"])
        elif inst in multi_epoch:
            epochs[inst].extend(record[f"{inst}_epoch"])

    epoch_arrays = {}

    for inst, arr in epochs.items():
        if inst == "spacecraft":
            coord = "ephemeris_epoch"
            attr = cdf_manager.get_variable_attributes(
                "ephemeris_epoch", check_schema=False
            )
        else:
            coord = f"{inst}_epoch"
            attr = cdf_manager.get_variable_attributes(
                f"{inst}_epoch", check_schema=False
            )
        epoch_arrays[coord] = xr.DataArray(
            data=np.array(arr, dtype=np.int64), name=coord, dims=[coord], attrs=attr
        )

    sc_gsm_position_component = xr.DataArray(
        ["sc X (GSM)", "sc Y (GSM)", "sc Z (GSM)"],
        name="sc_GSM_position_labels",
        dims=["sc_GSM_position_labels"],
        attrs=cdf_manager.get_variable_attributes(
            "sc_GSM_position_labels", check_schema=False
        ),
    )
    sc_gsm_velocity_component = xr.DataArray(
        ["sc Vx (GSM)", "sc Vy (GSM)", "sc Vz (GSM)"],
        name="sc_GSM_velocity_labels",
        dims=["sc_GSM_velocity_labels"],
        attrs=cdf_manager.get_variable_attributes(
            "sc_GSM_velocity_labels", check_schema=False
        ),
    )

    sc_gse_position_component = xr.DataArray(
        ["sc X (GSE)", "sc Y (GSE)", "sc Z (GSE)"],
        name="sc_GSE_position_labels",
        dims=["sc_GSE_position_labels"],
        attrs=cdf_manager.get_variable_attributes(
            "sc_GSE_position_labels", check_schema=False
        ),
    )
    sc_gse_velocity_component = xr.DataArray(
        ["sc Vx (GSE)", "sc Vy (GSE)", "sc Vz (GSE)"],
        name="sc_GSE_velocity_labels",
        dims=["sc_GSE_velocity_labels"],
        attrs=cdf_manager.get_variable_attributes(
            "sc_GSE_velocity_labels", check_schema=False
        ),
    )

    gsm_component = xr.DataArray(
        ["Bx (GSM)", "By (GSM)", "Bz (GSM)"],
        name="B_GSM_labels",
        dims=["B_GSM_labels"],
        attrs=cdf_manager.get_variable_attributes("B_GSM_labels", check_schema=False),
    )

    gse_component = xr.DataArray(
        ["Bx (GSE)", "By (GSE)", "Bz (GSE)"],
        name="B_GSE_labels",
        dims=["B_GSE_labels"],
        attrs=cdf_manager.get_variable_attributes("B_GSE_labels", check_schema=False),
    )

    rtn_component = xr.DataArray(
        ["B radial (RTN)", "B tangential (RTN)", "B normal (RTN)"],
        name="B_RTN_labels",
        dims=["B_RTN_labels"],
        attrs=cdf_manager.get_variable_attributes("B_RTN_labels", check_schema=False),
    )

    swe_electron_energy = xr.DataArray(
        data=np.float32(swe_energy),
        name="swe_electron_energy",
        dims=["swe_electron_energy"],
        attrs=cdf_manager.get_variable_attributes(
            "swe_electron_energy", check_schema=False
        ),
    )

    codice_hi_energy_centers = xr.DataArray(
        data=np.array(codice_hi_energy_center, dtype=np.float32),
        name="codice_hi_energy_center",
        dims=["codice_hi_energy_center"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_energy_center", check_schema=False
        ),
    )

    codice_energy_minus = xr.DataArray(
        data=np.array(codice_hi_energy_minus, dtype=np.float32),
        name="codice_hi_energy_minus",
        dims=["codice_hi_energy_center"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_energy_minus", check_schema=False
        ),
    )

    codice_energy_plus = xr.DataArray(
        data=np.array(codice_hi_energy_plus, dtype=np.float32),
        name="codice_hi_energy_plus",
        dims=["codice_hi_energy_center"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_energy_plus", check_schema=False
        ),
    )

    polar = xr.DataArray(
        HI_IALIRT_ELEVATION_ANGLE,
        name="codice_hi_polar",
        dims=["codice_hi_polar"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_polar", check_schema=False
        ),
    )

    polar_labels = xr.DataArray(
        [f"{float(v):.1f}deg" for v in polar.values],
        name="codice_hi_polar_labels",
        dims=["codice_hi_polar"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_polar_labels", check_schema=False
        ),
    )

    spin_sector = xr.DataArray(
        data=np.arange(4, dtype=np.uint8),
        name="codice_hi_spin_sector",
        dims=["codice_hi_spin_sector"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_spin_sector", check_schema=False
        ),
    )

    spin_sector_labels = xr.DataArray(
        [
            "0",
            "1",
            "2",
            "3",
        ],
        name="codice_hi_spin_sector_labels",
        dims=["codice_hi_spin_sector"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_spin_sector_labels", check_schema=False
        ),
    )

    coords = {
        "codice_hi_epoch": epoch_arrays["codice_hi_epoch"],
        "codice_lo_epoch": epoch_arrays["codice_lo_epoch"],
        "hit_epoch": epoch_arrays["hit_epoch"],
        "mag_epoch": epoch_arrays["mag_epoch"],
        "swapi_epoch": epoch_arrays["swapi_epoch"],
        "swe_epoch": epoch_arrays["swe_epoch"],
        "ephemeris_epoch": epoch_arrays["ephemeris_epoch"],
        "B_GSM_labels": gsm_component,
        "B_GSE_labels": gse_component,
        "B_RTN_labels": rtn_component,
        "sc_GSM_position_labels": sc_gsm_position_component,
        "sc_GSM_velocity_labels": sc_gsm_velocity_component,
        "sc_GSE_position_labels": sc_gse_position_component,
        "sc_GSE_velocity_labels": sc_gse_velocity_component,
        "codice_hi_energy_center": codice_hi_energy_centers,
        "codice_hi_energy_minus": codice_energy_minus,
        "codice_hi_energy_plus": codice_energy_plus,
        "codice_hi_polar": polar,
        "codice_hi_polar_labels": polar_labels,
        "codice_hi_spin_sector": spin_sector,
        "codice_hi_spin_sector_labels": spin_sector_labels,
        "swe_electron_energy": swe_electron_energy,
    }
    dataset = xr.Dataset(
        coords=coords,
        attrs=cdf_manager.get_global_attributes("imap_ialirt_l1_realtime"),
    )

    # Create variables with fill values.
    for key in IALIRT_DIMS:
        dims = IALIRT_DIMS[key]
        attrs = cdf_manager.get_variable_attributes(key, check_schema=False)
        fill = attrs["FILLVAL"]
        dtype = IALIRT_DTYPES[key]

        shape = [dataset.dims[d] for d in dims]

        data = np.full(shape, fill, dtype=dtype)
        dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)

    for i, record in enumerate(by_inst.get("mag", [])):
        for key in IALIRT_DIMS.keys():
            if key in ["mag_B_GSE", "mag_B_GSM", "mag_B_RTN"]:
                dataset[key].data[i, :] = record[key]

            if key in [
                "mag_B_magnitude",
                "mag_theta_B_GSE",
                "mag_phi_B_GSE",
                "mag_theta_B_GSM",
                "mag_phi_B_GSM",
            ]:
                dataset[key].data[i] = np.float32(record[key])

    for i, record in enumerate(by_inst.get("codice_hi", [])):
        # 4 codice-hi epochs per record
        t0 = 4 * i
        t1 = t0 + 4
        hi = np.asarray(record["codice_hi_h"], dtype=np.float32)
        dataset["codice_hi_h"].data[t0:t1, :, :, :] = hi

    for i, record in enumerate(by_inst.get("codice_lo", [])):
        for key in IALIRT_DIMS.keys():
            val = record.get(key)
            if (
                key.startswith("codice_lo_")
                and key != "codice_lo_epoch"
                and val is not None
            ):
                dataset[key].data[i] = np.float32(val)

    for i, record in enumerate(by_inst.get("hit", [])):
        for key in IALIRT_DIMS.keys():
            val = record.get(key)
            if (
                key.startswith("hit_")
                and key != "hit_epoch"
                and val is not None
                and key not in hit_restricted_fields
            ):
                dataset[key].data[i] = np.float32(val)

    for i, record in enumerate(by_inst.get("swapi", [])):
        for key in IALIRT_DIMS.keys():
            val = record.get(key)
            if key.startswith("swapi_") and key != "swapi_epoch" and val is not None:
                dataset[key].data[i] = np.float32(val)

    for i, record in enumerate(by_inst.get("swe", [])):
        dataset["swe_normalized_counts"].data[i, :] = np.asarray(
            record["swe_normalized_counts"], dtype=np.int64
        )
        dataset["swe_counterstreaming_electrons"].data[i] = np.uint8(
            record["swe_counterstreaming_electrons"]
        )

    for i, record in enumerate(by_inst.get("spacecraft", [])):
        for key in IALIRT_DIMS.keys():
            if key.startswith("sc_"):
                dataset[key].data[i, :] = np.asarray(record[key], dtype=np.float32)

    # Trim data that does not fit within the UTC day.
    for inst in epochs.keys():
        if inst == "spacecraft":
            dim = "ephemeris_epoch"
        else:
            dim = f"{inst}_epoch"

        if dim in dataset.coords:
            dataset = dataset.sel(
                {dim: (dataset[dim] >= start_ttj2000) & (dataset[dim] < end_ttj2000)}
            )

    return dataset
