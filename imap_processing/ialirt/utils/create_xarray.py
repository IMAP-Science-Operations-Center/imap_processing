"""Creates xarray based on structure of queried DynamoDB."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ialirt.utils.constants import IALIRT_KEYS


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

    n = len(records)
    attrs = cdf_manager.get_variable_attributes("default_int64_attrs")
    fillval = attrs.get("FILLVAL")
    ttj2000ns_values = np.full(n, fillval, dtype=np.int64)

    # Collect all keys that start with the instrument prefixes.
    for i, record in enumerate(records):
        ttj2000ns_values[i] = record["ttj2000ns"]

    epoch = xr.DataArray(
        data=ttj2000ns_values,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("epoch", check_schema=False),
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

    swe_electron_energy_labels = xr.DataArray(
        [
            "100.4 eV",
            "140 eV",
            "194 eV",
            "270 eV",
            "376 eV",
            "523 eV",
            "727 eV",
            "1011 eV",
        ],
        name="swe_electron_energy_labels",
        dims=["swe_electron_energy_labels"],
        attrs=cdf_manager.get_variable_attributes(
            "swe_electron_energy_labels", check_schema=False
        ),
    )

    energy_range = xr.DataArray(
        data=np.arange(15, dtype=np.uint8),
        name="codice_hi_h_energy_range",
        dims=["codice_hi_h_energy_range"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_h_energy_range", check_schema=False
        ),
    )

    azimuth = xr.DataArray(
        data=np.arange(4, dtype=np.uint8),
        name="codice_hi_h_azimuth",
        dims=["codice_hi_h_azimuth"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_h_azimuth", check_schema=False
        ),
    )

    spin_angle = xr.DataArray(
        data=np.arange(4, dtype=np.uint8),
        name="codice_hi_h_spin_angle",
        dims=["codice_hi_h_spin_angle"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_h_spin_angle", check_schema=False
        ),
    )

    spin_sector = xr.DataArray(
        data=np.arange(4, dtype=np.uint8),
        name="codice_hi_h_spin_sector",
        dims=["codice_hi_h_spin_sector"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_h_spin_sector", check_schema=False
        ),
    )

    coords = {
        "epoch": epoch,
        "B_GSM_labels": gsm_component,
        "B_GSE_labels": gse_component,
        "B_RTN_labels": rtn_component,
        "swe_electron_energy_labels": swe_electron_energy_labels,
        "codice_hi_h_spin_angle": spin_angle,
        "codice_hi_h_energy_range": energy_range,
        "codice_hi_h_spin_sector": spin_sector,
        "codice_hi_h_azimuth": azimuth,
    }
    dataset = xr.Dataset(
        coords=coords,
        attrs=cdf_manager.get_global_attributes("imap_ialirt_l1_realtime"),
    )

    # Create empty dataset for each key.
    for key in IALIRT_KEYS:
        attrs = cdf_manager.get_variable_attributes(key, check_schema=False)
        fillval = attrs.get("FILLVAL")
        if key in ["mag_B_GSE", "sc_position_GSE", "sc_velocity_GSE"]:
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["epoch", "B_GSE_labels"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key in ["mag_B_GSM", "sc_position_GSM", "sc_velocity_GSM"]:
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["epoch", "B_GSM_labels"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key == "mag_B_RTN":
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["epoch", "B_RTN_labels"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key.startswith("codice_hi"):
            data = np.full((n, 4, 15, 4, 4), fillval, dtype=np.float32)
            dims = [
                "epoch",
                "codice_hi_h_spin_angle",
                "codice_hi_h_energy_range",
                "codice_hi_h_spin_sector",
                "codice_hi_h_azimuth",
            ]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key == "swe_counterstreaming_electrons":
            data = np.full(n, fillval, dtype=np.uint8)
            dims = ["epoch"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key.startswith("swe"):
            data = np.full((n, 8), fillval, dtype=np.uint32)
            dims = ["epoch", "swe_electron_energy_labels"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key.startswith("hit"):
            data = np.full(n, fillval, dtype=np.uint32)
            dims = ["epoch"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        else:
            data = np.full(n, fillval, dtype=np.float32)
            dims = ["epoch"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)

    # Populate the dataset variables
    for i, record in enumerate(records):
        for key, val in record.items():
            if key in [
                "apid",
                "met",
                "met_in_utc",
                "ttj2000ns",
                "last_modified",
                "mag_hk_status",
                "spice_kernels",
                "instrument",
            ]:
                continue
            elif key in ["mag_B_GSE", "mag_B_GSM", "mag_B_RTN"]:
                dataset[key].data[i, :] = val
            elif key.startswith("swe_normalized_counts"):
                dataset[key].data[i, :] = val
            elif key.startswith("codice_hi"):
                dataset[key].data[i, :, :, :, :] = val
            else:
                dataset[key].data[i] = val

    return dataset
