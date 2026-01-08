"""Creates xarray based on structure of queried DynamoDB."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice.constants import (
    HI_IALIRT_ELEVATION_ANGLE,
    HI_IALIRT_REF_SPIN_ANGLE,
)
from imap_processing.ialirt.utils.constants import (
    IALIRT_KEYS,
    codice_energy_bounds,
    swe_energy_labels,
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

    n = len(records)
    attrs = cdf_manager.get_variable_attributes("default_int64_attrs")
    fillval = attrs.get("FILLVAL")
    ttj2000ns_values = np.full(n, fillval, dtype=np.int64)
    codice_hi_ttj2000ns_values = np.full(n * 4, fillval, dtype=np.int64)
    mag_ttj2000ns_values = np.full(n * 4, fillval, dtype=np.int64)
    attrs = cdf_manager.get_variable_attributes("default_float32_attrs")
    fillval = attrs.get("FILLVAL")
    elevation_values = np.full(n * 4, fillval, dtype=np.float32)
    spin_angle_values = np.full(n * 4, fillval, dtype=np.float32)

    # Collect all keys that start with the instrument prefixes.
    for i, record in enumerate(records):
        ttj2000ns_values[i] = record["ttj2000ns"]
        if record["instrument"] == "codice_hi":
            codice_hi_ttj2000ns_values[4 * i : 4 * i + 4] = record["codice_hi_epoch"]
            elevation_values[4 * i : 4 * i + 4] = record["codice_hi_h_elevation_angle"]
            spin_angle_values[4 * i : 4 * i + 4] = record["codice_hi_h_spin_angle"]
        if record["instrument"] == "mag":
            mag_ttj2000ns_values[4 * i : 4 * i + 4] = record["mag_epoch"]

    epoch = xr.DataArray(
        data=ttj2000ns_values,
        name="epoch",
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("epoch", check_schema=False),
    )

    codice_hi_epoch = xr.DataArray(
        data=codice_hi_ttj2000ns_values,
        name="codice_hi_epoch",
        dims=["codice_hi_epoch"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_epoch", check_schema=False
        ),
    )

    mag_epoch = xr.DataArray(
        data=mag_ttj2000ns_values,
        name="mag_epoch",
        dims=["mag_epoch"],
        attrs=cdf_manager.get_variable_attributes("mag_epoch", check_schema=False),
    )

    sc_gsm_component = xr.DataArray(
        ["x (GSM)", "y (GSM)", "z (GSM)"],
        name="sc_GSM_labels",
        dims=["sc_GSM_labels"],
        attrs=cdf_manager.get_variable_attributes("sc_GSM_labels", check_schema=False),
    )

    sc_gse_component = xr.DataArray(
        ["x (GSE)", "y (GSE)", "z (GSE)"],
        name="sc_GSE_labels",
        dims=["sc_GSE_labels"],
        attrs=cdf_manager.get_variable_attributes("sc_GSE_labels", check_schema=False),
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
        data=[f"{label} eV" for label in swe_energy_labels],
        name="swe_electron_energy_labels",
        dims=["swe_electron_energy_labels"],
        attrs=cdf_manager.get_variable_attributes(
            "swe_electron_energy_labels", check_schema=False
        ),
    )

    energy_range = xr.DataArray(
        data=[f"{low:.4f}-{high:.4f} MeV" for low, high in codice_energy_bounds],
        name="codice_hi_h_energy_range_labels",
        dims=["codice_hi_h_energy_range_labels"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_h_energy_range_labels", check_schema=False
        ),
    )

    elevation = (
        xr.DataArray(
            HI_IALIRT_ELEVATION_ANGLE,
            name="codice_hi_elevation",
            dims=("codice_hi_elevation",),
            attrs=cdf_manager.get_variable_attributes(
                "codice_hi_elevation", check_schema=False
            ),
        ),
    )

    elevation_labels = xr.DataArray(
        HI_IALIRT_ELEVATION_ANGLE.astype(str),
        name="codice_hi_elevation_labels",
        dims=["codice_hi_elevation"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_elevation_labels", check_schema=False
        ),
    )

    # Calculate spin angle for CoDICE-Hi
    # Formula:
    #   θ_(g,n) = (θ_(g,0)+90°* n)  mod 360°
    # where
    #   n is number of sectored angles, 0 to 3,
    #   g is size of the group (inst_az), 0 to 3,
    # Calculate spin angle by adding a base angle from L2_HI_SECTORED_ANGLE
    # for each SSD index and then adding multiple of 30 degrees for each elevation.
    # Then mod by 360 to keep it within 0-360 range.
    spin_angles = (
        HI_IALIRT_REF_SPIN_ANGLE[:, np.newaxis] + np.array([0, 1, 2, 3]) * 90
    ) % 360.0
    spin_angle = xr.DataArray(
        data=spin_angles,
        name="codice_hi_spin_angle",
        dims=["spin_sector", "elevation_angle"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_spin_angle", check_schema=False
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

    spin_sector_labels = xr.DataArray(
        data=["0", "1", "2", "3"],
        name="codice_hi_h_spin_sector_labels",
        dims=["codice_hi_h_spin_sector"],
        attrs=cdf_manager.get_variable_attributes(
            "codice_hi_h_spin_sector_labels", check_schema=False
        ),
    )

    coords = {
        "epoch": epoch,
        "mag_epoch": mag_epoch,
        "codice_hi_epoch": codice_hi_epoch,
        "B_GSM_labels": gsm_component,
        "B_GSE_labels": gse_component,
        "B_RTN_labels": rtn_component,
        "sc_GSM_labels": sc_gsm_component,
        "sc_GSE_labels": sc_gse_component,
        "swe_electron_energy_labels": swe_electron_energy_labels,
        "codice_hi_h_energy_range": energy_range,
        "codice_hi_elevation": elevation,
        "codice_hi_elevation_labels": elevation_labels,
        "codice_hi_spin_angle": spin_angle,
        "codice_hi_spin_sector": spin_sector,
        "codice_hi_spin_sector_labels": spin_sector_labels,
    }
    dataset = xr.Dataset(
        coords=coords,
        attrs=cdf_manager.get_global_attributes("imap_ialirt_l1_realtime"),
    )

    # Create empty dataset for each key.
    for key in IALIRT_KEYS:
        attrs = cdf_manager.get_variable_attributes(key, check_schema=False)
        fillval = attrs.get("FILLVAL")
        if key == "mag_B_GSE":
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["mag_epoch", "B_GSE_labels"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key in ["sc_position_GSE", "sc_velocity_GSE"]:
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["epoch", "sc_GSE_labels"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key in ["sc_position_GSM", "sc_velocity_GSM"]:
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["epoch", "sc_GSM_labels"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key == "mag_B_GSM":
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["mag_epoch", "B_GSM_labels"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key == "mag_B_RTN":
            data = np.full((n, 3), fillval, dtype=np.float32)
            dims = ["mag_epoch", "B_RTN_labels"]
            dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)
        elif key.startswith("codice_hi"):
            data = np.full((4 * n, 15, 4, 4), fillval, dtype=np.float32)
            dims = [
                "codice_hi_epoch",
                "codice_hi_energy_range",
                "codice_hi_spin_sector",
                "codice_hi_elevation",
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
