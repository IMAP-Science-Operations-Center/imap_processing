"""Creates xarray based on structure of queried DynamoDB."""

from collections import defaultdict
import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice.constants import (
    HI_IALIRT_ELEVATION_ANGLE,
    HI_IALIRT_REF_SPIN_ANGLE,
)
from imap_processing.ialirt.utils.constants import (
    IALIRT_DTYPES,
    IALIRT_DIMS,
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

    ONE_EPOCH = {"codice_lo", "hit", "swapi", "swe", "spacecraft", "mag"}
    MULTI_EPOCH = {"codice_hi"}

    epochs = {inst: [] for inst in (ONE_EPOCH | MULTI_EPOCH)}
    by_inst: dict[str, list[dict]] = defaultdict(list)

    for r in records:
        inst = r.get("instrument")
        by_inst[r["instrument"]].append(r)
        if inst in ONE_EPOCH:
            epochs[inst].append(r["mag_epoch"] if inst == "mag" else r["ttj2000ns"])
        elif inst in MULTI_EPOCH:
            epochs[inst].extend(r["codice_hi_epoch"])

    epoch_arrays = {}

    for inst, arr in epochs.items():
        coord = f"{inst}_epoch"
        epoch_arrays[coord] = xr.DataArray(
            data=arr,
            name=coord,
            dims=[coord],
            attrs=cdf_manager.get_variable_attributes(coord, check_schema=False),
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

    elevation = xr.DataArray(
        HI_IALIRT_ELEVATION_ANGLE,
        name="codice_hi_elevation",
        dims=["codice_hi_elevation"],
        attrs=cdf_manager.get_variable_attributes("codice_hi_elevation", check_schema=False),
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
        dims=["codice_hi_spin_sector", "codice_hi_elevation"],
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

    coords = {
        "codice_hi_epoch": epoch_arrays["codice_hi_epoch"],
        "codice_lo_epoch": epoch_arrays["codice_lo_epoch"],
        "hit_epoch": epoch_arrays["hit_epoch"],
        "mag_epoch": epoch_arrays["mag_epoch"],
        "swapi_epoch": epoch_arrays["swapi_epoch"],
        "swe_epoch": epoch_arrays["swe_epoch"],
        "B_GSM_labels": gsm_component,
        "B_GSE_labels": gse_component,
        "B_RTN_labels": rtn_component,
        "sc_GSM_labels": sc_gsm_component,
        "sc_GSE_labels": sc_gse_component,
        "swe_electron_energy_labels": swe_electron_energy_labels,
        "codice_hi_h_energy_range_labels": energy_range,
        "codice_hi_elevation": elevation,
        "codice_hi_spin_angle": spin_angle,
        "codice_hi_spin_sector": spin_sector,
    }
    dataset = xr.Dataset(
        coords=coords,
        attrs=cdf_manager.get_global_attributes("imap_ialirt_l1_realtime"),
    )

    for key in IALIRT_DIMS.keys():
        dims = IALIRT_DIMS[key]
        attrs = cdf_manager.get_variable_attributes(key, check_schema=False)
        fill = attrs["FILLVAL"]
        dtype = IALIRT_DTYPES[key]

        shape = [dataset.dims[d] for d in dims]

        data = np.full(shape, fill, dtype=dtype)
        dataset[key] = xr.DataArray(data, dims=dims, attrs=attrs)

    for i, r in enumerate(by_inst.get("mag", [])):
        dataset["mag_B_GSE"].data[i, :] = np.asarray(r["mag_B_GSE"], dtype=np.float32)
        dataset["mag_B_GSM"].data[i, :] = np.asarray(r["mag_B_GSM"], dtype=np.float32)
        dataset["mag_B_RTN"].data[i, :] = np.asarray(r["mag_B_RTN"], dtype=np.float32)

        dataset["mag_B_magnitude"].data[i] = np.float32(r["mag_B_magnitude"])
        dataset["mag_theta_B_GSE"].data[i] = np.float32(r["mag_theta_B_GSE"])
        dataset["mag_phi_B_GSE"].data[i] = np.float32(r["mag_phi_B_GSE"])
        dataset["mag_theta_B_GSM"].data[i] = np.float32(r["mag_theta_B_GSM"])
        dataset["mag_phi_B_GSM"].data[i] = np.float32(r["mag_phi_B_GSM"])

    for i, r in enumerate(by_inst.get("codice_hi", [])):
        sl = slice(4 * i, 4 * i + 4)

        hi = np.asarray(r["codice_hi_h"], dtype=np.float32)

        # Allow either (4,15,4,4) or (15,4,4)
        if hi.shape == (15, 4, 4):
            hi = np.repeat(hi[None, ...], 4, axis=0)

        dataset["codice_hi_h"].data[sl, :, :, :] = hi

    CODICE_LO_KEYS = [
        "codice_lo_c_over_o_abundance",
        "codice_lo_mg_over_o_abundance",
        "codice_lo_fe_over_o_abundance",
        "codice_lo_c_plus_6_over_c_plus_5_ratio",
        "codice_lo_o_plus_7_over_o_plus_6_ratio",
        "codice_lo_fe_low_over_fe_high_ratio",
    ]

    for i, r in enumerate(by_inst.get("codice_lo", [])):
        for k in CODICE_LO_KEYS:
            dataset[k].data[i] = np.float32(r[k])

    HIT_KEYS = [
        "hit_e_a_side_low_en",
        "hit_e_a_side_med_en",
        "hit_e_b_side_low_en",
        "hit_e_b_side_med_en",
        "hit_h_omni_low_en",
        "hit_h_omni_med_en",
        "hit_he_omni_low_en",
        "hit_he_omni_high_en",
    ]

    for i, r in enumerate(by_inst.get("hit", [])):
        for k in HIT_KEYS:
            dataset[k].data[i] = np.uint32(r[k])

    for i, r in enumerate(by_inst.get("swapi", [])):
        dataset["swapi_pseudo_proton_density"].data[i] = np.float32(r["swapi_pseudo_proton_density"])
        dataset["swapi_pseudo_proton_speed"].data[i] = np.float32(r["swapi_pseudo_proton_speed"])
        dataset["swapi_pseudo_proton_temperature"].data[i] = np.float32(r["swapi_pseudo_proton_temperature"])

    for i, r in enumerate(by_inst.get("swe", [])):
        dataset["swe_normalized_counts"].data[i, :] = np.asarray(r["swe_normalized_counts"], dtype=np.uint32)
        dataset["swe_counterstreaming_electrons"].data[i] = np.uint8(r["swe_counterstreaming_electrons"])

    for i, r in enumerate(by_inst.get("spacecraft", [])):
        dataset["sc_position_GSE"].data[i, :] = np.asarray(r["sc_position_GSE"], dtype=np.float32)
        dataset["sc_velocity_GSE"].data[i, :] = np.asarray(r["sc_velocity_GSE"], dtype=np.float32)
        dataset["sc_position_GSM"].data[i, :] = np.asarray(r["sc_position_GSM"], dtype=np.float32)
        dataset["sc_velocity_GSM"].data[i, :] = np.asarray(r["sc_velocity_GSM"], dtype=np.float32)

    return dataset
