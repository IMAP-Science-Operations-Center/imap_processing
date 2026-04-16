"""Functions to support status processing."""

import logging

import xarray as xr

from imap_processing.ialirt.utils.grouping import (
    _populate_instrument_header_items,
)
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def process_status(xarray_data: xr.Dataset) -> list[dict]:
    """
    Create L1 data dictionary.

    Parameters
    ----------
    xarray_data : xr.Dataset
        Parsed data.

    Returns
    -------
    status_data : list[dict]
        Dictionary final data product.
    """
    status_data = []

    # Subsecond time conversion specified in 7516-9054 GSW-FSW ICD.
    # Value of SCLK subseconds, unsigned, (LSB = 1/256 sec)
    met = calculate_time(
        xarray_data["sc_sclk_sec"], xarray_data["sc_sclk_sub_sec"], 256
    )

    # Add required parameters.
    xarray_data["met"] = met

    sc_swapi_status = xarray_data["sc_swapi_status"]
    sc_mag_status = xarray_data["sc_mag_status"]
    sc_hit_status = xarray_data["sc_hit_status"]
    sc_codice_status = xarray_data["sc_codice_status"]
    sc_lo_status = xarray_data["sc_lo_status"]
    sc_hi_45_status = xarray_data["sc_hi_45_status"]
    sc_hi_90_status = xarray_data["sc_hi_90_status"]
    sc_ultra_45_status = xarray_data["sc_ultra_45_status"]
    sc_ultra_90_status = xarray_data["sc_ultra_90_status"]
    sc_swe_status = xarray_data["sc_swe_status"]
    sc_idex_status = xarray_data["sc_idex_status"]
    sc_glows_status = xarray_data["sc_glows_status"]
    sc_autonomy_status = xarray_data["sc_autonomy"]

    for i in range(len(xarray_data["met"])):
        status_data.append(
            _populate_instrument_header_items(met)
            | {
                "instrument": "spacecraft_status",
                "status_epoch": int(met_to_ttj2000ns(met[i])),
                "sc_swapi_status": int(sc_swapi_status[i]),
                "sc_mag_status": int(sc_mag_status[i]),
                "sc_hit_status": int(sc_hit_status[i]),
                "sc_codice_status": int(sc_codice_status[i]),
                "sc_lo_status": int(sc_lo_status[i]),
                "sc_hi_45_status": int(sc_hi_45_status[i]),
                "sc_hi_90_status": int(sc_hi_90_status[i]),
                "sc_ultra_45_status": int(sc_ultra_45_status[i]),
                "sc_ultra_90_status": int(sc_ultra_90_status[i]),
                "sc_swe_status": int(sc_swe_status[i]),
                "sc_idex_status": int(sc_idex_status[i]),
                "sc_glows_status": int(sc_glows_status[i]),
                "sc_autonomy_status": int(sc_autonomy_status[i]),
            }
        )

    return status_data
