"""Simple class to hold ENA Sky Map configuration properties."""

import json
from pathlib import Path

from imap_processing.ena_maps import ena_maps
from imap_processing.spice import geometry


class MapProperties:
    """
    Class to store ENA Sky Map configuration properties.

    Parameters
    ----------
    properties : dict | str | Path
        A dictionary or a path to a JSON file containing the map properties.
        The dictionary should contain the following keys:
            - sky_tiling_type: str
                Type of sky tiling (HEALPIX or RECTANGULAR).
            - spice_reference_frame: int | str
                Reference frame for the map (e.g., ECLIPJ2000).
            - projection_method_and_values: dict
                Dictionary with keys "PUSH" and/or "PULL" containing lists of values.
            - If sky_tiling_type is HEALPIX:
                - nside: int
                    Nside parameter for HEALPIX maps.
                - nested: bool
                    Nested parameter for HEALPIX maps.
            - If sky_tiling_type is RECTANGULAR:
                - spacing_deg: float
                    Spacing in degrees for rectangular maps.
    """

    def __init__(self, properties: dict | str | Path):
        if isinstance(properties, str | Path):
            with open(properties) as f:
                properties = json.load(f)
        if not isinstance(properties, dict):
            raise TypeError(
                "Properties must be a dictionary or a path to a JSON file which"
                " contains a dictionary."
            )
        self.sky_tiling_type = ena_maps.SkyTilingType[properties["sky_tiling_type"]]
        spice_reference_frame = properties["spice_reference_frame"]

        if isinstance(spice_reference_frame, int):
            self.spice_reference_frame = geometry.SpiceFrame(spice_reference_frame)
        else:
            self.spice_reference_frame = geometry.SpiceFrame[spice_reference_frame]

        projection_method_and_values = properties["projection_method_and_values"]
        self.values_to_push_project = projection_method_and_values.get("PUSH", [])
        self.values_to_pull_project = projection_method_and_values.get("PULL", [])

        if self.sky_tiling_type is ena_maps.SkyTilingType.HEALPIX:
            self.nside = properties["nside"]
            self.nested = properties["nested"]
        elif self.sky_tiling_type is ena_maps.SkyTilingType.RECTANGULAR:
            self.spacing_deg = properties["spacing_deg"]

    def to_skymap(self) -> ena_maps.HealpixSkyMap | ena_maps.RectangularSkyMap:
        """
        Generate an empty SkyMap object based on the properties.

        Returns
        -------
        ena_maps.HealpixSkyMap | ena_maps.RectangularSkyMap
            The generated SkyMap object.
            This object can be either a HealpixSkyMap or RectangularSkyMap.

        Raises
        ------
        ValueError
            If the sky tiling type is unknown.
        """
        if self.sky_tiling_type is ena_maps.SkyTilingType.HEALPIX:
            return ena_maps.HealpixSkyMap(
                nside=self.nside,
                nested=self.nested,
                spice_frame=self.spice_reference_frame,
            )
        elif self.sky_tiling_type is ena_maps.SkyTilingType.RECTANGULAR:
            return ena_maps.RectangularSkyMap(
                spacing_deg=self.spacing_deg, spice_frame=self.spice_reference_frame
            )
        else:
            raise ValueError(f"Unknown sky tiling type: {self.sky_tiling_type}")


# Default properties for the Ultra L2 map
DEFAULT_ULTRA_L2_MAP_PROPERTIES = MapProperties(
    {
        "sky_tiling_type": "HEALPIX",
        "spice_reference_frame": "ECLIPJ2000",
        "projection_method_and_values": {
            "PUSH": [
                "counts",
                "exposure_factor",
                "sensitivity",
                "background_rates",
            ],
            "PULL": [],
        },
        "nside": 32,
        "nested": False,
    }
)
