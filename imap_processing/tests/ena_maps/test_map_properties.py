from pathlib import Path

import pytest

from imap_processing import imap_module_directory
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils import map_properties
from imap_processing.spice.geometry import SpiceFrame


class TestMapProperties:
    def test_instantiate_dict_healpix(
        self,
    ):
        # First try (expecting error without nside)
        props_dict_healpix = {
            "sky_tiling_type": "HEALPIX",
            "spice_reference_frame": "ECLIPJ2000",
            "projection_method_and_values": {
                "PUSH": ["density", "flux"],
                "PULL": ["foo", "bar"],
            },
            "nested": True,
        }

        with pytest.raises(KeyError):
            props = map_properties.MapProperties(props_dict_healpix)

        # Now with nside set
        props_dict_healpix["nside"] = 32

        props = map_properties.MapProperties(props_dict_healpix)
        assert props.sky_tiling_type is ena_maps.SkyTilingType.HEALPIX
        assert props.nside == 32
        assert props.nested is True
        assert props.spice_reference_frame is SpiceFrame.ECLIPJ2000

        assert props.values_to_push_project == ["density", "flux"]
        assert props.values_to_pull_project == ["foo", "bar"]

    def test_instantiate_dict_rectangular(
        self,
    ):
        # First try (expecting error without spacing_deg)
        props_dict_rect = {
            "sky_tiling_type": "RECTANGULAR",
            "spice_reference_frame": "ECLIPJ2000",
            "projection_method_and_values": {
                "PUSH": ["density", "flux"],
                "PULL": ["foo", "bar"],
            },
        }

        with pytest.raises(KeyError):
            props = map_properties.MapProperties(props_dict_rect)

        # Now with spacing_deg set
        props_dict_rect["spacing_deg"] = 1.0

        props = map_properties.MapProperties(props_dict_rect)
        assert props.sky_tiling_type is ena_maps.SkyTilingType.RECTANGULAR
        assert props.spice_reference_frame is SpiceFrame.ECLIPJ2000
        assert props.spacing_deg == 1.0

        assert props.values_to_push_project == ["density", "flux"]
        assert props.values_to_pull_project == ["foo", "bar"]

    def test_instantiate_json_path(
        self,
    ):
        json_path = (
            Path(imap_module_directory)
            / "tests"
            / "ena_maps"
            / "data"
            / "test_ultra_l2_map_config.json"
        )

        props_from_path = map_properties.MapProperties(json_path)
        assert props_from_path.sky_tiling_type is ena_maps.SkyTilingType.HEALPIX
        assert props_from_path.nside == 32
        assert props_from_path.spice_reference_frame is SpiceFrame.ECLIPJ2000

        props_from_str = map_properties.MapProperties(str(json_path))
        assert props_from_str.sky_tiling_type is ena_maps.SkyTilingType.HEALPIX
        assert props_from_str.nside == 32
        assert props_from_str.spice_reference_frame is SpiceFrame.ECLIPJ2000

    def test_to_skymap_healpix(
        self,
    ):
        props_dict_healpix = {
            "sky_tiling_type": "HEALPIX",
            "spice_reference_frame": "ECLIPJ2000",
            "projection_method_and_values": {
                "PUSH": ["density", "flux"],
                "PULL": ["foo", "bar"],
            },
            "nside": 32,
            "nested": True,
        }
        props = map_properties.MapProperties(props_dict_healpix)
        skymap = props.to_skymap()
        assert isinstance(skymap, ena_maps.HealpixSkyMap)
        assert skymap.nside == 32
        assert skymap.nested is True
        assert skymap.spice_reference_frame is SpiceFrame.ECLIPJ2000

    def test_to_skymap_rectangular(
        self,
    ):
        props_dict_rect = {
            "sky_tiling_type": "RECTANGULAR",
            "spice_reference_frame": "ECLIPJ2000",
            "projection_method_and_values": {
                "PUSH": ["density", "flux"],
                "PULL": ["foo", "bar"],
            },
            "spacing_deg": 1.0,
        }
        props = map_properties.MapProperties(props_dict_rect)
        skymap = props.to_skymap()
        assert isinstance(skymap, ena_maps.RectangularSkyMap)
        assert skymap.spacing_deg == 1.0
        assert skymap.spice_reference_frame is SpiceFrame.ECLIPJ2000
