import pytest

from imap_processing.ena_maps.ena_maps import HealpixSkyMap, RectangularSkyMap
from imap_processing.ena_maps.utils.naming import (
    MapDescriptor,
    MappableInstrumentShortName,
    build_friendly_date_descriptor,
    ns_to_duration_months,
)
from imap_processing.spice.geometry import SpiceFrame


class TestNaming:
    def test_get_instrument_descriptor(
        self,
    ):
        assert (
            MapDescriptor.get_instrument_descriptor(
                instrument=MappableInstrumentShortName.HI,
                sensor="45",
            )
            == "h45"
        )
        assert (
            MapDescriptor.get_instrument_descriptor(
                instrument=MappableInstrumentShortName.LO_HI_THROUGHPUT,
                sensor=75,
            )
            == "t075"
        )
        assert (
            MapDescriptor.get_instrument_descriptor(
                instrument=MappableInstrumentShortName.IDEX,
                sensor="",
            )
            == "idx"
        )
        assert (
            MapDescriptor.get_instrument_descriptor(
                instrument=MappableInstrumentShortName.ULTRA, sensor="90"
            )
            == "u90"
        )
        assert (
            MapDescriptor.get_instrument_descriptor(
                instrument=MappableInstrumentShortName.ULTRA,
                sensor="combined",
            )
            == "ulc"
        )

        with pytest.raises(
            ValueError, match="Integer sensor values are only valid for LO instruments."
        ):
            MapDescriptor.get_instrument_descriptor(
                instrument=MappableInstrumentShortName.HI,
                sensor=123,
            )

    def test_parse_instrument_descriptor(
        self,
    ):
        assert MapDescriptor.parse_instrument_descriptor("h45") == (
            MappableInstrumentShortName.HI,
            "45",
        )
        assert MapDescriptor.parse_instrument_descriptor("t075") == (
            MappableInstrumentShortName.LO_HI_THROUGHPUT,
            75,
        )
        assert MapDescriptor.parse_instrument_descriptor("idx") == (
            MappableInstrumentShortName.IDEX,
            "",
        )
        assert MapDescriptor.parse_instrument_descriptor("u90") == (
            MappableInstrumentShortName.ULTRA,
            "90",
        )
        assert MapDescriptor.parse_instrument_descriptor("ulc") == (
            MappableInstrumentShortName.ULTRA,
            "combined",
        )
        with pytest.raises(
            ValueError, match="'abc' is not a valid MappableInstrumentShortName"
        ):
            MapDescriptor.parse_instrument_descriptor("abc123456")

    def test_parse_map_duration(
        self,
    ):
        assert MapDescriptor.parse_map_duration("6mo") == "6mo"
        assert MapDescriptor.parse_map_duration("12mo") == "1yr"
        assert MapDescriptor.parse_map_duration(365) == "1yr"
        assert MapDescriptor.parse_map_duration(200) == "6mo"
        assert MapDescriptor.parse_map_duration(60) == "2mo"
        with pytest.raises(ValueError, match="Invalid duration type."):
            MapDescriptor.parse_map_duration(["invalid", "duration"])

    def test_build_l2_map_descriptor_with_int_duration(
        self,
    ):
        descriptor = MapDescriptor(
            instrument=MappableInstrumentShortName.LO_HI_THROUGHPUT,
            frame_descriptor="sf",
            resolution_str="4deg",
            duration=120,
            sensor=75,
            principal_data="ena",
            species="o",
            survival_corrected="nsp",
            spin_phase="anti",
            coordinate_system="hae",
        ).to_string()
        assert descriptor == "t075-ena-o-sf-nsp-anti-hae-4deg-4mo"

    def test_build_l2_map_descriptor_with_spice_frame(
        self,
    ):
        descriptor = MapDescriptor(
            instrument=MappableInstrumentShortName.IDEX,
            frame_descriptor="hk",
            resolution_str="nside32",
            duration=365,
            principal_data="drt",
            species="dust",
            survival_corrected="sp",
            spin_phase="full",
            coordinate_system="hae",
        ).to_string()
        assert descriptor == "idx-drt-dust-hk-sp-full-hae-nside32-1yr"

    # Parameterize over the start_datestring
    @pytest.mark.parametrize(
        "start_datestring",
        [
            "2026-05",
            "2026-05-15",
            "2026-05-15T12:00:00",
        ],
    )
    def test_build_friendly_date_descriptor(self, start_datestring):
        friendly_date_descriptor = build_friendly_date_descriptor(
            start_datestring=start_datestring,
            duration_months=3,
        )
        assert friendly_date_descriptor == "202605m03"

    def test_ns_to_duration_months(
        self,
    ):
        days_per_avg_year = 365.25
        ns_per_day = 24 * 60 * 60 * 1e9

        for fraction_of_year, expected_months in [
            (1 / 2, 6),
            (1 / 3, 4),
            (1 / 4, 3),
            (3 / 4, 9),
        ]:
            assert (
                ns_to_duration_months(fraction_of_year * days_per_avg_year * ns_per_day)
                == expected_months
            )

    def test_get_map_frame(
        self,
    ):
        # Test with a string frame
        assert MapDescriptor.get_map_coord_frame("hae") is SpiceFrame.IMAP_HAE
        assert MapDescriptor.get_map_coord_frame("hre") is SpiceFrame.IMAP_HRE
        assert MapDescriptor.get_map_coord_frame("hnu") is SpiceFrame.IMAP_HNU
        assert MapDescriptor.get_map_coord_frame("gcs") is SpiceFrame.IMAP_GCS
        assert MapDescriptor.get_map_coord_frame("hrc") is SpiceFrame.IMAP_HRC

        # Test with not implemented 'hgi'
        with pytest.raises(KeyError, match="Coordinate frame hgi"):
            MapDescriptor.get_map_coord_frame("hgi")

    def test_get_output_map_structure_from_descriptor_string(self):
        descriptor_str_half_deg = "h45-ena-he-hf-sp-ram-hae-0.5deg-2mo"
        output_map_structure_half_deg = MapDescriptor.from_string(
            descriptor_str_half_deg
        ).to_empty_map()
        assert isinstance(output_map_structure_half_deg, RectangularSkyMap)
        assert output_map_structure_half_deg.spacing_deg == 0.5
        assert (
            output_map_structure_half_deg.spice_reference_frame is SpiceFrame.IMAP_HAE
        )

        descriptor_str_nside32 = "ulc-ena-h-sf-nsp-full-hae-nside32-1yr"
        output_map_structure_nside32 = MapDescriptor.from_string(
            descriptor_str_nside32
        ).to_empty_map()
        assert isinstance(output_map_structure_nside32, HealpixSkyMap)
        assert output_map_structure_nside32.nside == 32

        with pytest.raises(
            ValueError,
            match="Could not interpret resolution string",
        ):
            MapDescriptor.from_string(
                "ulc-ena-h-sf-nsp-full-hae-2abcd32-1yr"
            ).to_empty_map()


class TestMapDescriptor:
    def test_init_and_instrument_descriptor_hi45(self):
        md_h45 = MapDescriptor(
            instrument=MappableInstrumentShortName.HI,
            frame_descriptor="hf",
            resolution_str="2deg",
            duration=60,
            sensor="45",
            principal_data="ena",
            species="he",
            survival_corrected="sp",
            spin_phase="ram",
            coordinate_system="hae",
        )
        assert md_h45.instrument == MappableInstrumentShortName.HI
        assert md_h45.instrument_descriptor == "h45"
        assert md_h45.duration == "2mo"
        assert md_h45.coordinate_system == "hae"
        assert md_h45.map_spice_coord_frame == SpiceFrame.IMAP_HAE

    def test_init_and_instrument_descriptor_lo_hi_throughput_075(self):
        md_lo_hi_075 = MapDescriptor(
            instrument=MappableInstrumentShortName.LO_HI_THROUGHPUT,
            frame_descriptor="sf",
            resolution_str="4deg",
            duration=180,
            sensor=75,
            principal_data="ena",
            species="o",
            survival_corrected="nsp",
            spin_phase="anti",
            coordinate_system="hae",
        )
        assert md_lo_hi_075.instrument == MappableInstrumentShortName.LO_HI_THROUGHPUT
        assert md_lo_hi_075.instrument_descriptor == "t075"
        assert md_lo_hi_075.duration == "6mo"
        assert md_lo_hi_075.coordinate_system == "hae"
        assert md_lo_hi_075.map_spice_coord_frame == SpiceFrame.IMAP_HAE

    def test_from_string(
        self,
    ):
        # Test with Hi45
        descriptor_str = "h45-ena-he-hf-sp-ram-hae-2deg-2mo"
        md = MapDescriptor.from_string(descriptor_str)
        assert md.instrument == MappableInstrumentShortName.HI
        assert md.frame_descriptor == "hf"
        assert md.resolution_str == "2deg"
        assert md.duration == "2mo"
        assert md.sensor == "45"
        assert md.principal_data == "ena"
        assert md.species == "he"
        assert md.survival_corrected == "sp"
        assert md.spin_phase == "ram"
        assert md.coordinate_system == "hae"

    def test_to_string(self):
        # Test with Hi45
        md = MapDescriptor(
            instrument=MappableInstrumentShortName.HI,
            frame_descriptor="hf",
            resolution_str="2deg",
            duration=60,
            sensor="45",
            principal_data="ena",
            species="he",
            survival_corrected="sp",
            spin_phase="ram",
            coordinate_system="hae",
        )
        descriptor_str = md.to_string()
        assert descriptor_str == "h45-ena-he-hf-sp-ram-hae-2deg-2mo"

        # Test with Lo high-throughput 075
        md_lo_hi_075 = MapDescriptor(
            instrument=MappableInstrumentShortName.LO_HI_THROUGHPUT,
            frame_descriptor="sf",
            resolution_str="4deg",
            duration=120,
            sensor=75,
            principal_data="ena",
            species="o",
            survival_corrected="nsp",
            spin_phase="anti",
            coordinate_system="hae",
        )
        descriptor_str_lo_hi_075 = md_lo_hi_075.to_string()
        assert descriptor_str_lo_hi_075 == "t075-ena-o-sf-nsp-anti-hae-4deg-4mo"

        # Test with Ultra combined and 365 days
        md_ultra_combined = MapDescriptor(
            instrument=MappableInstrumentShortName.ULTRA,
            frame_descriptor="sf",
            resolution_str="nside32",
            duration=365,
            sensor="combined",
            principal_data="ena",
            species="h",
            survival_corrected="nsp",
        )
        descriptor_str_ultra_combined = md_ultra_combined.to_string()
        assert descriptor_str_ultra_combined == "ulc-ena-h-sf-nsp-full-hae-nside32-1yr"

    @pytest.mark.parametrize(
        "descriptor_str, expected_catdesc",
        [
            (
                "h45-spx-h-hf-sp-ram-hae-4deg-3mo",
                "IMAP Hi45 H Spectral Index, "
                "HAE Helio Frame, Surv Corr, Ram, 4 deg, 3 Mon",
            ),
            (
                "h45-spx0305-h-hf-sp-ram-hae-4deg-3mo",
                "IMAP Hi45 H Spectral Index, "
                "HAE Helio Frame, Surv Corr, Ram, 4 deg, 3 Mon",
            ),
            (
                "hic-ena-h-hf-sp-ram-hae-4deg-3mo",
                "IMAP Hi Combined H Inten, HAE Helio Frame, Surv Corr, Ram,"
                " 4 deg, 3 Mon",
            ),
            (
                "u45-ena-h-hf-sp-ram-hae-4deg-3mo",
                "IMAP Ultra45 H Inten, HAE Helio Frame, Surv Corr, Ram, 4 deg, 3 Mon",
            ),
            (
                "u45-ena-h-hf-sp-full-hae-4deg-3mo",
                "IMAP Ultra45 H Inten, HAE Helio Frame, Surv Corr, Full Spin,"
                " 4 deg, 3 Mon",
            ),
            (
                "u45-ena-h-hf-sp-ram-hae-nside128-3mo",
                "IMAP Ultra45 H Inten, HAE Helio Frame, Surv Corr, Ram, NSide 128,"
                " 3 Mon",
            ),
            (
                "u45-enaCUSTOM-h-hf-sp-ram-hae-4deg-3mo",
                "IMAP Ultra45 H Inten, HAE Helio Frame, Surv Corr, Ram, 4 deg, 3 Mon",
            ),
            (
                "l090-enanbs-h-sf-nsp-ram-hae-6deg-1yr",
                "IMAP Lo90 H Inten, HAE SC Frame, No Surv Corr, Ram, 6 deg, 1 Yr,"
                " No sputter/bootstrap",
            ),
            (
                "t090-ena-o-sf-nsp-ram-hae-6deg-1yr",
                "IMAP Lo90 O Inten, HAE SC Frame, No Surv Corr, Ram, 6 deg, 1 Yr",
            ),
            (
                "l090-ena-h-hf-nsp-ram-gcs-6deg-1yr",
                "IMAP Lo90 H Inten, GCS Helio Frame, No Surv Corr, Ram, 6 deg, 1 Yr",
            ),
            (
                "l090-isn-h-sf-nsp-ram-hae-6deg-1yr",
                "IMAP Lo90 ISN H Rate, HAE SC Frame, No Surv Corr, Ram, 6 deg, 1 Yr",
            ),
            (
                "l090-isnnbkgnd-h-sf-nsp-ram-hae-6deg-1yr",
                "IMAP Lo90 ISN H Rate, HAE SC Frame, No Surv Corr, Ram, 6 deg, 1 Yr,"
                " No bkgnd sub",
            ),
            (
                "glx-int-uv-sf-nsp-full-hae-6deg-1yr",
                "IMAP GLOWS UV Inten, HAE SC Frame, No Surv Corr, Full Spin, 6 deg,"
                " 1 Yr",
            ),
            (
                "idx-drt-dust-sf-nsp-full-hae-6deg-1yr",
                "IMAP IDEX Dust Rate, HAE SC Frame, No Surv Corr, Full Spin, 6 deg,"
                " 1 Yr",
            ),
        ],
    )
    def test_build_map_var_catdesc_with_primary_vars(
        self, descriptor_str, expected_catdesc
    ):
        # Use case is primarily from descriptor str to CATDESC
        md = MapDescriptor.from_string(descriptor_str)
        actual_catdesc = md.build_map_var_catdesc(md.principal_data_var)
        assert actual_catdesc == expected_catdesc

    @pytest.mark.parametrize(
        "descriptor_str, var_name, expected_catdesc",
        [
            (
                "hic-spx-h-hf-sp-ram-hae-4deg-3mo",
                "ena_spectral_index_stat_uncert",
                "IMAP Hi Combined H Spectral Stat. Unc., "
                "HAE Helio Frame, Surv Corr, Ram, 4 deg, 3 Mon",
            ),
            (
                "hic-spx-h-hf-sp-ram-hae-4deg-3mo",
                "ena_spectral_scalar",
                "IMAP Hi Combined H Spectral Scalar, "
                "HAE Helio Frame, Surv Corr, Ram, 4 deg, 3 Mon",
            ),
            (
                "hic-spx-h-hf-sp-ram-hae-4deg-3mo",
                "ena_spectral_scalar_stat_uncert",
                "IMAP Hi Combined H Spectral Scalar Stat. Unc., "
                "HAE Helio Frame, Surv Corr, Ram, 4 deg, 3 Mon",
            ),
            (
                "hic-spx-h-hf-sp-ram-hae-4deg-3mo",
                "ena_spectral_index_chisq",
                "IMAP Hi Combined H Spectral Index Chisq, "
                "HAE Helio Frame, Surv Corr, Ram, 4 deg, 3 Mon",
            ),
            (
                "hic-ena-h-hf-sp-ram-hae-4deg-3mo",
                "ena_intensity_stat_uncert",
                "IMAP Hi Combined H Inten Stat. Unc., "
                "HAE Helio Frame, Surv Corr, Ram, 4 deg, 3 Mon",
            ),
            (
                "hic-ena-h-hf-sp-ram-hae-4deg-3mo",
                "ena_intensity_sys_err",
                "IMAP Hi Combined H Inten Sys. Err., "
                "HAE Helio Frame, Surv Corr, Ram, 4 deg, 3 Mon",
            ),
            (
                "l075-enanbs-h-sf-nsp-anti-hae-6deg-6mo",
                "bg_intensity",
                "IMAP Lo75 H Background Inten, "
                "HAE SC Frame, No Surv Corr, Anti, 6 deg, 6 Mon, No sputter/bootstrap",
            ),
            (
                "l075-enanbs-h-sf-nsp-anti-hae-6deg-6mo",
                "bg_intensity_stat_uncert",
                "IMAP Lo75 H Background Inten Stat. Unc., "
                "HAE SC Frame, No Surv Corr, Anti, 6 deg, 6 Mon, No sputter/bootstrap",
            ),
            (
                "l075-enanbs-h-sf-nsp-anti-hae-6deg-6mo",
                "bg_intensity_sys_err",
                "IMAP Lo75 H Background Inten Sys. Err., "
                "HAE SC Frame, No Surv Corr, Anti, 6 deg, 6 Mon, No sputter/bootstrap",
            ),
            (
                "l075-enanbs-h-sf-nsp-anti-hae-6deg-6mo",
                "bg_rate",
                "IMAP Lo75 H Background Count Rate, "
                "HAE SC Frame, No Surv Corr, Anti, 6 deg, 6 Mon, No sputter/bootstrap",
            ),
            (
                "l075-enanbs-h-sf-nsp-anti-hae-6deg-6mo",
                "ena_count_rate",
                "IMAP Lo75 H ENA Count Rate, "
                "HAE SC Frame, No Surv Corr, Anti, 6 deg, 6 Mon, No sputter/bootstrap",
            ),
            (
                "l075-enanbs-h-sf-nsp-anti-hae-6deg-6mo",
                "ena_intensity_sys_err_minus",
                "IMAP Lo75 H Inten Sys. Err. Lower, "
                "HAE SC Frame, No Surv Corr, Anti, 6 deg, 6 Mon, No sputter/bootstrap",
            ),
            (
                "l075-enanbs-h-sf-nsp-anti-hae-6deg-6mo",
                "ena_intensity_sys_err_plus",
                "IMAP Lo75 H Inten Sys. Err. Upper, "
                "HAE SC Frame, No Surv Corr, Anti, 6 deg, 6 Mon, No sputter/bootstrap",
            ),
            (
                "u45-ena-h-hf-nsp-full-hae-4deg-6mo",
                "counts",
                "IMAP Ultra45 H Counts, "
                "HAE Helio Frame, No Surv Corr, Full Spin, 4 deg, 6 Mon",
            ),
            (
                "u45-ena-h-hf-nsp-full-hae-4deg-6mo",
                "aliens",
                None,
            ),
            (
                "u45-ena-h-hf-nsp-full-hae-4deg-6mo",
                "survival_probability",
                "IMAP Ultra45 H Survival Probability, "
                "HAE Helio Frame, No Surv Corr, Full Spin, 4 deg, 6 Mon",
            ),
        ],
    )
    def test_build_map_var_catdesc_more_vars(
        self, descriptor_str, var_name, expected_catdesc
    ):
        md = MapDescriptor.from_string(descriptor_str)
        actual_catdesc = md.build_map_var_catdesc(var_name)
        assert actual_catdesc == expected_catdesc

    @pytest.mark.parametrize(
        "descriptor_str, expected_principal_data_var",
        [
            ("hic-ena-h-hf-sp-ram-hae-4deg-3mo", "ena_intensity"),
            ("h45-spx0305-h-hf-sp-ram-hae-4deg-3mo", "ena_spectral_index"),
            ("idx-drt-dust-sf-nsp-full-hae-6deg-1yr", "dust_rate"),
            ("glx-int-uv-sf-nsp-full-hae-6deg-1yr", "glows_rate"),
            ("l090-isnnbkgnd-h-sf-nsp-ram-hae-6deg-1yr", "isn_rate"),
            ("l090-isn-h-sf-nsp-ram-hae-6deg-1yr", "isn_rate_bg_subtracted"),
        ],
    )
    def test_principal_data_var(self, descriptor_str, expected_principal_data_var):
        md = MapDescriptor.from_string(descriptor_str)
        assert md.principal_data_var == expected_principal_data_var
