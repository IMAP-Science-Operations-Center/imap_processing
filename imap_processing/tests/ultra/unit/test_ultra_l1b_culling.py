"""Tests Culling for ULTRA L1b."""

from imap_processing.ultra.l1b.ultra_l1b_culling import get_spin

# @pytest.fixture()
# def pointing_frame_kernels(spice_test_data_path):
#     """List SPICE kernels."""
#     required_kernels = [
#         "imap_wkcp.tf",
#         "imap_sclk_0000.tsc",
#         "naif0012.tls",
#     ]
#     kernels = [str(spice_test_data_path / kernel) for kernel in required_kernels]
#     return kernels

# @pytest.mark.usefixtures("use_fake_spin_data_for_time")
# def test_cull_something(use_fake_spin_data_for_time, l1c_culling_test_data,
#                         pointing_frame_kernels):
#
#     df = pd.read_csv(l1c_culling_test_data, delim_whitespace=True,
#                      comment='#', header=0)
#
#     spice.furnsh(pointing_frame_kernels)
#     # Get IDs.
#     # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.gipool
#     id_imap_sclk = spice.gipool("CK_-43000_SCLK", 0, 1)
#     # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.sce2c
#     # Convert start and end times to SCLK.
#     sclk_begtim = spice.sce2c(int(id_imap_sclk), df['tdb'].min())
#     met_begtime = sclkticks_to_met(sclk_begtim)
#
#     # Do one pointing at a time
#     use_fake_spin_data_for_time(met_begtime)


def test_de_dataset(use_fake_spin_data_for_time, l1b_de_dataset):
    l1b_de_dataset["de_event_met"]
    spin_number = l1b_de_dataset["SPINNUMBER"].values

    # Create fake universal spin table
    use_fake_spin_data_for_time(
        l1b_de_dataset["de_event_met"][0], l1b_de_dataset["de_event_met"][-1]
    )
    # From universal spin table
    # From l1b de dataset
    energy = de_dataset["ENERGY_PH"].values
    # 0-10, 10-20, above 20 keV

    get_spin(l1b_de_dataset["de_event_met"], l1b_de_dataset["energy"])
    print("hi")
