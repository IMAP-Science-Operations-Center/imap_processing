from imap_processing.swe.l2.swe_l2 import calculate_particle_energy


def test_calculate_particle_energy():
    """Test calculate_particle_energy function."""
    esa_table_num = 1
    energy = calculate_particle_energy(esa_table_num)
    assert energy.shape == (720,)
