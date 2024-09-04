from imap_processing.spice.spin import generate_spin_table


def test_spin_table():
    start_time = 453051323
    end_time = 453051323 + 86400

    spin_data = generate_spin_table(start_time, end_time)
    print(spin_data)
