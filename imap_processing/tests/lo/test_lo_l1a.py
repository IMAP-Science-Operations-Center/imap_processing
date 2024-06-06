from imap_processing.lo.l1a.lo_l1a import lo_l1a


def test_lo_l1a_de():
    print(lo_l1a("imap_lo_l0_de_20240604_v01.pkts"))


def test_lo_l1a_spin():
    print(lo_l1a("imap_lo_l0_spin_20240604_v01.pkts"))
