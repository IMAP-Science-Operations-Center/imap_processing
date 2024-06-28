from tools.xtce_generation.xtce_generator_template import editing_packets_dictionary


def test_editing_packets_dictionary(monkeypatch):
    packets = {
        "ana": 0,
        "daralynn": 1,
        "lizzy": 3,
        "maxine": 4,
        "matthew": 5,
        "shawn": 6,
    }
    user_inputs = iter(
        [
            "y"  # Would you like to edit the dictionary? Y/N:
            "1"  # Would like to (1) edit packet_name, (2) edit app_id,
            # (3) delete item, (4) add item (Q) Quit:
            "not in packet"  # What packet name would you like to edit:
            # {packet keys} Enter name:
            "not in packet again"  # Incorrect packet name.
            # Choose from {packet keys} Enter name:
            "shawn"  # Incorrect packet name. Choose from {packet keys} Enter name:
            "sean"  # Enter new packet name:
        ]
    )

    monkeypatch.setattr("builtins.input", lambda _: next(user_inputs))

    editing_packets_dictionary(packets)
