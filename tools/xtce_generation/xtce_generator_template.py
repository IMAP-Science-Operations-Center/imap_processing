"""Docstring."""

from pathlib import Path

from tools.xtce_generation.telemetry_generator import TelemetryGenerator

# I am changing this


def editing_packets_dictionary(packets: dict):
    """
    Small file to edit dictionary.

    Parameters
    ----------
    packets : dict
       Object holding packets.
    """
    change_input = input("Would you like to edit the dictionary? Y/N: ")
    keep_editing_condition = True
    while keep_editing_condition is True:
        if change_input in ("y", "Y"):
            edit_action = input(
                "Would like to (1) edit packet_name, (2) edit app_id, "
                "(3) delete item, (4) add item (Q) Quit: "
            )
            # Edit packet_name
            if edit_action == "1":
                packet_name_edit = input(
                    "What packet name would you like to edit: "
                    + str(packets.keys())
                    + "\nEnter name: "
                )
                while packet_name_edit not in packets.keys():
                    packet_name_edit = input(
                        "Incorrect packet name. Choose from "
                        + str(packets.keys())
                        + "\nEnter name: "
                    )
                    # Getting new name
                packet_name_final = input("Enter new packet name: ")
                # Implementing new name
                packets[packet_name_final] = packets[packet_name_edit]
                del packets[packet_name_edit]

                print("Here is the dictionary you have generated: ")
                print(packets)

            # Edit app_id
            if edit_action == "2":
                app_id_edit = input(
                    "Enter packet_name of app_id you would like to edit: "
                    + str(packets)
                    + "\nEnter name: "
                )
                while app_id_edit not in packets.keys():
                    app_id_edit = input(
                        "Incorrect packet name. Choose from "
                        + str(packets)
                        + "\nEnter name:"
                    )
                app_id_final = input("Enter new app_id: ")
                packets[app_id_edit] = app_id_final

                print("Here is the dictionary you have generated: ")
                print(packets)

            # Delete item
            if edit_action == "3":
                delete_item = input(
                    "What packet would you like to delete: "
                    + str(packets.keys())
                    + "\nEnter name: "
                )
                while delete_item not in packets.keys():
                    delete_item = input(
                        "Incorrect packet name. Choose from "
                        + str(packets.keys())
                        + "\nEnter name:"
                    )
                del packets[delete_item]

                print("Here is the dictionary you have generated: ")
                print(packets)

            # Add item
            if edit_action == "4":
                packet_name_input = input("Enter instrument name: ")
                app_id_num_input = input("Enter app id: ")

                # Setting dictionary item
                packets.update({packet_name_input: app_id_num_input})
            if edit_action in ("Q", "q"):
                keep_editing_condition = False

            print("Here is the dictionary you have generated: ")
            print(packets)

            # Continue Editing check
            keep_editing = input("Would like you to keep editing? Y/N: ")
            if keep_editing in ("y", "Y"):
                keep_editing_condition = True
            else:
                keep_editing_condition = False


def main():
    """Function used by instrument to generate XTCE. Change values where TODO is."""

    # TODO: change instrument name (DONE)
    instrument_name = input("Enter instrument name: ")
    # Input validation
    if not instrument_name.islower():
        instrument_name = instrument_name.lower()
    while instrument_name not in [
        "swe",
        "codice",
        "glows",
        "hi",
        "hit",
        "lo",
        "mag",
        "swapi",
        "ultra",
        "idex",
    ]:
        instrument_name = input("Invalid instrument name. Enter again: ")
        if not instrument_name.islower():
            instrument_name = instrument_name.lower()

    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"

    # TODO: Copy packet definition to tools/xtce_generation/ folder
    excel_file_name = input("Enter the excel file name to write to: ")
    path_to_excel_file = f"{current_directory}/" + excel_file_name
    # TODO: Add check if file doesn't open / doesn't exist.

    # TODO: update packets dictionary with packet name and appId
    # Eg.
    # packets = {
    #     "P_COD_HI_PHA": 1169,
    #     "P_COD_LO_PHA": 1153,
    #     "P_COD_LO_NSW_SPECIES_COUNTS": 1157,
    #     "P_COD_HI_OMNI_SPECIES_COUNTS": 1172,
    # }

    # Creating packets dictionary
    # packets = {"<packet_name>": "<app_id_number>"}
    packets = dict()
    loop = True
    print("Generate packets dictionary. Press Q to exit: ")
    while loop:
        packet_name_input = input("Enter instrument name: ")
        # If user exits:
        if packet_name_input in ("q", "Q"):
            break

        app_id_num_input = input("Enter app id: ")
        # If user exits
        if app_id_num_input in ("q", "Q"):
            break
        # Setting dictionary item
        packets.update({packet_name_input: app_id_num_input})

    print("Here is the dictionary you have generated: ")
    print(packets)

    # Changing/editing packets dictionary
    editing_packets_dictionary(packets)

    print("Here is the dictionary you have generated: ")
    print(packets)

    for packet_name, app_id in packets.items():
        print(packet_name)
        telemetry_generator = TelemetryGenerator(
            packet_name=packet_name, path_to_excel_file=path_to_excel_file, apid=app_id
        )
        telemetry_generator.generate_telemetry_xml(
            f"{packet_definition_path}/{packet_name}.xml", packet_name
        )


if __name__ == "__main__":
    main()
