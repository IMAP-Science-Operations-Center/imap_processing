"""XTCE generator script."""

import argparse
from pathlib import Path

import imap_data_access

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


# Function to parse command line arguments
def _parse_args():
    """
    Parse the command line arguments.

    The expected input format is:
    --instrument "instrument_name"
    --filename "file_name"
    --packets [{"packet_name_1": app_id_1, "packet_name_2": app_id_2}]

    Returns
    -------
    args : argparse.Namespace
        An object containing the parsed arguments and their values.
    """
    description = (
        "This command line program generates an instrument specific XTCE file "
        "Example usage: "
        '--instrument "swapi"'
        '--filename "TLM_SWP_20231006-121021.xlsx"'
        '--packets "['
        '   {"P_SWP_HK": 1184,'
        '    "P_SWP_SCI": 1188,'
        '   "P_SWP_AUT": 1192,'
        '}]"'
    )
    instrument_help = (
        "The instrument to process. Acceptable values are: "
        f"{imap_data_access.VALID_INSTRUMENTS}"
    )
    filename_help = (
        # TODO: what help should I provide?
    )
    packets_help = (
        # TODO: Overall packets_help statement
        "Example usage: "
        '--instrument "swapi"'
        '--filename "TLM_SWP_20231006-121021.xlsx"'
        '--packets "['
        '   {"P_SWP_HK": 1184,'
        '    "P_SWP_SCI": 1188,'
        '   "P_SWP_AUT": 1192,'
        '}]"'
    )

    parser = argparse.ArgumentParser(prog="imap_xtce", description=description)
    parser.add_argument("--instrument", type=str, required=True, help=instrument_help)
    parser.add_argument("--filename", type=str, required=True, help=filename_help)
    parser.add_argument("--packets", type=str, required=True, help=packets_help)

    args = parser.parse_args()

    return args


def _validate_args(args):
    """
    Ensure that the  arguments are valid before kicking off the processing.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the parsed arguments and their values.
    """
    if args.instrument not in imap_data_access.VALID_INSTRUMENTS:
        raise ValueError(
            f"{args.instrument} is not in the supported instrument list: "
            f"{imap_data_access.VALID_INSTRUMENTS}"
        )


def main():
    """Docstring."""
    # Command line arguments form:
    # --instrument
    # "instrument_name"
    # --file - name
    # "file_name"
    # --packets[{"packet_name_1": app_id_1, "packet_name_2": app_id_2}].

    args = _parse_args()
    _validate_args(args)

    # TODO: change instrument name
    instrument_name = args.instrument
    current_directory = Path(__file__).parent
    module_path = f"{current_directory}/../../imap_processing"
    packet_definition_path = f"{module_path}/{instrument_name}/packet_definitions"
    # TODO: Copy packet definition to tools/xtce_generation/ folder
    path_to_excel_file = f"{current_directory}/{args.filename}"

    # TODO: update packets dictionary with packet name and appId
    # Eg.
    # packets = {
    #     "P_COD_HI_PHA": 1169,
    #     "P_COD_LO_PHA": 1153,
    #     "P_COD_LO_NSW_SPECIES_COUNTS": 1157,
    #     "P_COD_HI_OMNI_SPECIES_COUNTS": 1172,
    # }

    packets = {args.packets}

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
