"""XTCE generator script that uses CLI for generalized format."""

import argparse
import json
from pathlib import Path

import imap_data_access

from tools.xtce_generation.telemetry_generator import TelemetryGenerator


# Function to parse command line arguments
def _parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    The expected input format is:
    --instrument "instrument_name"
    --file-path "full_file_path"
    --packets '{"packet_name_1": app_id_1, "packet_name_2": app_id_2}'

    Returns
    -------
    args : argparse.Namespace
        An object containing the parsed arguments and their values.
    """
    description = (
        "This command line program generates an instrument specific XTCE file."
        "Example usage: "
        '--instrument "swapi"'
        "--file-path "
        '"/Users/anma6676/Desktop/Repositories/imap_processing/tools/xtce_generation/TLM_SWP_20231006-121021.xlsx"'
        "--packets '"
        '{"P_SWP_HK": 1184, '
        '"P_SWP_SCI": 1188, '
        '"P_SWP_AUT": 1192, '
        "}'"
    )
    instrument_help = (
        "The instrument to process. Acceptable values are: "
        f"{imap_data_access.VALID_INSTRUMENTS}"
    )
    file_path_help = "Provide full file path to write packets to."
    packets_help = (
        "Provide packet dictionary using packet_name, and app_id."
        '{"<packet_name>": <app_id>}'
    )

    parser = argparse.ArgumentParser(prog="imap_xtce", description=description)
    parser.add_argument("--instrument", type=str, required=True, help=instrument_help)
    parser.add_argument("--file-path", type=str, required=True, help=file_path_help)
    parser.add_argument("--packets", type=str, required=True, help=packets_help)

    args = parser.parse_args()

    return args


def _validate_args(args: argparse.Namespace) -> None:
    """
    Ensure that the instrument is valid.

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

    directory = Path(args.file_path)
    if not directory.exists():
        raise FileNotFoundError(f"{args.file_path} not found, and may not exist.")


def main() -> None:
    """
    Generate xtce file from CLI information given.

    The xtce file will be written in an instrument specific subfolder.
    """

    # Parse arguments, and validate instrument
    args = _parse_args()
    _validate_args(args)

    instrument_name = args.instrument
    current_directory = Path(__file__).parent.parent.parent
    module_path = current_directory / "imap_processing"
    packet_definition_path = module_path / instrument_name / "packet_definitions"
    path_to_excel_file = args.file_path

    # Update packets dictionary with given CLI information
    packets = json.loads(args.packets)

    for packet_name, app_id in packets.items():
        print(packet_name)
        telemetry_generator = TelemetryGenerator(
            packet_name=packet_name,
            path_to_excel_file=path_to_excel_file,
            apid=int(app_id),
        )
        telemetry_generator.generate_telemetry_xml(
            packet_definition_path / f"{packet_name}.xml", packet_name
        )


if __name__ == "__main__":
    main()
