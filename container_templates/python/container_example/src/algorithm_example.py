"""
Container example for Level 2 developers

Authors: Laura Sandoval, Gavin Medley, Matt Watwood
"""
# Standard
import argparse
from hashlib import md5
import logging
import os
import sys
# Installed
import h5py as h5
import numpy as np
import pandas as pd
# Local
from libera_utils.io.smart_open import smart_open
from libera_utils.io.manifest import Manifest, ManifestType
from static.l2_quality_flags import L2QualityFlag


logger = logging.getLogger(__name__)


def main():
    # Initialize logging. You can configure logging however you like
    logging.basicConfig(level="DEBUG")

    # Get logger instance, named for the current function for traceability
    logger.debug(f"Command executed in container: {sys.argv}")
    logger.info("Parsing CLI arguments.")
    args = parse_cli_args(sys.argv[1:])

    manifest = args.manifest
    logger.info(f"Manifest file to read: {manifest}")

    logger.info("Additional options passed are "
                f"exampleint={args.exampleint}, examplefloat={args.examplefloat}, examplebool={args.examplebool}")

    processing_dropbox = os.environ['PROCESSING_DROPBOX']

    # read json information
    manifest = Manifest.from_file(manifest)
    if not manifest.manifest_type == ManifestType.INPUT:
        raise ValueError("Incorrect manifest type received as input.")

    logger.debug(f"Manifest type is {manifest.manifest_type}")
    logger.debug(f"Manifest contains files:\n{manifest.files}")

    # read hdf5
    for record in manifest.files:
        checksum = record['checksum']
        filename = record['filename']
        # Validate checksums
        with smart_open(filename, 'rb') as fh:
            if checksum != md5(fh.read()).hexdigest():
                raise ValueError("Checksums do not match!")
            logger.debug(f"Checksum matches for {filename}")

        with h5.File(smart_open(filename, 'rb'), 'r') as h5file:
            # Get the data from each file and put data into some format that will be used in the algorithm
            data_in = np.array(h5file['HDFEOS/SWATHS/Swath1/DataField/Temperature'])
            logger.info(f"Found input data in HDF5 file:\n{data_in}")

    df = generate_example_data()

    # write example output data to a new HDF5 file
    output_files = []
    output_filepath = os.path.join(processing_dropbox, 'example_output.h5')
    logger.info(f"Writing output file: {output_filepath}")
    with h5.File(smart_open(output_filepath, 'xb'), 'x') as hdf:
        hdf.create_group('new_group')
        hdf.attrs['someattr'] = "hello, world"
        hdf.create_dataset('data/array1', data=df.data_out)
        hdf.create_dataset('quality_out/array1', data=df.quality_int)

    # get the checksum of the written file
    with smart_open(output_filepath, 'rb') as fh:
        checksum = md5(fh.read()).hexdigest()

    output_files.append({"filename": output_filepath, "checksum": checksum})

    # Write output manifest file containing a list of the product files that the processing created
    output_manifest_path = os.path.join(processing_dropbox, "libera_output_manifest_20220923t111111.json")
    logger.info(f"Writing output manifest: {output_filepath}")
    output_manifest = Manifest(manifest_type=ManifestType.OUTPUT,
                               filename=output_manifest_path,
                               files=output_files,
                               configuration={})
    output_manifest.write(output_manifest_path)
    logger.info("Algorithm complete. Exiting.")


def generate_example_data():
    """
    Function created to make up data and quality flags

    Returns
    -------
    pandas.core.frame.DataFrame
        Dataframe containing data and quality flag columns
    """

    # generate fake data product to write as output
    df = pd.DataFrame()
    df['data_out'] = [-np.inf, -np.inf, 30, 40, 50, 60]
    df['quality'] = np.full(len(df.data_out), fill_value=L2QualityFlag.NONE)
    df['quality_int'] = np.full(len(df.data_out), fill_value=0)

    # flag neg infinite values
    was_inf = np.equal(df.data_out, -np.inf)
    df.loc[was_inf, 'quality'] = df.loc[was_inf, 'quality'][0] | L2QualityFlag.INF | L2QualityFlag.NEG
    df.loc[was_inf, 'quality_int'] = (df.loc[was_inf, 'quality'][0] | L2QualityFlag.INF | L2QualityFlag.NEG).value

    # view decomposed flag and value
    view_summary = df.quality.values[0].summary
    view_decomposed = (df.quality.values[0]).decompose()
    print(view_summary)
    print(view_decomposed)

    return df


def parse_cli_args(cli_args: list):
    """
    Function that parses CLI arguments

    Parameters
    ----------
    cli_args : list
        List of string arguments to parse

    Returns
    -------
    Namespace
        A Namespace object containing the parsed arguments as attributes
    """
    parser = argparse.ArgumentParser(description='program arguments.')

    parser.add_argument('--exampleint',
                        type=int,
                        default=42,
                        help='An example integer input option'
                        )

    parser.add_argument('--examplefloat',
                        type=float,
                        default=3.14159,
                        help='An example float input option')

    parser.add_argument('--examplebool',
                        type=bool,
                        default=True,
                        help='An example boolean input option')

    parser.add_argument('manifest',
                        type=str,
                        help="Input JSON manifest file containing a list of files to use in the processing algorithm.")

    return parser.parse_args(cli_args)


if __name__ == "__main__":
    main()