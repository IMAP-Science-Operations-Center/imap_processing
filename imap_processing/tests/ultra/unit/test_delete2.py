import pandas as pd
import numpy as np
import cdflib
from pathlib import Path


def convert_gf_csv_to_cdf(csv_path, output_dir):
    """
    Converts a CSV file (ultra_90_dps_gf.csv) to a GZIP-compressed CDF file.

    Parameters:
        csv_path (str or Path): Path to the input CSV file.
        output_dir (str or Path): Directory to save the output CDF file.

    Returns:
        Path to the generated CDF file.
    """

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the CSV file (assuming tab or comma separation)
    df = pd.read_csv(csv_path, delimiter=",", skipinitialspace=True)

    # Extract data columns
    right_ascension = df["Right Ascension (deg)"].to_numpy()
    declination = df["Declination (deg)"].to_numpy()
    response = df["Response (cm2-sr)"].to_numpy()

    # Define output CDF filename
    cdf_filename = output_dir / "ultra_90_dps_gf.cdf"

    # Create a new CDF file
    cdf = cdflib.cdfwrite.CDF(str(cdf_filename))

    # Define variable specifications dynamically
    var_specs = [
        {
            "Variable": "Right_Ascension",
            "Data_Type": 41,  # CDF_FLOAT (32-bit float)
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [],
            "Compression": {"Compression_Type": "GZIP_COMPRESSION", "Compression_Level": 9},
        },
        {
            "Variable": "Declination",
            "Data_Type": 41,  # CDF_FLOAT (32-bit float)
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [],
            "Compression": {"Compression_Type": "GZIP_COMPRESSION", "Compression_Level": 9},
        },
        {
            "Variable": "Response",
            "Data_Type": 41,  # CDF_FLOAT (32-bit float)
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [],
            "Compression": {"Compression_Type": "GZIP_COMPRESSION", "Compression_Level": 9},
        },
    ]

    # Write variables to the CDF file
    for var_spec, var_data in zip(var_specs, [right_ascension, declination, response]):
        cdf.write_var(var_spec, var_data=var_data)

    # Close CDF file
    cdf.close()

    print(f"Saved CDF: {cdf_filename}")
    return cdf_filename


# Define input/output paths
csv_path = "/Users/lasa6858/Downloads/ultra_90_dps_gf.csv"
output_dir = "/Users/lasa6858/Downloads/cdf_output"

# Convert CSV to CDF
cdf_file = convert_gf_csv_to_cdf(csv_path, output_dir)

print("\nGenerated CDF file:", cdf_file)
