from pathlib import Path

import cdflib
import numpy as np
import pandas as pd


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

    # Read the CSV file (assuming comma separation)
    df = pd.read_csv(csv_path, delimiter=",", skipinitialspace=True)

    # Ensure all data is float32 to prevent integer storage
    df = df.astype(np.float32)

    # Extract data columns as float32
    right_ascension = df["Right Ascension (deg)"].astype(np.float32).to_numpy()
    declination = df["Declination (deg)"].astype(np.float32).to_numpy()
    response = df["Response (cm2-sr)"].astype(np.float32).to_numpy()

    # Define output CDF filename
    cdf_filename = output_dir / "ultra_90_dps_gf.cdf"

    # Create a new CDF file
    cdf = cdflib.cdfwrite.CDF(str(cdf_filename))

    # Define variable specifications dynamically
    var_specs = [
        {
            "Variable": "Right_Ascension",
            "Data_Type": 21,  # ✅ CDF_REAL4 (float32) instead of 41
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [],
            "PadValue": np.nan,  # ✅ Ensures correct float storage
            "Compression": {
                "Compression_Type": "GZIP_COMPRESSION",
                "Compression_Level": 9,
            },
        },
        {
            "Variable": "Declination",
            "Data_Type": 21,  # ✅ CDF_REAL4 (float32) instead of 41
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [],
            "PadValue": np.nan,  # ✅ Ensures correct float storage
            "Compression": {
                "Compression_Type": "GZIP_COMPRESSION",
                "Compression_Level": 9,
            },
        },
        {
            "Variable": "Response",
            "Data_Type": 21,  # ✅ CDF_REAL4 (float32) instead of 41
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [],
            "PadValue": np.nan,  # ✅ Ensures correct float storage
            "Compression": {
                "Compression_Type": "GZIP_COMPRESSION",
                "Compression_Level": 9,
            },
        },
    ]

    # Write variables to the CDF file
    for var_spec, var_data in zip(var_specs, [right_ascension, declination, response]):
        cdf.write_var(var_spec, var_data=var_data)

    # Close CDF file
    cdf.close()

    print(f"✅ Saved CDF: {cdf_filename}")
    return cdf_filename


# Define input/output paths
csv_path = "/Users/lasa6858/Downloads/ultra_90_dps_gf.csv"
output_dir = "/Users/lasa6858/Downloads/cdf_output"

# Convert CSV to CDF
cdf_file = convert_gf_csv_to_cdf(csv_path, output_dir)

print("\nGenerated CDF file:", cdf_file)
