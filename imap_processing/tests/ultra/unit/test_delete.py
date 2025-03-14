from pathlib import Path

import cdflib
import numpy as np
import pandas as pd


def split_csv_to_cdf(csv_path, output_dir, energy_splits):
    """
    Splits a large CSV file into multiple CDFs based on energy bin groups.

    Parameters:
        csv_path (str or Path): Path to the input CSV file.
        output_dir (str or Path): Directory to save the output CDF files.
        energy_splits (list of tuples): Ranges of energy bins for each split.

    Returns:
        List of generated CDF file paths.
    """

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the CSV (assuming comma separation, adjust if needed)
    df = pd.read_csv(csv_path, delimiter=",", skipinitialspace=True)

    # Ensure all data is float32 to prevent integer storage
    df = df.astype(np.float32)

    # Extract spatial columns (ensure float32)
    right_ascension = df["Right Ascension (deg)"].astype(np.float32).to_numpy()
    declination = df["Declination (deg)"].astype(np.float32).to_numpy()

    # Extract energy bin headers
    energy_bins = df.columns[2:]  # All columns after Right Ascension & Declination

    # Print Data Types for Debugging
    print("🔍 Data Types Before Writing to CDF:")
    print(df.dtypes)

    # Create a list to store generated CDF file paths
    cdf_files = []

    # Split the data into groups based on energy ranges
    for idx, (start_keV, end_keV) in enumerate(energy_splits):
        # Select columns within the energy range
        selected_bins = [
            col
            for col in energy_bins
            if start_keV <= float(col.replace("keV", "")) <= end_keV
        ]

        # Ensure energy data is float32
        selected_data = df[selected_bins].astype(np.float32).to_numpy()

        # Define output CDF filename
        cdf_filename = output_dir / f"efficiencies_{start_keV}-{end_keV}keV.cdf"

        # Create a new CDF file
        cdf = cdflib.cdfwrite.CDF(str(cdf_filename))

        # Dynamically generate variable specifications for all variables
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
        ]

        # Add energy bin variables dynamically
        for energy_bin in selected_bins:
            var_specs.append(
                {
                    "Variable": energy_bin,
                    "Data_Type": 21,  # ✅ CDF_REAL4 (float32) instead of 41
                    "Num_Elements": 1,
                    "Rec_Vary": True,
                    "Dim_Sizes": [],
                    "PadValue": np.nan,  # ✅ Ensures correct float storage
                    "Compression": {
                        "Compression_Type": "GZIP_COMPRESSION",
                        "Compression_Level": 9,
                    },
                }
            )

        # Write Right Ascension & Declination
        for var_spec, var_data in zip(var_specs[:2], [right_ascension, declination]):
            cdf.write_var(var_spec, var_data=var_data)

        # Write energy bins
        for var_spec, var_data in zip(
            var_specs[2:], selected_data.T
        ):  # Transpose for correct column-wise storage
            cdf.write_var(var_spec, var_data=var_data)

        # Close CDF file
        cdf.close()

        print(f"✅ Saved CDF: {cdf_filename}")
        cdf_files.append(cdf_filename)

    return cdf_files


# Define input/output paths
csv_path = "/Users/lasa6858/imap_processing/imap_processing/ultra/lookup_tables/Ultra_90_DPS_efficiencies_all.csv"
output_dir = "/Users/lasa6858/imap_processing/imap_processing/ultra/lookup_tables"

# Define energy bin splits (adjust as needed)
energy_splits = [(3.0, 20.0), (20.5, 50.0), (50.5, 80.0)]  # Three CDFs

# Run the splitting process
cdf_files = split_csv_to_cdf(csv_path, output_dir, energy_splits)

print("\nGenerated CDF files:", cdf_files)
