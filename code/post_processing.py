import argparse
import glob
import os
from shutil import rmtree

import pandas as pd


def convert_csv(input_folder, output_folder):
    """
    Converts all True/False values in the 'answer' column of CSV files to Yes/No using pandas.

    Args:
        input_folder (str): Path to the folder containing CSV files
        output_folder (str, optional): Path to save converted files. If None, files will be overwritten in the input folder.
    """
    # Delete output folder if it exists
    if os.path.exists(output_folder):
        rmtree(output_folder)
        print(f"Deleting existing output folder: {output_folder}")

    print(f"Creating new output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    # Get all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    # Check if any CSV files were found
    if not csv_files:
        print(f"No CSV files found in '{input_folder}'")
        return

    # Process each CSV file
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        output_file = (
            os.path.join(output_folder, f"recipe_{file_name}")
            .replace("_questions", "")
            .lower()
        )

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Check if 'answer' column exists
            if "answer" not in df.columns:
                print(f"File '{file_name}' does not have an 'answer' column. Skipping.")
                continue

            # Replace True/False with Yes/No in the 'answer' column
            df["answer"] = df["answer"].replace({True: "Yes", False: "No"})

            # Write the modified data to the output file
            df.to_csv(output_file, index=False)

            # Check if the `answer` column has only Yes/No values
            if not df["answer"].isin(["Yes", "No"]).all():
                print(
                    f"Warning: '{file_name}' contains values other than Yes/No in 'answer' column."
                )
            else:
                print(
                    f"File '{file_name}' processed successfully. Converted True/False to Yes/No."
                )

        except Exception as e:
            print(f"Error processing '{file_name}': {str(e)}")

    print(
        f"Conversion complete. Processed {len(csv_files)} CSV files. Written to {output_folder}"
    )


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert True/False values in CSV files to Yes/No"
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing CSV files"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to save converted files.",
    )

    args = parser.parse_args()

    convert_csv(
        args.input_folder,
        args.output_folder
        if args.output_folder
        else exit("No output folder specified"),
    )
