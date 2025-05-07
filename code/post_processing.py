import argparse
import glob
import json
import os
import re
from shutil import rmtree

import pandas as pd
from helpers import constants


# Helper function to parse goal and step count from prompt text
def parse_prompt_for_stats(prompt_text: str) -> tuple[str | None, int]:
    goal = None
    num_steps = 0
    lines = prompt_text.splitlines()
    for line in lines:
        if line.startswith("Goal: "):
            goal = line.replace("Goal: ", "").strip()
        elif re.match(r"Step \d+: ", line):
            num_steps += 1
    return goal, num_steps


def convert_csv(input_folder, output_folder):
    """
    Converts True/False values in 'answer' column of CSV files to Yes/No,
    and generates statistics for the final dataset.

    Args:
        input_folder (str): Path to the folder containing CSV files
        output_folder (str): Path to save converted files and statistics.
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

    # Initialize statistics
    stats = {
        "total_procedures": 0,
        "avg_steps_per_procedure": 0.0,
        "min_steps": 0,
        "max_steps": 0,
        "total_qa_pairs_generated": 0,  # Total QA pairs in the output files
        "qa_pairs_used": 0,  # Same as total_qa_pairs_generated for this script
    }
    # Using a dictionary to store step counts for unique procedures: { "Recipe Title": num_steps }
    procedure_details = {}

    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        output_file_name = f"recipe_{file_name.replace('_questions', '').lower()}"
        output_file_path = os.path.join(output_folder, output_file_name)

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            if "prompt" not in df.columns:
                print(
                    f"File '{file_name}' does not have a 'prompt' column. Skipping this file."
                )
                continue

            # Accumulate procedure details from prompts in this file
            current_file_qa_count = len(df)
            for index, row in df.iterrows():
                prompt_text = row["prompt"]
                goal, num_steps_for_procedure = parse_prompt_for_stats(prompt_text)
                if goal:
                    # This will store the number of steps for each unique goal.
                    # If a goal appears multiple times (e.g. from different QAs for the same recipe),
                    # it will just overwrite with the same step count, which is fine.
                    procedure_details[goal] = num_steps_for_procedure

            # Process 'answer' column if it exists
            if "answer" in df.columns:
                df["answer"] = df["answer"].replace(constants.BOOLEAN_TO_YES_NO_MAP)
                # Optional: Check if the `answer` column has only Yes/No values after conversion
                if (
                    not df["answer"].isin(["Yes", "No", "yes", "no"]).all()
                ):  # Added lowercase for robustness
                    # Further filter: check for non-string answers that were not True/False
                    # This check is more about unexpected values that weren't boolean originally.
                    # For this example, we assume original boolean or string "30-35 minutes" etc.
                    # If the original values are mixed, this check might be complex.
                    # The problem implies boolean answers are the primary target for Yes/No.
                    pass  # Simple pass, or more detailed warning if needed.

                print(f"File '{file_name}': 'answer' column processed.")
            else:
                print(
                    f"File '{file_name}' does not have an 'answer' column. Answer conversion skipped."
                )

            # Write the modified DataFrame to the output file
            df.to_csv(output_file_path, index=False)
            print(f"File '{file_name}' processed and saved to '{output_file_path}'.")

            # Add this file's QA count to the total, as it's part of the output dataset
            stats["total_qa_pairs_generated"] += current_file_qa_count

        except Exception as e:
            print(
                f"Error processing '{file_name}': {str(e)}. This file will be skipped."
            )

    # Calculate derived statistics after processing all files
    if procedure_details:
        stats["total_procedures"] = len(procedure_details)
        all_step_counts = list(procedure_details.values())
        if all_step_counts:  # Ensure list is not empty
            stats["avg_steps_per_procedure"] = round(
                sum(all_step_counts) / len(all_step_counts), 2
            )
            stats["min_steps"] = min(all_step_counts)
            stats["max_steps"] = max(all_step_counts)
        # If all_step_counts is empty (e.g. all goals were None), these will remain 0

    stats["qa_pairs_used"] = stats["total_qa_pairs_generated"]

    # Print statistics
    print("\n--- Dataset Statistics ---")
    # Custom titles for better readability
    for key, value in stats.items():
        title = constants.POST_PROCESSING_STAT_TITLES.get(
            key, key.replace("_", " ").title()
        )
        print(f"{title}: {value}")

    # Save statistics to a JSON file
    stats_file_path = os.path.join(output_folder, "dataset_statistics.json")
    try:
        with open(stats_file_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
        print(f"Statistics saved to {stats_file_path}")
    except Exception as e:
        print(f"Error saving statistics to JSON: {str(e)}")

    print(
        f"\nConversion and statistics generation complete. Processed {len(csv_files)} CSV files."
    )
    print(f"Output files and statistics are in: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert True/False values in CSV files to Yes/No and generate dataset statistics."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder containing CSV files to process.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to save converted files and dataset_statistics.json.",
    )

    args = parser.parse_args()

    convert_csv(args.input_folder, args.output_folder)
