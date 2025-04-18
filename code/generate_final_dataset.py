import argparse
import json
from collections import defaultdict
from pathlib import Path
from shutil import rmtree

import loguru
import pandas as pd
from tqdm import tqdm


def load_json(file_path: Path):
    """Loads JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        loguru.logger.error(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        loguru.logger.error(f"Error: Could not decode JSON from {file_path}")
        return None


def load_text_list(file_path: Path) -> set:
    """Loads a list of strings from a text file (one item per line)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Read lines, strip whitespace, filter out empty lines
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        loguru.logger.error(f"Error: File not found at {file_path}")
        return set()


def format_prompt(
    recipe_title: str, instructions_dict: dict, question_text: str
) -> str:
    """Formats the prompt string including goal, steps, and question."""
    prompt_lines = []
    prompt_lines.append(f"Goal: {recipe_title}")
    prompt_lines.append("\nSteps:")
    # Ensure steps are sorted numerically even if keys are strings
    try:
        sorted_steps = sorted(instructions_dict.items(), key=lambda item: int(item[0]))
    except ValueError:
        # Fallback if keys are not numeric strings
        sorted_steps = sorted(instructions_dict.items())

    for step_num, step_text in sorted_steps:
        prompt_lines.append(f"Step {step_num}: {step_text}")
    prompt_lines.append("\nQuestion:")
    prompt_lines.append(question_text)
    return "\n".join(prompt_lines)


def sanitize_filename(name: str) -> str:
    """Removes or replaces characters unsuitable for filenames."""
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove characters that are problematic in filenames
    name = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
    # Ensure it's not empty
    if not name:
        name = "default"
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Generate final CSV datasets for each analysis category."
    )
    parser.add_argument(
        "processed_recipes_folder",
        help="Folder containing the processed recipe JSON files from the *original* run (e.g., outs/outs_6/).",
    )
    parser.add_argument(
        "curated_entity_list",
        help="Path to the final, manually curated entity list file (e.g., outs/outs_6_sub/final_entity_subset.txt).",
    )
    parser.add_argument(
        "output_csv_folder",
        help="Folder where the final CSV files will be saved.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=100,
        help="Target number of questions per analysis category CSV.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level.",
    )

    args = parser.parse_args()

    # *** Setup ***
    processed_recipes_dir = Path(args.processed_recipes_folder)
    curated_entity_list_path = Path(args.curated_entity_list)
    output_csv_dir = Path(args.output_csv_folder)
    target_count = args.target_count

    # Delete output folder if it exists
    if output_csv_dir.exists():
        rmtree(output_csv_dir)
        loguru.logger.info(f"Deleting existing output folder: {output_csv_dir}")

    loguru.logger.info(f"Creating new output folder: {output_csv_dir}")
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    loguru.logger.remove()
    log_file_path = output_csv_dir / "csv_generation.log"
    loguru.logger.add(log_file_path, level=args.log_level)
    loguru.logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")

    # *** Load Inputs ***
    loguru.logger.info(f"Loading curated entity list from: {curated_entity_list_path}")
    curated_entities = load_text_list(curated_entity_list_path)
    if not curated_entities:
        loguru.logger.error(
            "Curated entity list is empty or could not be loaded. Exiting."
        )
        return
    loguru.logger.info(f"Loaded {len(curated_entities)} curated entities.")

    # *** Filter Recipes and Collect Questions ***
    loguru.logger.info(f"Scanning processed recipes in: {processed_recipes_dir}")
    all_eligible_questions = []
    recipe_instructions_cache = {}  # Cache instructions to avoid reloading later
    recipe_files = list(processed_recipes_dir.glob("*.json"))

    # Filter out known non-recipe JSONs from the original output
    recipe_files = [
        f
        for f in recipe_files
        if f.name
        not in {
            "all_entities.json",
            "all_questions.json",
            "question_counts.json",
            "processing_stats.json",
        }
    ]

    if not recipe_files:
        loguru.logger.error(
            f"No processed recipe JSON files found in {processed_recipes_dir}. Exiting."
        )
        return

    loguru.logger.info(f"Found {len(recipe_files)} potential recipe files to process.")

    relevant_recipe_count = 0
    for recipe_file in tqdm(recipe_files, desc="Filtering Recipes & Questions"):
        recipe_data = load_json(recipe_file)
        if not recipe_data:
            loguru.logger.warning(f"Could not load or parse {recipe_file}, skipping.")
            continue

        title = recipe_data.get("title")
        recipe_entities = set(recipe_data.get("entities", {}).keys())
        questions = recipe_data.get("questions", [])
        instructions = recipe_data.get("instructions", {})

        if not title or not questions or not instructions:
            loguru.logger.warning(
                f"Missing title, questions, or instructions in {recipe_file}, skipping."
            )
            continue

        # Check for overlap with curated entities
        if curated_entities.intersection(recipe_entities):
            relevant_recipe_count += 1
            # Store instructions for later prompt generation
            recipe_instructions_cache[title] = instructions
            # Add questions from this relevant recipe
            for q_data in questions:
                q_type = q_data.get("type")
                question_text = q_data.get("question")
                answer = q_data.get("answer")
                if q_type and question_text and answer is not None:
                    all_eligible_questions.append(
                        {
                            "recipe_title": title,
                            "type": q_type,
                            "question": question_text,
                            "answer": answer,
                        }
                    )
        # else:
        # loguru.logger.debug(f"Recipe '{title}' skipped (no overlap with curated entities). Entities: {recipe_entities}")

    loguru.logger.info(
        f"Found {relevant_recipe_count} relevant recipes containing curated entities."
    )
    loguru.logger.info(
        f"Collected {len(all_eligible_questions)} eligible questions from these recipes."
    )

    if not all_eligible_questions:
        loguru.logger.error(
            "No eligible questions found after filtering. Cannot generate CSVs. Exiting."
        )
        return

    # *** Score Recipes ***
    loguru.logger.info("Scoring recipes based on question type diversity...")
    questions_by_recipe = defaultdict(lambda: defaultdict(list))
    for q_data in all_eligible_questions:
        questions_by_recipe[q_data["recipe_title"]][q_data["type"]].append(q_data)

    recipe_scores = defaultdict(int)
    for title, types_dict in questions_by_recipe.items():
        recipe_scores[title] = len(types_dict)  # Score is the count of unique types

    loguru.logger.info("Finished scoring recipes.")

    # *** Select Final Questions ***
    loguru.logger.info(
        f"Selecting top {target_count} questions per category, prioritizing diverse recipes..."
    )
    final_questions_by_type = defaultdict(list)
    all_question_types = sorted(list(set(q["type"] for q in all_eligible_questions)))

    for q_type in tqdm(all_question_types, desc="Selecting Questions per Type"):
        # Get all questions of this type
        type_questions = [q for q in all_eligible_questions if q["type"] == q_type]

        # Sort them: 1. Recipe Score (desc), 2. Recipe Title (asc), 3. Question Text (asc)
        sorted_type_questions = sorted(
            type_questions,
            key=lambda q: (
                recipe_scores.get(q["recipe_title"], 0),  # Score (higher is better)
                q["recipe_title"],  # Title (for tie-breaking)
                q["question"],  # Question text (for final tie-breaking)
            ),
            reverse=True,  # Reverse only for score (higher first)
        )

        # Select top N
        selected_questions = sorted_type_questions[:target_count]
        final_questions_by_type[q_type] = selected_questions

        if len(selected_questions) < target_count:
            loguru.logger.warning(
                f"Found only {len(selected_questions)} questions for type '{q_type}', which is less than the target {target_count}."
            )
        else:
            loguru.logger.debug(
                f"Selected {len(selected_questions)} questions for type '{q_type}'."
            )

    # *** Format and Write CSVs ***
    loguru.logger.info("Formatting prompts and writing CSV files...")
    num_csv_written = 0
    for q_type, questions in tqdm(final_questions_by_type.items(), desc="Writing CSVs"):
        if not questions:
            loguru.logger.warning(
                f"No questions selected for type '{q_type}', skipping CSV generation."
            )
            continue

        output_data = []
        for q_data in questions:
            recipe_title = q_data["recipe_title"]
            question_text = q_data["question"]
            answer = q_data["answer"]

            # Retrieve cached instructions
            instructions = recipe_instructions_cache.get(recipe_title)
            if not instructions:
                loguru.logger.error(
                    f"Could not find cached instructions for recipe '{recipe_title}'. Skipping question: {question_text}"
                )
                continue

            # Format the prompt
            prompt = format_prompt(recipe_title, instructions, question_text)

            output_data.append({"prompt": prompt, "answer": answer})

        # Create DataFrame
        df = pd.DataFrame(output_data)

        # Save to CSV
        csv_filename = sanitize_filename(q_type) + "_questions.csv"
        csv_path = output_csv_dir / csv_filename
        try:
            df.to_csv(csv_path, index=False, encoding="utf-8")
            loguru.logger.debug(f"Successfully wrote {len(df)} rows to {csv_path}")
            num_csv_written += 1
        except Exception as e:
            loguru.logger.error(f"Failed to write CSV file {csv_path}: {e}")

    loguru.logger.success(f"Finished writing {num_csv_written} CSV files to: {output_csv_dir}")


if __name__ == "__main__":
    main()
