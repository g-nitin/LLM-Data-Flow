import argparse
import json
import re
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
            # Read lines, strip whitespace, filter out empty lines, convert to lowercase
            return set(line.strip().lower() for line in f if line.strip())
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


def check_question_relevance(
    question_text: str, q_type: str, curated_entities_lower: set
) -> bool:
    """
    Checks if the primary entity discussed in the question matches a curated entity.
    Uses regex patterns tailored to specific question types.

    Args:
        question_text: The text of the question.
        q_type: The analysis type of the question.
        curated_entities_lower: A set of curated entity names, already lowercased.

    Returns:
        True if the question is relevant (either type doesn't need check or primary entity matches), False otherwise.
    """
    # Types that don't require a specific entity match in the question text
    types_to_skip_entity_check = {
        "Interval Analysis",
        "Concurrency Analysis",
    }
    if q_type in types_to_skip_entity_check:
        loguru.logger.trace(
            f"Skipping entity check for type '{q_type}': {question_text}"
        )
        return True

    # Define regex patterns to extract the primary entity phrase for each relevant type
    # The patterns capture the group assumed to be the (potentially malformed) entity name
    patterns = {
        "Reaching Definitions": r"In Step \d+, is the (.*?) from Step \d+ being used\?",
        "Very Busy Expressions": r"Is (.*?) from Step \d+ used in multiple future steps.*?",
        "Available Expressions": r"Is (.*?) from Step \d+ still available in Step \d+\?",
        "Live Variable Analysis": r"Is (.*?) live after Step \d+\?",
        "Type-State Analysis": r"If we skip Step \d+, is it still valid to .+ the (.*?) in Step \d+\?",
        "Taint Analysis": r"Does using (.*?) in Step \d+ introduce.*?",
    }

    pattern = patterns.get(q_type)
    if not pattern:
        loguru.logger.warning(
            f"No specific regex pattern defined for question type '{q_type}'. Cannot perform strict entity check for: {question_text}"
        )
        # Fallback: Check if *any* curated entity is mentioned (original less strict check)
        question_lower = question_text.lower()
        for entity in curated_entities_lower:
            if len(entity) > 1:
                entity_pattern = r"\b" + re.escape(entity) + r"\b"
                try:
                    if re.search(entity_pattern, question_lower):
                        loguru.logger.trace(
                            f"Fallback check passed for '{q_type}': Found '{entity}' in '{question_text}'"
                        )
                        return True
                except re.error as e:
                    loguru.logger.warning(
                        f"Regex error during fallback check for entity '{entity}': {e}"
                    )
                    continue
        loguru.logger.debug(
            f"Fallback check failed for '{q_type}': No curated entity found in '{question_text}'"
        )
        return False  # Strict: if no pattern, fail unless fallback finds something

    # Try to match the pattern and extract the entity phrase
    match = re.match(
        pattern, question_text, re.IGNORECASE
    )  # Use re.match to anchor at the beginning
    if match:
        # Extract the captured group (the entity phrase)
        extracted_entity_phrase = match.group(1).strip()
        extracted_entity_lower = extracted_entity_phrase.lower()

        # Check if this EXACT extracted phrase is in the curated list
        if extracted_entity_lower in curated_entities_lower:
            loguru.logger.trace(
                f"Strict check passed for '{q_type}': Extracted '{extracted_entity_phrase}' is in curated list. Q: {question_text}"
            )
            return True
        else:
            loguru.logger.debug(
                f"Strict check failed for '{q_type}': Extracted '{extracted_entity_phrase}' NOT in curated list. Q: {question_text}"
            )
            # Less strict check
            # question_lower = question_text.lower()
            # for entity in curated_entities_lower:
            #     if len(entity) > 1 and re.search(r"\b" + re.escape(entity) + r"\b", question_lower):
            #         loguru.logger.trace(f"Strict check failed, but fallback found '{entity}' in '{question_text}'")
            #         return True # Allow if strict fails but fallback passes? Decide based on requirement. Currently: No.
            return False
    else:
        # If the specific pattern for the question type didn't match the question text structure
        loguru.logger.warning(
            f"Regex pattern for '{q_type}' did not match question structure: {question_text}"
        )
        # Apply fallback check here as well? Or just fail? Let's fail for now to be strict.
        return False


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
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
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
    loguru.logger.add(log_file_path, level=args.log_level.upper())
    loguru.logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")

    # *** Load Inputs ***
    loguru.logger.info(f"Loading curated entity list from: {curated_entity_list_path}")
    # Load curated entities and convert to lowercase immediately
    curated_entities_lower = load_text_list(curated_entity_list_path)
    if not curated_entities_lower:
        loguru.logger.error(
            "Curated entity list is empty or could not be loaded. Exiting."
        )
        return
    loguru.logger.info(
        f"Loaded {len(curated_entities_lower)} curated entities (lowercased)."
    )

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
            "final_entity_subset.txt",
            "subset_finding.log",
            "final_verification_report.json",
            "final_entity_subset.txt",
        }
    ]

    if not recipe_files:
        loguru.logger.error(
            f"No processed recipe JSON files found in {processed_recipes_dir}. Exiting."
        )
        return

    loguru.logger.info(f"Found {len(recipe_files)} potential recipe files to process.")

    relevant_recipe_count = 0
    processed_question_count = 0
    skipped_entity_mismatch_count = 0

    for recipe_file in tqdm(recipe_files, desc="Filtering Recipes & Questions"):
        recipe_data = load_json(recipe_file)
        if not recipe_data:
            loguru.logger.warning(f"Could not load or parse {recipe_file}, skipping.")
            continue

        title = recipe_data.get("title")
        # Get recipe entities identified in the JSON and lowercase them for comparison
        recipe_entities_in_json_lower = set(
            e.lower() for e in recipe_data.get("entities", {}).keys()
        )
        questions = recipe_data.get("questions", [])
        instructions = recipe_data.get("instructions", {})

        if not title or not questions or not instructions:
            loguru.logger.warning(
                f"Missing title, questions, or instructions in {recipe_file}, skipping."
            )
            continue

        # Check if *this recipe* contains *any* curated entity (using lowercased sets)
        recipe_has_curated_entity = bool(
            curated_entities_lower.intersection(recipe_entities_in_json_lower)
        )

        if recipe_has_curated_entity:
            relevant_recipe_count += 1
            # Store instructions for later prompt generation IF the recipe is relevant
            recipe_instructions_cache[title] = instructions

            # Now, filter the questions *within* this relevant recipe
            for q_data in questions:
                processed_question_count += 1
                q_type = q_data.get("type")
                question_text = q_data.get("question")
                answer = q_data.get("answer")

                # Basic validity check
                is_valid_question = q_type and question_text and answer is not None
                if not is_valid_question:
                    loguru.logger.trace(
                        f"Skipping invalid question data: {q_data} from {title}"
                    )
                    continue

                # Perform the NEW strict relevance check
                is_relevant = check_question_relevance(
                    question_text, q_type, curated_entities_lower
                )

                if is_relevant:
                    all_eligible_questions.append(
                        {
                            "recipe_title": title,
                            "type": q_type,
                            "question": question_text,
                            "answer": answer,
                        }
                    )
                else:
                    # Log only if it failed the strict check (debug level already handles details)
                    skipped_entity_mismatch_count += 1
                    # No need for extra log here, check_question_relevance logs the failure reason

        else:
            loguru.logger.trace(
                f"Recipe '{title}' skipped (no overlap with curated entities)."
            )

    loguru.logger.info(
        f"Found {relevant_recipe_count} relevant recipes containing at least one curated entity."
    )
    loguru.logger.info(
        f"Processed {processed_question_count} questions from relevant recipes."
    )
    loguru.logger.info(
        f"Skipped {skipped_entity_mismatch_count} questions failing strict entity check."
    )
    loguru.logger.info(
        f"Collected {len(all_eligible_questions)} eligible questions meeting all criteria."
    )

    if not all_eligible_questions:
        loguru.logger.error(
            "No eligible questions found after filtering. Cannot generate CSVs. Exiting."
        )
        return

    # Score Recipes
    loguru.logger.info("Scoring recipes based on eligible question type diversity...")
    questions_by_recipe = defaultdict(lambda: defaultdict(list))
    for q_data in all_eligible_questions:
        questions_by_recipe[q_data["recipe_title"]][q_data["type"]].append(q_data)

    recipe_scores = defaultdict(int)
    for title, types_dict in questions_by_recipe.items():
        recipe_scores[title] = len(types_dict)  # Score is the count of unique types

    loguru.logger.info("Finished scoring recipes.")

    # Select Final Questions
    loguru.logger.info(
        f"Selecting top {target_count} questions per category from eligible pool, prioritizing diverse recipes..."
    )
    final_questions_by_type = defaultdict(list)
    all_question_types = sorted(list(set(q["type"] for q in all_eligible_questions)))

    for q_type in tqdm(all_question_types, desc="Selecting Questions per Type"):
        # Get all eligible questions of this type
        type_questions = [q for q in all_eligible_questions if q["type"] == q_type]

        if not type_questions:
            loguru.logger.warning(
                f"No eligible questions found for type '{q_type}' after filtering."
            )
            continue

        # Sort them: 1. Recipe Score (desc), 2. Recipe Title (asc), 3. Question Text (asc)
        # Note: Higher score is better, so reverse=True applies to the score tuple element
        sorted_type_questions = sorted(
            type_questions,
            key=lambda q: (
                -recipe_scores.get(
                    q["recipe_title"], 0
                ),  # Score (negated for descending sort)
                q["recipe_title"],  # Title (ascending)
                q["question"],  # Question text (ascending)
            ),
        )

        # Select top N
        selected_questions = sorted_type_questions[:target_count]
        final_questions_by_type[q_type] = selected_questions

        if len(selected_questions) < target_count:
            loguru.logger.warning(
                f"Found only {len(selected_questions)} eligible questions for type '{q_type}', which is less than the target {target_count}."
            )
        else:
            loguru.logger.info(  # Changed to INFO for successful selection
                f"Selected {len(selected_questions)} questions for type '{q_type}'."
            )

    # *** Format and Write CSVs ***
    loguru.logger.info("Formatting prompts and writing CSV files...")
    num_csv_written = 0
    final_counts_summary = {}
    for q_type, questions in tqdm(final_questions_by_type.items(), desc="Writing CSVs"):
        if not questions:
            # This case should be handled by the warning above, but double-check
            loguru.logger.warning(
                f"No questions selected for type '{q_type}', skipping CSV generation."
            )
            final_counts_summary[q_type] = 0
            continue

        output_data = []
        for q_data in questions:
            recipe_title = q_data["recipe_title"]
            question_text = q_data["question"]
            answer = q_data["answer"]

            # Retrieve cached instructions
            instructions = recipe_instructions_cache.get(recipe_title)
            if not instructions:
                # This indicates an issue if a question was selected but its recipe wasn't cached
                loguru.logger.error(
                    f"CRITICAL: Could not find cached instructions for recipe '{recipe_title}' associated with an eligible question. Skipping question: {question_text}"
                )
                continue

            # Format the prompt
            prompt = format_prompt(recipe_title, instructions, question_text)

            output_data.append({"prompt": prompt, "answer": answer})

        # Create DataFrame
        df = pd.DataFrame(output_data)
        final_counts_summary[q_type] = len(df)  # Store final count for summary

        # Save to CSV
        csv_filename = sanitize_filename(q_type) + "_questions.csv"
        csv_path = output_csv_dir / csv_filename
        try:
            df.to_csv(csv_path, index=False, encoding="utf-8")
            loguru.logger.debug(f"Successfully wrote {len(df)} rows to {csv_path}")
            num_csv_written += 1
        except Exception as e:
            loguru.logger.error(f"Failed to write CSV file {csv_path}: {e}")

    loguru.logger.success(
        f"Finished writing {num_csv_written} CSV files to: {output_csv_dir}"
    )
    loguru.logger.info("Final question counts per category:")
    for q_type, count in sorted(final_counts_summary.items()):
        loguru.logger.info(f"- {q_type}: {count}")

    all_targets_met = True
    generated_types = set(final_counts_summary.keys())

    # Check counts for generated types
    for q_type, count in final_counts_summary.items():
        if count < target_count:
            all_targets_met = False
            loguru.logger.warning(
                f"Target count NOT MET for '{q_type}' (found {count})"
            )

    # Check for missing types
    all_expected_types = {
        "Reaching Definitions",
        "Very Busy Expressions",
        "Available Expressions",
        "Live Variable Analysis",
        "Interval Analysis",
        "Type-State Analysis",
        "Taint Analysis",
        "Concurrency Analysis",
    }
    missing_types = all_expected_types - generated_types
    if missing_types:
        all_targets_met = False
        loguru.logger.warning(
            f"Missing CSVs for types: {', '.join(sorted(list(missing_types)))}"
        )

    if all_targets_met:
        loguru.logger.success(
            f"Target count of {target_count} met or exceeded for all expected categories."
        )
    else:
        loguru.logger.warning(
            f"Target count of {target_count} was NOT met for one or more categories OR some categories are missing."
        )


if __name__ == "__main__":
    main()
