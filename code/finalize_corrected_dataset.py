import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import loguru
import pandas as pd
import spacy
from helpers import ProceduralText, QuestionGenerator, constants
from post_processing import convert_csv

# Global constants
ANNOTATED_FILE_SUFFIX = "_annotated.csv"
FINAL_FILE_PREFIX = "recipe_"


# Class within finalize_corrected_dataset.py
class InteractiveQuestionHandler:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.pt_cache = {}  # Cache ProceduralText objects
        # Define a mapping from the canonical analysis type string to the method suffix
        self.analysis_to_method_suffix = {
            "Reaching Definitions": "reaching_definitions",
            "Very Busy Expressions": "very_busy_expressions",
            "Available Expressions": "available_expressions",
            "Live Variable Analysis": "live_variable",
            "Interval Analysis": "interval_analysis",
            "Type-State Analysis": "type_state",
            "Taint Analysis": "taint_analysis",
            "Concurrency Analysis": "concurrency_analysis",
        }

    def _get_procedural_text(
        self, recipe_title: str, instructions_dict: dict
    ) -> ProceduralText.ProceduralText | None:
        if recipe_title in self.pt_cache:
            return self.pt_cache[recipe_title]

        if not instructions_dict:
            loguru.logger.error(
                f"No instructions for recipe '{recipe_title}' to create ProceduralText."
            )
            return None

        try:
            # Convert instructions_dict {"1": "text", ...} to list ["text", ...]
            max_step = 0
            if instructions_dict.keys():  # Check if dict is not empty
                max_step = max(int(k) for k in instructions_dict.keys())

            instructions_list = [""] * max_step
            for k, v in instructions_dict.items():
                instructions_list[int(k) - 1] = v

            # Check for gaps if not empty
            if any(s == "" for s in instructions_list if max_step > 0):
                loguru.logger.warning(
                    f"Instructions for '{recipe_title}' seem incomplete. Proceeding cautiously."
                )

        except ValueError:
            loguru.logger.error(
                f"Cannot convert instructions for '{recipe_title}' to list. Skipping."
            )
            return None

        pt = ProceduralText.ProceduralText(
            goal=recipe_title, steps=instructions_list, nlp=self.nlp
        )
        self.pt_cache[recipe_title] = pt
        return pt

    def generate_and_verify_replacement(
        self, recipe_title: str, instructions_dict: dict, analysis_type: str
    ) -> tuple[str | None, Any | None]:
        """
        Generates a new question for the given recipe/type and verifies it.
        Returns (new_question_text, new_answer) if verified, else (None, None) or ("SKIP_RECIPE_FOR_TYPE", None).
        """
        pt = self._get_procedural_text(recipe_title, instructions_dict)
        if not pt:
            return None, None  # Return a 2-tuple

        q_gen = QuestionGenerator.QuestionGenerator(pt, self.nlp)

        method_suffix = self.analysis_to_method_suffix.get(analysis_type)
        if not method_suffix:
            loguru.logger.error(
                f"Unknown analysis type for method mapping: {analysis_type}"
            )
            return None, None  # Return a 2-tuple

        generator_func_name = f"generate_{method_suffix}_question"

        if not hasattr(q_gen, generator_func_name):
            loguru.logger.error(
                f"QuestionGenerator missing method: {generator_func_name} "
                f"(derived from analysis type '{analysis_type}' with mapped suffix '{method_suffix}')"
            )
            return None, None  # Return a 2-tuple

        generator_method = getattr(q_gen, generator_func_name)

        for attempt in range(3):  # Max 3 retries
            loguru.logger.info(
                f"Attempt {attempt + 1} to generate replacement for '{recipe_title}', type '{analysis_type}'"
            )
            q_result = generator_method()

            new_q_text, new_answer = None, None
            if isinstance(q_result, list) and q_result:  # For VBE
                # Ensure q_result[0] is a tuple of two items before unpacking
                if isinstance(q_result[0], tuple) and len(q_result[0]) == 2:
                    new_q_text, new_answer = q_result[0]
                else:
                    loguru.logger.warning(
                        f"Unexpected format for VBE result item: {q_result[0]}"
                    )
            elif isinstance(q_result, tuple) and len(q_result) == 2:
                new_q_text, new_answer = q_result

            if new_q_text and new_answer is not None:
                print("\n***** INTERACTIVE VERIFICATION (Replacement) *****")
                # Display recipe context (title, steps from instructions_dict)
                print(f"Recipe: {recipe_title}")
                for step_num_str, step_text_str in sorted(
                    instructions_dict.items(), key=lambda x: int(x[0])
                ):
                    print(f"  Step {step_num_str}: {step_text_str}")
                print(f"\nAnalysis Type: {analysis_type}")
                print(f"Generated Question: {new_q_text}")
                print(f"Proposed Answer: {new_answer}")

                while True:
                    choice = input(
                        "Is this correct? (y/n/s - skip this recipe for this type): "
                    ).lower()
                    if choice == "y":
                        return new_q_text, new_answer
                    elif choice == "n":
                        loguru.logger.info(
                            "User marked as incorrect. Retrying generation..."
                        )
                        break  # To next attempt in the for loop
                    elif choice == "s":
                        loguru.logger.info(
                            f"User chose to skip recipe '{recipe_title}' for type '{analysis_type}' for replacements."
                        )
                        return "SKIP_RECIPE_FOR_TYPE", None  # Special marker
                    else:
                        print("Invalid input.")
            else:
                loguru.logger.warning(
                    f"Generator for '{analysis_type}' failed for '{recipe_title}' (Attempt {attempt + 1}). Message: {new_q_text if new_q_text else 'N/A'}"
                )

        loguru.logger.warning(
            f"Failed to generate a verified replacement for '{recipe_title}', type '{analysis_type}'."
        )
        return None, None  # Explicitly return two values

    def verify_existing_qa(
        self,
        recipe_title: str,
        instructions_dict: dict,
        question_text: str,
        answer: Any,
        analysis_type: str,
    ) -> bool:
        """
        Presents an existing Q&A (e.g., from all_questions.json) for verification.
        Returns True if user confirms, False otherwise.
        """
        print("\n***** INTERACTIVE VERIFICATION (Gap Filling) *****")
        # Display recipe context
        print(f"Recipe: {recipe_title}")
        for step_num_str, step_text_str in sorted(
            instructions_dict.items(), key=lambda x: int(x[0])
        ):
            print(f"  Step {step_num_str}: {step_text_str}")
        print(f"\nAnalysis Type: {analysis_type}")
        print(f"Candidate Question: {question_text}")
        print(f"Candidate Answer: {answer}")

        while True:
            choice = input("Use this Q&A? (y/n): ").lower()
            if choice == "y":
                return True
            elif choice == "n":
                return False
            else:
                print("Invalid input.")


def parse_prompt_details(
    prompt_str: str,
) -> tuple[str | None, str | None, dict | None, str | None]:
    """
    Parses a prompt string to extract recipe title, original question text,
    instructions as a dictionary, and the full original prompt.
    Returns: (recipe_title, original_question_text, instructions_dict, original_prompt)
    """
    lines = prompt_str.splitlines()
    recipe_title = None
    original_question_text = None
    instructions_dict = {}

    parsing_state = "GOAL"  # GOAL, STEPS, QUESTION

    current_question_lines = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith("Goal:"):
            recipe_title = line_stripped.replace("Goal:", "").strip()
            parsing_state = "AFTER_GOAL"
        elif line_stripped == "Steps:":
            parsing_state = "STEPS"
        elif line_stripped == "Question:":
            parsing_state = "QUESTION"
            # The actual question text starts on the next line
            current_question_lines = lines[i + 1 :]  # Assuming question is last
            original_question_text = "\n".join(current_question_lines).strip()
            break  # Question is the last part
        elif parsing_state == "STEPS":
            match = re.match(r"Step (\d+): (.*)", line_stripped)
            if match:
                instructions_dict[match.group(1)] = match.group(2).strip()

    if not recipe_title or not original_question_text or not instructions_dict:
        # Log an error or warning if parsing fails
        loguru.logger.warning(
            f"Could not fully parse prompt: {prompt_str[:100]}..."
        )  # Log first 100 chars
        return None, None, None, prompt_str

    return recipe_title, original_question_text, instructions_dict, prompt_str


def format_prompt(
    recipe_title: str, instructions_dict: dict, question_text: str
) -> str:
    prompt_lines = [f"Goal: {recipe_title}", "\nSteps:"]
    for step_num, step_text in sorted(
        instructions_dict.items(), key=lambda item: int(item[0])
    ):
        prompt_lines.append(f"Step {step_num}: {step_text}")
    prompt_lines.append("\nQuestion:")
    prompt_lines.append(question_text)
    return "\n".join(prompt_lines)


def main():
    parser = argparse.ArgumentParser(description="Finalize corrected dataset.")
    parser.add_argument(
        "--annotated_csvs_folder",
        type=str,
        required=True,
        help="Path to folder with annotated CSVs.",
    )
    parser.add_argument(
        "--processed_json_folder",
        type=str,
        required=True,
        help="Path to folder with original processed recipe JSONs (from procedural_text_process.py output).",
    )
    parser.add_argument(
        "--all_questions_json_path",
        type=str,
        required=True,
        help="Path to all_questions.json (from procedural_text_process.py output).",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to output folder for final corrected CSVs.",
    )
    parser.add_argument(
        "--final_converted_folder",
        type=str,
        required=True,
        help="Path to output folder for final converted CSVs.",
    )

    args = parser.parse_args()

    # Ensure output folder exists
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Set up logging
    loguru.logger.add(
        output_folder / "finalize_corrected_dataset.log",
        level="INFO",
        format="{time} {level} {message}",
    )
    loguru.logger.info("Starting dataset finalization...")
    loguru.logger.info(f"Output folder: {output_folder}")
    loguru.logger.info(f"Annotated CSVs folder: {args.annotated_csvs_folder}")
    loguru.logger.info(f"Processed JSONs folder: {args.processed_json_folder}")
    loguru.logger.info(f"All questions JSON path: {args.all_questions_json_path}")
    loguru.logger.info(f"Final converted folder: {args.final_converted_folder}")

    # Initialize NLP and handler
    nlp = spacy.load(constants.NLP_MODEL_TRF)
    q_handler = InteractiveQuestionHandler(nlp)

    # Load all_questions.json for gap filling
    all_questions_path = Path(args.all_questions_json_path)
    all_questions_pool = []
    if all_questions_path.exists():
        with open(all_questions_path, "r", encoding="utf-8") as f:
            all_questions_pool = json.load(f)
    else:
        loguru.logger.error(
            f"{all_questions_path} not found. Gap filling will be limited."
        )

    # Load processed recipe JSONs to get instructions for gap-filling prompts
    processed_recipes_data = {}  # {recipe_title: instructions_dict}
    for recipe_json_file in Path(args.processed_json_folder).glob("*.json"):
        if recipe_json_file.name in constants.NON_RECIPE_JSON_FILENAMES:
            continue
        try:
            with open(recipe_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "title" in data and "instructions" in data:
                    processed_recipes_data[data["title"]] = data["instructions"]
        except Exception as e:
            loguru.logger.warning(f"Could not load/parse {recipe_json_file}: {e}")

    for analysis_type_enum in constants.ANALYSIS_TYPES:
        annotated_filename_stem = (
            f"recipe_{analysis_type_enum.lower().replace(' ', '_')}"
        )
        annotated_file_path = Path(args.annotated_csvs_folder) / (
            annotated_filename_stem + ANNOTATED_FILE_SUFFIX
        )

        if not annotated_file_path.exists():
            loguru.logger.warning(
                f"Annotated file not found: {annotated_file_path}. Skipping {analysis_type_enum}."
            )
            continue

        loguru.logger.info(
            f"Processing {analysis_type_enum} from {annotated_file_path}..."
        )
        df_annotated = pd.read_csv(annotated_file_path)

        final_qa_pairs_for_type = []
        skipped_recipes_for_replacement = set()  # Stores (recipe_title, analysis_type)

        for _, row in df_annotated.iterrows():
            original_prompt = row["prompt"]
            original_answer = row["answer"]
            # Handle cases where corrected_answer might be NaN if empty in CSV
            corrected_answer = (
                row["corrected_answer"] if pd.notna(row["corrected_answer"]) else ""
            )

            recipe_title, _, instructions_dict, _ = parse_prompt_details(
                original_prompt
            )

            if not recipe_title or not instructions_dict:
                loguru.logger.warning(
                    f"Could not parse prompt for a row in {annotated_file_path}. Keeping original if possible."
                )
                # Decide how to handle: skip row, or keep original if corrected_answer is empty
                if not corrected_answer or str(corrected_answer) == str(
                    original_answer
                ):
                    final_qa_pairs_for_type.append(
                        {"prompt": original_prompt, "answer": original_answer}
                    )
                else:
                    loguru.logger.error(
                        "Skipping row due to parse error and correction needed."
                    )
                continue

            if not corrected_answer or str(corrected_answer) == str(
                original_answer
            ):  # Correct or no correction provided
                final_qa_pairs_for_type.append(
                    {"prompt": original_prompt, "answer": original_answer}
                )
            else:  # Incorrect, needs replacement
                if (
                    recipe_title,
                    analysis_type_enum,
                ) in skipped_recipes_for_replacement:
                    loguru.logger.info(
                        f"Skipping replacement for '{recipe_title}' ({analysis_type_enum}) as previously directed by user."
                    )
                    # This Q&A is dropped, will be handled by gap-filling
                    continue

                loguru.logger.info(
                    f"Replacing incorrect Q&A for '{recipe_title}' ({analysis_type_enum})."
                )
                new_q_text, new_a = q_handler.generate_and_verify_replacement(
                    recipe_title, instructions_dict, analysis_type_enum
                )

                if new_q_text == "SKIP_RECIPE_FOR_TYPE":
                    skipped_recipes_for_replacement.add(
                        (recipe_title, analysis_type_enum)
                    )
                    # This Q&A is dropped
                elif new_q_text and new_a is not None:
                    # Format new prompt using original recipe context but new question
                    new_prompt = format_prompt(
                        recipe_title, instructions_dict, new_q_text
                    )  # You'll need this helper
                    final_qa_pairs_for_type.append(
                        {"prompt": new_prompt, "answer": new_a}
                    )
                else:
                    loguru.logger.warning(
                        f"Failed to get replacement for '{recipe_title}' ({analysis_type_enum}). This slot needs filling."
                    )
                    # This Q&A is dropped

        # Gap Filling
        num_needed = 100 - len(final_qa_pairs_for_type)
        if num_needed > 0:
            loguru.logger.info(
                f"Need to fill {num_needed} more Q&As for {analysis_type_enum}."
            )

            # Create a set of prompts already included to avoid duplicates
            # Note: format_prompt should be consistent.
            existing_prompts_texts = set(
                item["prompt"] for item in final_qa_pairs_for_type
            )

            # Filter all_questions_pool for the current analysis type
            candidate_pool = [
                q for q in all_questions_pool if q.get("type") == analysis_type_enum
            ]
            random.shuffle(candidate_pool)

            for cand_q_data in candidate_pool:
                if num_needed <= 0:
                    break

                cand_recipe_title = cand_q_data.get(
                    "recipe"
                )  # 'recipe' key in all_questions.json
                cand_q_text = cand_q_data.get("question")
                cand_answer = cand_q_data.get("answer")

                if not cand_recipe_title or not cand_q_text or cand_answer is None:
                    continue

                # Get instructions for this candidate's recipe
                cand_instructions = processed_recipes_data.get(cand_recipe_title)
                if not cand_instructions:
                    loguru.logger.trace(
                        f"No instructions found in cache for '{cand_recipe_title}' from all_questions.json. Skipping candidate."
                    )
                    continue

                # Format potential new prompt
                potential_new_prompt = format_prompt(
                    cand_recipe_title, cand_instructions, cand_q_text
                )
                if potential_new_prompt in existing_prompts_texts:
                    loguru.logger.trace(
                        f"Skipping duplicate candidate prompt for {cand_recipe_title}."
                    )
                    continue

                if q_handler.verify_existing_qa(
                    cand_recipe_title,
                    cand_instructions,
                    cand_q_text,
                    cand_answer,
                    analysis_type_enum,
                ):
                    final_qa_pairs_for_type.append(
                        {"prompt": potential_new_prompt, "answer": cand_answer}
                    )
                    existing_prompts_texts.add(potential_new_prompt)
                    num_needed -= 1
                    loguru.logger.info(
                        f"Gap filled for {analysis_type_enum}. Remaining: {num_needed}"
                    )

            if num_needed > 0:
                loguru.logger.error(
                    f"COULD NOT FILL ALL GAPS for {analysis_type_enum}. Still need {num_needed}."
                )

        # Save the final CSV for this type
        output_df = pd.DataFrame(final_qa_pairs_for_type)
        if len(output_df) > 100:  # If somehow we got more than 100, truncate
            output_df = output_df.sample(
                n=100, random_state=constants.RANDOM_SEED
            ).reset_index(drop=True)

        final_csv_path = Path(args.output_folder) / (
            FINAL_FILE_PREFIX + analysis_type_enum.lower().replace(" ", "_") + ".csv"
        )
        output_df.to_csv(
            final_csv_path, index=False, columns=["prompt", "answer"]
        )  # Ensure only these columns
        loguru.logger.info(
            f"Saved {len(output_df)} Q&As for {analysis_type_enum} to {final_csv_path}"
        )

    loguru.logger.info("Running final post-processing steps...")
    convert_csv(args.output_folder, args.final_converted_folder)
    loguru.logger.info("Dataset finalization complete.")


if __name__ == "__main__":
    main()
