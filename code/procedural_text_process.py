import argparse
import json
import os
import random
from collections import Counter
from shutil import rmtree
from typing import Any, Dict, List, Optional

import loguru
import matplotlib.pyplot as plt
import spacy
from helpers import ProceduralText, QuestionGenerator
from tqdm import tqdm

# Load spaCy model for NLP processing
spacy.require_gpu()
# nlp = spacy.load("en_core_web_sm")  # Smaller model for faster processing
nlp = spacy.load("en_core_web_trf")  # Larger model for better accuracy


def plot_entity_frequency(entity_counter: Counter, output_folder: str) -> None:
    """
    Plot the frequency of entities and save the plot.

    :param entity_counter: Counter object with entity counts
    :param output_folder: Folder where to save the plot
    """
    # Get the most common entities (top 20)
    most_common = entity_counter.most_common(20)

    if not most_common:
        loguru.logger.warning("No entities to plot.")
        return

    entities, counts = zip(*most_common)

    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(entities)), counts, color="skyblue")

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            str(int(height)),
            ha="center",
            va="bottom",
        )

    plt.xticks(range(len(entities)), entities, rotation=45, ha="right")
    plt.title("Top 20 Most Frequent Entities")
    plt.xlabel("Entity")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_folder, "entity_frequency.png")
    plt.savefig(plot_path)
    loguru.logger.info(f"Entity frequency plot saved to {plot_path}")

    # Create a more detailed plot for the top 50 entities
    if len(entity_counter) > 20:
        top_50 = entity_counter.most_common(50)
        entities_50, counts_50 = zip(*top_50)

        plt.figure(figsize=(15, 10))
        plt.bar(range(len(entities_50)), counts_50, color="lightgreen")
        plt.xticks(range(len(entities_50)), entities_50, rotation=45)
        plt.title("Top 50 Most Frequent Entities")
        plt.xlabel("Entity")
        plt.ylabel("Frequency")
        plt.tight_layout()

        plot_path_50 = os.path.join(output_folder, "entity_frequency_top50.png")
        plt.savefig(plot_path_50)
        loguru.logger.info(f"Extended entity frequency plot saved to {plot_path_50}")


def process_single_recipe(title: str, instructions: List[str]) -> Dict[str, Any]:
    """
    Process a single recipe and generate questions.

    :param title: Recipe title
    :param instructions: List of recipe instructions
    :return: Dictionary with processed recipe data
    """
    # Create procedural text
    proc_text = ProceduralText.ProceduralText(title, instructions, nlp)

    # Generate questions
    question_gen = QuestionGenerator.QuestionGenerator(proc_text, nlp)
    questions = question_gen.generate_all_questions(num_per_type=1)

    # Format instructions as a dictionary with step numbers
    formatted_instructions = {i + 1: step for i, step in enumerate(instructions)}

    # Format entities for output
    formatted_entities = {}
    for entity_name, entity in proc_text.entities.items():
        formatted_entities[entity_name] = {
            "step_introduced": entity.step_introduced + 1,  # Convert to 1-indexed
            "states": entity.states,
            "used_in": sorted([step + 1 for step in entity.used_in]),
            "defined_in": sorted([step + 1 for step in entity.defined_in]),
            "consumed_in": sorted([step + 1 for step in entity.consumed_in]),
        }

    # Format questions for output
    formatted_questions = []
    for q_type, (question, answer) in questions:
        if answer is not None:  # Only include valid questions
            formatted_questions.append(
                {"type": q_type, "question": question, "answer": answer}
            )

    # Create the output dictionary
    result = {
        "title": title,
        "instructions": formatted_instructions,
        "questions": formatted_questions,
        "entities": formatted_entities,
        "metadata": {
            "num_steps": len(instructions),
            "num_entities": len(formatted_entities),
            "num_questions": len(formatted_questions),
        },
    }

    return result


def read_recipe_from_json(file_path: str) -> Dict[str, Any]:
    """
    Read a recipe from a JSON file.

    :param file_path: Path to the JSON file
    :return: Dictionary containing the recipe data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_recipe_folder(
    folder_path: str, output_folder: str, limit: Optional[int] = None
) -> None:
    """
    Process recipe JSON files in a folder until reaching the specified limit of questions for each type.

    :param folder_path: Path to the folder containing recipe JSON files
    :param output_folder: Path to the folder where output JSON files will be saved
    :param limit: Optional limit on the number of questions to collect for each type
    """
    # Delete output folder if it exists
    if os.path.exists(output_folder):
        rmtree(output_folder)
        loguru.logger.info(f"Deleting existing output folder: {output_folder}")

    loguru.logger.info(f"Creating new output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    random.shuffle(json_files)  # Shuffle to get a random selection

    # Track questions by type
    question_counts = {
        "Reaching Definitions": 0,
        "Very Busy Expressions": 0,
        "Available Expressions": 0,
        "Live Variable Analysis": 0,
        "Interval Analysis": 0,
        "Type-State Analysis": 0,
        "Taint Analysis": 0,
        "Concurrency Analysis": 0,
    }

    # Process each file until we reach the limit for all question types
    all_entities = Counter()
    recipe_stats = []
    processed_recipes = 0
    all_questions = []  # Store all generated questions

    # Define the target: we want at least 'limit' questions of each type
    target_reached = False

    with tqdm(
        total=limit * len(question_counts) if limit else len(json_files),
        desc="Collecting questions",
    ) as pbar:
        for file_name in json_files:
            # Check if we've reached the target for all question types
            if limit and all(count >= limit for count in question_counts.values()):
                target_reached = True
                break

            file_path = os.path.join(folder_path, file_name)
            recipe_data = read_recipe_from_json(file_path)

            # Extract relevant information
            title = recipe_data.get("title", "Untitled Recipe")
            instructions = recipe_data.get("instructions_list", [])

            # Skip recipes with no instructions or too few steps
            if not instructions:
                loguru.logger.warning(
                    f"No instructions found in {file_name}, skipping."
                )
                continue

            if len(instructions) < 3:
                loguru.logger.warning(
                    f"Recipe {file_name} has fewer than 3 steps, skipping."
                )
                continue

            # Process the recipe
            try:
                # Create procedural text
                proc_text = ProceduralText.ProceduralText(title, instructions, nlp)

                # Generate questions
                question_gen = QuestionGenerator.QuestionGenerator(proc_text, nlp)
                questions = question_gen.generate_all_questions(num_per_type=1)

                # Count valid questions by type and update progress
                valid_questions = []
                for q_type, (question, answer) in questions:
                    if answer is not None:  # Only include valid questions
                        valid_questions.append(
                            {"type": q_type, "question": question, "answer": answer}
                        )

                        # Only increment if we haven't reached the limit for this type
                        if limit and question_counts[q_type] < limit:
                            question_counts[q_type] += 1
                            pbar.update(1)

                            # Add to our collection of all questions
                            all_questions.append(
                                {
                                    "type": q_type,
                                    "question": question,
                                    "answer": answer,
                                    "recipe": title,
                                }
                            )

                # If this recipe generated valid questions, process and save it
                if valid_questions:
                    # Format instructions as a dictionary with step numbers
                    formatted_instructions = {
                        i + 1: step for i, step in enumerate(instructions)
                    }

                    # Format entities for output
                    formatted_entities = {}
                    for entity_name, entity in proc_text.entities.items():
                        formatted_entities[entity_name] = {
                            "step_introduced": entity.step_introduced
                            + 1,  # Convert to 1-indexed
                            "states": entity.states,
                            "used_in": [
                                step + 1 for step in entity.used_in
                            ],  # Convert to 1-indexed
                            "defined_in": [
                                step + 1 for step in entity.defined_in
                            ],  # Convert to 1-indexed
                            "consumed_in": [
                                step + 1 for step in entity.consumed_in
                            ],  # Convert to 1-indexed
                        }

                    # Count entities
                    for entity in formatted_entities.keys():
                        all_entities[entity] += 1

                    # Create the output dictionary
                    result = {
                        "title": title,
                        "instructions": formatted_instructions,
                        "questions": valid_questions,
                        "entities": formatted_entities,
                        "metadata": {
                            "num_steps": len(instructions),
                            "num_entities": len(formatted_entities),
                            "num_questions": len(valid_questions),
                        },
                    }

                    # Save the processed recipe
                    output_path = os.path.join(output_folder, file_name)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)

                    # Collect stats
                    recipe_stats.append(
                        {
                            "title": title,
                            "num_steps": len(instructions),
                            "num_entities": len(formatted_entities),
                            "num_questions": len(valid_questions),
                        }
                    )

                    loguru.logger.info(
                        f"Processed {title} - {len(instructions)} steps, {len(formatted_entities)} entities, {len(valid_questions)} questions"
                    )

                    processed_recipes += 1

            except Exception as e:
                loguru.logger.error(f"Error processing {file_name}: {str(e)}")

    # Save the entity counts
    with open(
        os.path.join(output_folder, "all_entities.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {"entities": list(all_entities.keys()), "entity_counts": all_entities},
            f,
            indent=2,
        )

    # Plot entity frequency
    plot_entity_frequency(all_entities, output_folder)

    # Save all collected questions
    with open(
        os.path.join(output_folder, "all_questions.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_questions, f, indent=2)

    # Save question counts by type
    with open(
        os.path.join(output_folder, "question_counts.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(question_counts, f, indent=2)

    # Save processing statistics
    with open(
        os.path.join(output_folder, "processing_stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "total_recipes_processed": processed_recipes,
                "total_entities": len(all_entities),
                "question_counts": question_counts,
                "target_reached": target_reached,
                "recipes": recipe_stats,
            },
            f,
            indent=2,
        )

    # Log summary
    loguru.logger.info(f"Successfully processed {processed_recipes} recipes")
    loguru.logger.info(f"Question counts by type: {question_counts}")
    if target_reached:
        loguru.logger.info(f"Target of {limit} questions per type reached!")
    else:
        loguru.logger.warning(
            f"Target of {limit} questions per type NOT reached. Exhausted all available recipes."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Process recipe JSON files and generate questions"
    )
    parser.add_argument("input_folder", help="Folder containing recipe JSON files")
    parser.add_argument(
        "output_folder", help="Folder where processed recipes will be saved"
    )
    parser.add_argument(
        "--limit",
        default=500,
        type=int,
        help="The limit on the number of questions per type to collect",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Configure logging
    loguru.logger.remove()
    loguru.logger.add(
        os.path.join(args.output_folder, "processing.log"),
        level=args.log_level,
    )
    loguru.logger.add(lambda msg: tqdm.write(msg, end=""), level=args.log_level)

    random.seed(13)

    loguru.logger.info(
        f"Starting recipe processing. Input: '{args.input_folder}', Output: '{args.output_folder}', Limit/Type: {args.limit}, Log Level: {args.log_level}"
    )

    process_recipe_folder(args.input_folder, args.output_folder, args.limit)

    loguru.logger.info("Recipe processing complete!")


if __name__ == "__main__":
    main()
