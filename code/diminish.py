import argparse
import json
import os
import re
from collections import defaultdict
from math import ceil
from shutil import rmtree

import loguru
from tqdm import tqdm

# Explicit list of common cooking verbs (lemmas) to filter out
# Path assumes running from home dir.
VERB_STOP_LIST = set(line.strip() for line in open("data/verbs_stop_list.txt", "r"))


def load_json(file_path: str):
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


def is_likely_valid_entity(entity_name: str) -> bool:
    """
    Checks if an entity name is likely valid (e.g., doesn't start with a common verb).
    Simple rule-based check.
    """
    if not entity_name or not isinstance(entity_name, str):
        return False

    # Basic length check
    if len(entity_name.strip()) < 2:
        return False

    # Check if it's purely numeric
    if entity_name.isdigit():
        return False

    # Check if the first word is a verb from the stop list
    first_word = entity_name.split(" ", 1)[0].lower()
    if first_word in VERB_STOP_LIST:
        loguru.logger.debug(
            f"Filtering out entity '{entity_name}' because it starts with verb '{first_word}'"
        )
        return False

    # Add more rules here if needed (e.g., check for units, generic terms)

    return True


def check_question_relevance(question_text: str, entity_subset: set) -> bool:
    """
    Checks if a question text mentions any entity from the subset.
    Uses case-insensitive matching with word boundaries.
    """
    question_lower = question_text.lower()
    for entity in entity_subset:
        # Use word boundaries (\b) to match whole words/phrases
        # Escape special regex characters in the entity name
        # Only check if entity is reasonably long to avoid matching 'a', 's', etc.
        if len(entity) > 1:
            pattern = r"\b" + re.escape(entity.lower()) + r"\b"
            try:
                if re.search(pattern, question_lower):
                    return True
            except re.error as e:
                # Log regex errors if entity name contains problematic chars
                loguru.logger.warning(f"Regex error for entity '{entity}': {e}")
                continue  # Skip this entity if pattern is invalid
    return False


def verify_subset(
    all_questions: list, entity_subset: set, target_count: int
) -> tuple[dict, bool, set]:
    """
    Verifies if the entity subset meets the target question count per type
    based on the heuristic check. Returns counts, overall status (all met),
    and indices of matched questions.
    """
    counts_per_type = defaultdict(int)
    relevant_questions_indices = set()  # Store indices of questions matched

    # Ensure all expected types are initialized
    expected_types = {
        "Reaching Definitions",
        "Very Busy Expressions",
        "Available Expressions",
        "Live Variable Analysis",
        "Interval Analysis",
        "Type-State Analysis",
        "Taint Analysis",
        "Concurrency Analysis",
    }
    for q_type in expected_types:
        counts_per_type[q_type] = 0

    # Count questions matching the heuristic
    for idx, q_data in enumerate(all_questions):
        question_text = q_data.get("question", "")
        q_type = q_data.get("type", "Unknown")

        if q_type in expected_types and check_question_relevance(
            question_text, entity_subset
        ):
            counts_per_type[q_type] += 1
            relevant_questions_indices.add(idx)

    # Check if targets are met (this check is separate from counting)
    # all_met = True  # Assume true initially
    loguru.logger.debug(f"Verification Counts (Subset Size: {len(entity_subset)}):")
    final_counts = {}
    for q_type in sorted(list(expected_types)):  # Sort for consistent output
        count = counts_per_type[q_type]
        final_counts[q_type] = count
        loguru.logger.debug(f"- {q_type}: {count}")

    # Return raw counts, placeholder for all_met (re-evaluated in caller), and indices
    return (
        final_counts,
        True,
        relevant_questions_indices,
    )  # Return True for all_met placeholder


def find_entity_subset(
    filtered_entity_counts: dict,
    all_questions: list,
    target_count: int,
    initial_percentage: float = 0.1,
    increment_percentage: float = 0.05,
) -> tuple[list | None, dict | None]:
    """
    Finds a subset of entities likely sufficient to meet the target question count,
    ignoring types where the entity-mention heuristic is known to fail.
    """
    if not filtered_entity_counts or not all_questions:
        loguru.logger.error(
            "Filtered entity counts or questions data is missing or empty."
        )
        return None, None

    # Types where the entity mention heuristic is unreliable/irrelevant
    heuristic_ignore_types = {"Interval Analysis", "Concurrency Analysis"}
    loguru.logger.info(
        f"Heuristic Check: Will ignore {heuristic_ignore_types} when determining sufficiency."
    )

    # Sort entities by frequency (using the filtered counts)
    sorted_entities = sorted(
        filtered_entity_counts.items(), key=lambda item: item[1], reverse=True
    )
    all_entity_names = [name for name, count in sorted_entities]
    total_unique_entities = len(all_entity_names)

    if total_unique_entities == 0:
        loguru.logger.error("No valid entities remaining after filtering.")
        return None, None

    current_subset_size = ceil(total_unique_entities * initial_percentage)
    if current_subset_size == 0 and total_unique_entities > 0:
        current_subset_size = 1  # Ensure at least one entity if possible

    increment_count = ceil(total_unique_entities * increment_percentage)
    if increment_count == 0:
        increment_count = 1  # Ensure we add at least one

    loguru.logger.info(
        f"Total unique *filtered* entities: {total_unique_entities}. Target questions/type: {target_count}"
    )
    loguru.logger.info(
        f"Starting with initial {initial_percentage * 100:.1f}% ({current_subset_size} entities)."
    )
    loguru.logger.info(
        f"Will increment by {increment_percentage * 100:.1f}% ({increment_count} entities) if needed."
    )

    pbar = tqdm(total=total_unique_entities, desc="Finding sufficient entity subset")
    last_verified_counts = None

    while current_subset_size <= total_unique_entities:
        current_subset_names = set(all_entity_names[:current_subset_size])
        pbar.n = current_subset_size
        pbar.set_postfix({"Subset Size": current_subset_size})
        pbar.refresh()

        loguru.logger.info(f"\n*** Verifying subset of size {current_subset_size} ***")
        # verify_subset now just returns raw counts based on heuristic
        current_counts, _, _ = verify_subset(
            all_questions, current_subset_names, target_count
        )
        last_verified_counts = current_counts  # Store the latest counts

        # Check if target is met for relevant types ONLY
        all_relevant_types_met = True
        for q_type, count in current_counts.items():
            if (
                q_type not in heuristic_ignore_types
            ):  # Only check types we care about for the heuristic
                loguru.logger.debug(
                    f"Checking {q_type}: Count={count}, Target={target_count}"
                )
                if count < target_count:
                    all_relevant_types_met = False
                    loguru.logger.debug(f"  -> Target NOT MET for {q_type}")
                    break  # No need to check further for this subset size
            else:
                loguru.logger.debug(
                    f"Ignoring {q_type} for sufficiency check (heuristic unreliable). Count={count}"
                )

        if all_relevant_types_met:
            loguru.logger.success(
                f"Found a sufficient subset (for heuristic-relevant types) with {current_subset_size} entities."
            )
            pbar.close()
            # Return the names and the counts from this successful verification
            return all_entity_names[:current_subset_size], current_counts

        # If not met, increase subset size, unless already at max
        if current_subset_size == total_unique_entities:
            break  # Avoid infinite loop if increment is 0 and target never met

        current_subset_size += increment_count
        current_subset_size = min(current_subset_size, total_unique_entities)

    pbar.close()
    loguru.logger.warning(
        "Could not find a subset meeting the target for all *heuristic-relevant* types, even using all *filtered* entities."
    )
    loguru.logger.warning(
        "This might indicate issues with the heuristic or the data for other types."
    )
    # Return all filtered entities as the best effort, along with the last calculated counts
    return all_entity_names, last_verified_counts


def main():
    parser = argparse.ArgumentParser(
        description="Find and verify an entity subset for recipe question generation."
    )
    parser.add_argument(
        "input_folder",
        help="Folder containing the output files from the previous run (all_entities.json, all_questions.json)",
    )
    parser.add_argument(
        "output_folder",
        help="Folder where the proposed/final entity list and report will be saved.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=100,
        help="Target number of questions per analysis type.",
    )
    parser.add_argument(
        "--initial-perc",
        type=float,
        default=0.01,
        help="Initial percentage of top entities to try (e.g., 0.1 for 10%).",
    )
    parser.add_argument(
        "--increment-perc",
        type=float,
        default=0.01,
        help="Percentage of entities to add in each iteration (e.g., 0.05 for 5%).",
    )
    parser.add_argument(
        "--verify-manual-list",
        type=str,
        default=None,
        metavar="MANUAL_LIST_FILE",
        help="Path to a manually edited entity list file (one entity per line) to verify.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level.",
    )

    args = parser.parse_args()

    # Configure output folder and logging
    # Delete output folder if it exists
    if not args.verify_manual_list:
        if os.path.exists(args.output_folder):
            rmtree(args.output_folder)
            loguru.logger.info(f"Deleting existing output folder: {args.output_folder}")

        loguru.logger.info(f"Creating new output folder: {args.output_folder}")
    loguru.logger.info(f"Output folder: {args.output_folder}")
    os.makedirs(args.output_folder, exist_ok=True)

    loguru.logger.remove()
    log_file_path = os.path.join(args.output_folder, "subset_finding.log")
    loguru.logger.add(log_file_path, level=args.log_level)
    loguru.logger.add(
        lambda msg: tqdm.write(msg, end=""), level="INFO"
    )  # Also print INFO+ to console

    # Mode 1: Find and Propose Subset
    if args.verify_manual_list is None:
        loguru.logger.info("*** Mode: Finding and Proposing Entity Subset ***")

        # Load required data
        entities_path = os.path.join(args.input_folder, "all_entities.json")
        questions_path = os.path.join(args.input_folder, "all_questions.json")

        entity_data = load_json(entities_path)
        all_questions = load_json(questions_path)

        if not entity_data or not all_questions:
            loguru.logger.error("Failed to load necessary input files. Exiting.")
            return

        original_entity_counts = entity_data.get("entity_counts", {})
        if not original_entity_counts:
            loguru.logger.error(
                "No 'entity_counts' found in all_entities.json. Exiting."
            )
            return

        # Filter the loaded entities
        loguru.logger.info(
            f"Filtering {len(original_entity_counts)} entities based on rules (e.g., removing those starting with verbs)..."
        )
        filtered_entity_counts = {
            name: count
            for name, count in original_entity_counts.items()
            if is_likely_valid_entity(name)
        }
        loguru.logger.info(
            f"Retained {len(filtered_entity_counts)} entities after filtering."
        )
        if not filtered_entity_counts:
            loguru.logger.error(
                "No entities remained after filtering. Cannot proceed. Check filtering rules or input data."
            )
            return

        # --- Modified: Pass filtered_entity_counts ---
        proposed_subset_names, final_heuristic_counts = find_entity_subset(
            filtered_entity_counts,
            all_questions,
            args.target_count,
            args.initial_perc,
            args.increment_perc,
        )

        if proposed_subset_names:
            # Save the proposed subset for manual review
            proposed_list_path = os.path.join(
                args.output_folder, "proposed_entity_subset.txt"
            )
            # Sort alphabetically for easier review
            proposed_subset_names.sort()
            with open(proposed_list_path, "w", encoding="utf-8") as f:
                for entity in proposed_subset_names:
                    f.write(f"{entity}\n")

            loguru.logger.success(
                f"Proposed entity subset saved to: {proposed_list_path}"
            )
            loguru.logger.info(f"Proposed subset size: {len(proposed_subset_names)}")
            loguru.logger.info(
                "Heuristic verification counts with this proposed subset:"
            )
            loguru.logger.info("(Counts based on entity mentions in question text)")
            if final_heuristic_counts:
                for q_type in sorted(final_heuristic_counts.keys()):
                    count = final_heuristic_counts[q_type]
                    ignored_note = (
                        "(Ignored for sufficiency check)"
                        if q_type in {"Interval Analysis", "Concurrency Analysis"}
                        else ""
                    )
                    target_met_note = ""
                    if q_type not in {"Interval Analysis", "Concurrency Analysis"}:
                        target_met_note = f"(Target: {args.target_count} - {'MET' if count >= args.target_count else 'NOT MET'})"

                    loguru.logger.info(
                        f"- {q_type}: {count} {target_met_note} {ignored_note}"
                    )
            else:
                loguru.logger.warning("Could not retrieve final heuristic counts.")

            loguru.logger.info("\n*** Explanation & Action Required ***")
            loguru.logger.info(
                "The script filtered the initial entity list (e.g., removing items starting with verbs) and then found a subset where *most* question types meet the target count based on a simple text-mention heuristic."
            )
            loguru.logger.info(
                "Types like 'Interval Analysis' and 'Concurrency Analysis' were ignored during this check because the heuristic is unreliable for them (questions often don't mention specific entities)."
            )
            loguru.logger.info(
                "However, your *original* run successfully generated >100 questions for these types, indicating the underlying recipe data supports them."
            )
            loguru.logger.info(
                "Therefore, this proposed subset is likely sufficient for the *actual* generation process."
            )
            loguru.logger.info(f"1. Please review the file: {proposed_list_path}")
            loguru.logger.info(
                "2. Edit the file: Remove any *remaining* entities you consider incorrect, noisy, or irrelevant (e.g., units, generic terms like 'mixture' if unwanted)."
            )
            loguru.logger.info(
                "3. Save the edited file (you can overwrite it or save with a new name)."
            )
            loguru.logger.info(
                "4. Re-run this script using the --verify-manual-list flag, pointing to your edited file:"
            )
            loguru.logger.info(
                f"   uv run python code/diminish.py {args.input_folder} {args.output_folder} --verify-manual-list YOUR_EDITED_LIST.txt --target-count {args.target_count}"
            )
            loguru.logger.info("***")
        else:
            loguru.logger.error(
                "Failed to find a suitable entity subset even with filtered entities. Check logs for details."
            )

    # Mode 2: Verify Manual List
    else:
        loguru.logger.info("*** Mode: Verifying Manually Edited Entity Subset ***")
        manual_list_path = args.verify_manual_list

        if not os.path.exists(manual_list_path):
            loguru.logger.error(
                f"Manual list file not found: {manual_list_path}. Exiting."
            )
            return

        # Load the manually edited list
        try:
            with open(manual_list_path, "r", encoding="utf-8") as f:
                # Read lines, strip whitespace, filter out empty lines
                manual_subset = set(line.strip() for line in f if line.strip())
            loguru.logger.info(
                f"Loaded {len(manual_subset)} entities from manual list: {manual_list_path}"
            )
        except Exception as e:
            loguru.logger.error(f"Error reading manual list file: {e}")
            return

        if not manual_subset:
            loguru.logger.error("Manual list is empty. Cannot verify. Exiting.")
            return

        # Load questions data
        questions_path = os.path.join(args.input_folder, "all_questions.json")
        all_questions = load_json(questions_path)
        if not all_questions:
            loguru.logger.error("Failed to load questions data. Exiting.")
            return

        # Perform verification using the heuristic
        loguru.logger.info(
            "Verifying the manually provided subset using the entity-mention heuristic..."
        )
        # The 'all_met' flag from verify_subset is not used here directly, we check counts below
        final_heuristic_counts, _, relevant_indices = verify_subset(
            all_questions, manual_subset, args.target_count
        )

        # Determine if all *relevant* types met the target according to the heuristic
        all_relevant_heuristic_targets_met = True
        heuristic_ignore_types = {"Interval Analysis", "Concurrency Analysis"}
        for q_type, count in final_heuristic_counts.items():
            if q_type not in heuristic_ignore_types:
                if count < args.target_count:
                    all_relevant_heuristic_targets_met = False
                    break

        # Save the final report
        report_path = os.path.join(args.output_folder, "final_verification_report.json")
        report_data = {
            "manual_list_path": manual_list_path,
            "final_entity_subset_size": len(manual_subset),
            "target_questions_per_type": args.target_count,
            "heuristic_verification_counts": final_heuristic_counts,
            "all_relevant_heuristic_targets_met": all_relevant_heuristic_targets_met,
            "heuristic_details": "Counts based on case-insensitive, whole-word substring match of entities in question text.",
            "heuristic_ignored_types": list(heuristic_ignore_types),
            "relevant_question_count_heuristic": len(relevant_indices),
            "confidence_note": "Confidence that targets for ignored types (e.g., Interval Analysis) can be met relies on the success of the original full data run, not this heuristic count.",
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        loguru.logger.success(f"Final verification report saved to: {report_path}")
        loguru.logger.info("*** Final Heuristic Verification Results ***")
        loguru.logger.info(
            f"Using {len(manual_subset)} entities from {manual_list_path}:"
        )
        for q_type in sorted(final_heuristic_counts.keys()):
            count = final_heuristic_counts[q_type]
            status = ""
            if q_type not in heuristic_ignore_types:
                status = f"(Target: {args.target_count} - {'MET' if count >= args.target_count else 'NOT MET'})"
            else:
                status = "(Heuristic unreliable/ignored for sufficiency)"
            loguru.logger.info(f"- {q_type}: {count} {status}")

        if all_relevant_heuristic_targets_met:
            loguru.logger.success(
                "All heuristic targets for *relevant* types were met with the provided entity list."
            )
            loguru.logger.info(
                "Assume targets for ignored types can also be met based on original run."
            )
        else:
            loguru.logger.warning(
                "One or more heuristic targets for *relevant* types were NOT MET."
            )
            loguru.logger.warning(
                "You may need to add back entities or accept lower potential counts for those types."
            )

        # Save the final list used for verification
        final_list_path = os.path.join(args.output_folder, "final_entity_subset.txt")
        sorted_manual_list = sorted(list(manual_subset))
        with open(final_list_path, "w", encoding="utf-8") as f:
            for entity in sorted_manual_list:
                f.write(f"{entity}\n")
        loguru.logger.info(
            f"Final entity list used for verification saved to: {final_list_path}"
        )


if __name__ == "__main__":
    main()
