# Automated Question Generation for Procedural Text Analysis in Recipes

_Does LLMs understanding of data-flow change with content representation of recipes?_

## Overview

This project provides a framework for analyzing procedural texts, such as recipes, by applying concepts borrowed from software program analysis. The goal is to understand the underlying structure, dependencies, and flow of these texts and to automatically generate questions that test this understanding.

The core idea is to represent a procedural text as a structured object, extract key entities (ingredients, tools, intermediate products) and their relationships, model the dependencies between steps, and then apply various analyses (like reaching definitions, liveness, concurrency) to probe the procedure's properties.

## TL;DR Workflow

Starting from raw recipe data, this codebase allows you to generate structured datasets suitable for training or evaluating models on procedural understanding. **Check the respective files for command line arguments.**

```bash
# Step 1: Initial Processing & Question Generation (Oversampling)
# Processes raw recipes, extracts entities/dependencies, generates a large pool of questions for all analysis types.
# Input: Folder with recipe JSONs (e.g., containing 'title', 'instructions_list')
# Output: Folder with processed JSONs per recipe (entities, questions), plus aggregate files (all_entities.json, all_questions.json, etc.)
uv run python code/procedural_text_process.py /path/to/raw_recipes /path/to/initial_output

# Step 2a: Propose Curated Entity Subset
# Analyzes entity frequency and question coverage (heuristic) to propose a smaller, likely sufficient list of entities.
# Input: Folder from Step 1 (e.g., /path/to/initial_output)
# Output: Folder with proposed list (proposed_entity_subset.txt) and analysis logs.
uv run python code/diminish.py /path/to/initial_output /path/to/subset_analysis

# ***** MANUAL STEP *****
# Review and edit the proposed_entity_subset.txt file.
# Remove incorrect/noisy entities. Save the cleaned list (e.g., as final_entity_subset.txt).

# Step 2b: Verify Curated Entity Subset
# Verifies the manually edited entity list against the question pool using the heuristic.
# Input: Folder from Step 1, Path to manually edited entity list.
# Output: Folder with verification report (final_verification_report.json) and the final list used (final_entity_subset.txt).
uv run python code/diminish.py /path/to/initial_output /path/to/subset_analysis --verify-manual-list /path/to/subset_analysis/final_entity_subset.txt

# Step 3: Generate Initial Final Datasets (CSV)
# Filters the original question pool based on the curated entity list and recipe relevance, selects the target number of questions per type, and generates initial final CSVs.
# Input: Folder from Step 1, Path to the final curated entity list (from Step 3).
# Output: Folder containing one CSV per analysis type (e.g., Reaching_Definitions_questions.csv).
uv run python code/generate_final_dataset.py /path/to/initial_output /path/to/subset_analysis/final_entity_subset.txt /path/to/final_csv_datasets --target-count 100

# ***** MANUAL ANNOTATION STEP *****
# 1. Make 8 copies of the initial_final_csv_datasets folder (e.g., to /path/to/annotated_csv_datasets).
# 2. For each CSV file in the copied folder (e.g., Reaching_Definitions_questions.csv):
#    a. Rename it to include "_annotated" (e.g., recipe_reaching_definitions_annotated.csv).
#    b. Add a third column named `corrected_answer`.
#    c. For each question:
#       - If the original `answer` is correct, leave `corrected_answer` blank or copy the `answer`.
#       - If the original `answer` is incorrect, put the correct answer in the `corrected_answer` column.

# Step 4: Finalize Corrected Datasets (Interactive)
# Processes the annotated CSVs. For incorrect questions, it attempts to generate and interactively verify replacements.
# Fills any gaps to reach the target count per type, also with interactive verification.
# Input: Folder with annotated CSVs, Folder from Step 1 (initial_output), Path to all_questions.json (from Step 1).
# Output: Folder with final, manually verified CSVs (e.g., /path/to/final_corrected_datasets).
uv run python code/finalize_corrected_dataset.py /path/to/annotated_csv_datasets /path/to/initial_output /path/to/initial_output/all_questions.json /path/to/final_corrected_datasets /path/to/final_datasets_for_release

# Step 5 (Optional): Post-Process Final Datasets (Renaming/Formatting & Stats)
# Renames CSV files and columns to a specific required format if needed, converts boolean answers to Yes/No, and generates dataset statistics.
# Input: Folder from Step 5 (e.g., /path/to/final_corrected_datasets).
# Output: Folder with potentially renamed/reformatted CSVs and dataset_statistics.json (e.g., /path/to/final_datasets_for_release).
uv run python code/post_processing.py /path/to/final_corrected_datasets /path/to/final_datasets_for_release
```

## Data-Flow Analyses Analogies for Recipes

The core of this project involves applying concepts from static program analysis to procedural text. Here's how each analysis type is interpreted in the context of recipes and how the code generates corresponding questions:

- **Available Expressions**

  - **Recipe Analogy:** Checks if the result of a previous step (an 'expression', like 'creamed butter and sugar') is still valid and usable ('available') at a later point, meaning its components haven't been altered since it was created.
  - **Code Implementation:** The code identifies an entity (e.g., `creamed mixture`) defined at step `A`. It picks a later step `B` and checks if any step between `A` and `B` redefines the entity. If not, it asks: `"Is {creamed mixture} from Step {A+1} still available in Step {B+1}?"` The ground truth is `True` if it remains available (no intermediate redefinition).

- **Concurrency Analysis**

  - **Recipe Analogy:** Determines if two distinct steps can be performed simultaneously or in parallel without interfering with each other. This depends on whether they require the same limited resource (like the oven, a specific pan) or if one step depends on the output (data) of the other.
  - **Code Implementation:** The code selects two steps, `A` and `B`. It checks for conflicts using the `step_dependencies` graph (including dataflow edges) and resource analysis (`_get_resources_used` identifies exclusive resources like 'oven', 'mixer', 'skillet' based on verbs and entity mentions). Conflicts arise from:
    1.  Path Dependency: A path exists from `A` to `B` or `B` to `A` in the dependency graph.
    2.  Data Conflict: Read/Write (RAW), Write/Read (WAR), or Write/Write (WAW) conflicts on the same entity between the two steps.
    3.  Resource Conflict: Both steps require the same exclusive resource (e.g., both need the 'oven').
        The generator extracts primary actions (verbs) from the steps and asks: `"Can we {action_A} (Step {A+1}) and {action_B} (Step {B+1}) at the same time?"` The ground truth is `True` only if _none_ of the conflicts above are detected between the _full steps_. A filtering mechanism attempts to avoid questions where simple actions seem concurrent but underlying dependencies prevent it.

- **Interval Analysis**

  - **Recipe Analogy:** Focuses on numerical constraints, particularly time durations specified in steps (e.g., "bake for 30-35 minutes", "simmer for 10 minutes").
  - **Code Implementation:** The code parses steps looking for patterns indicating time ranges or exact durations (e.g., `(\d+)-(\d+) minutes`, `(\d+) hours`). If found for step `k`, it stores this interval. The generator asks: `"What is the time interval applied at Step {k+1}?"` The ground truth is the extracted interval string (e.g., `"30-35 minutes"`).

- **Live Variable Analysis**

  - **Recipe Analogy:** Determines if an ingredient, tool, or intermediate product ('variable') is still needed ('live') for any _future_ steps after a certain point. If something is no longer live, it could theoretically be discarded or put away. For example, are the banana peels needed after the mashing step? (Likely no). Is the oven needed after preheating but before baking? (Yes).
  - **Code Implementation:** The code selects an entity (e.g., `banana peels`) and a step index `k`. It checks if there exists _any_ step `j > k` where this entity is marked as 'used'. It generates a question: `"Is {banana peels} live after Step {k+1}?"` The ground truth is `True` if a future use exists, and `False` otherwise (considering potential consumption).

- **Reaching Definitions**

  - **Recipe Analogy:** Tracks whether an ingredient or intermediate product created ('defined') in one step is the _same one_ being used in a later step, without being modified or replaced in between. For example, is the 'mashed banana' used in the final mixing step the same 'mashed banana' created earlier?
  - **Code Implementation:** The `QuestionGenerator` finds an entity (e.g., `banana`) defined at step `A` and used at step `B` (`A < B`). It verifies that no other step between `A` and `B` also defines (`banana`). It then generates a question like: `"In Step {B+1}, is the {banana} from Step {A+1} being used?"` The ground truth is `True` if no intermediate redefinition was found.

- **Taint Analysis**

  - **Recipe Analogy:** Tracks potentially hazardous ingredients (e.g., raw eggs, raw meat - 'tainted' input) to ensure they undergo a 'sanitization' step (usually cooking) before the procedure implicitly ends (consumption). It checks if using a raw ingredient introduces a safety concern to the _final_ dish.
  - **Code Implementation:** The code identifies entities deemed potentially unsafe (e.g., "egg"). It tracks the state of these entities, marking them as 'cooked' if a cooking verb acts on them. It looks for steps where the unsafe entity is _used_ while still in a 'raw' state. The question asks: `"Does using {raw egg} in Step {X+1} introduce a potential safety concern to the recipe?"` The ground truth is `True` only if the entity is identified as unsafe AND is _never_ marked as 'cooked' throughout the entire recipe. If it's used raw but cooked later, the answer is `False`.

- **Type-State Analysis**

  - **Recipe Analogy:** Ensures ingredients or objects are in the correct state before an action is performed. You need to peel a banana _before_ you mash it; you need to open a can _before_ you drain it.
  - **Code Implementation:** The code identifies a dependency where step `A` defines or prepares an entity (e.g., `peeled banana`) and step `B` uses it (e.g., `mash the banana`). It generates a question testing the prerequisite: `"If we skip Step {A+1} (peeling), is it still valid to mash the {banana} in Step {B+1}?"` The ground truth is typically `False`, as skipping the prerequisite invalidates the subsequent action.

- **Very Busy Expressions**

  - **Recipe Analogy:** Identifies ingredients or intermediate products that are created in one step and then used in _multiple_ subsequent steps before being changed. For instance, if you prepare a spice mix early on and use it to season both the meat and the vegetables later, that spice mix is 'very busy'.
  - **Code Implementation:** The code looks for an entity (e.g., `spice mix`) defined at step `A`. It then finds multiple steps `B`, `C`, ... (`A < B`, `A < C`) where the entity is used. Crucially, it checks that there's no redefinition of the entity between step `A` and _each_ of these future uses. It generates questions like: `"Is {spice mix} from Step {A+1} used in multiple future steps without being redefined?"` The ground truth is `True` if such a pattern is found.
