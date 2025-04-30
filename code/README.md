# Automated Question Generation for Procedural Text Analysis

## Procedural Text Processing

This codebase provides a framework for analyzing procedural texts (like recipes or instructions) using concepts from program analysis. It extracts entities, their relationships, and dependencies between steps, then generates questions that test understanding of the procedural flow.

### Core Components

1. `Entity` Class

   - Represents objects or ingredients mentioned in the text
   - Tracks where entities are introduced, used, defined, and consumed
   - Maintains state information across different steps

2. `ProceduralText` Class

   - Main class that parses and represents the procedural text
   - Contains:
     - The goal of the procedure
     - A list of steps
     - Extracted entities
     - Two graphs:
       - `step_dependencies`: Shows dependencies between steps
       - `entity_flow_graph`: Tracks how entities flow between steps

3. `QuestionGenerator` Class

   - Generates questions based on program analysis concepts:
     - **Reaching Definitions**: Tracks if entities defined in one step reach later steps
     - **Very Busy Expressions**: Identifies entities used multiple times after definition
     - **Available Expressions**: Checks if entities remain valid across steps
     - **Live Variable Analysis**: Determines if entities are needed in future steps
     - **Interval Analysis**: Examines time constraints in steps
     - **Type-State Analysis**: Verifies entities are used in valid states
     - **Taint Analysis**: Identifies potential contamination or safety concerns
     - **Concurrency Analysis**: Determines which steps can run in parallel

### How It Works

1. The code parses each step using spaCy NLP to extract entities and their roles
2. It builds dependency graphs to represent relationships between steps and entities
3. For each analysis type, it identifies relevant patterns in the procedural text
4. It generates natural language questions with ground truth answers
5. These questions test understanding of the procedural flow and entity relationships

### Running

An example of processing up to 100 recipes from the input folder, save the results to the output folder, and log at the INFO level.

```bash
uv run python code/procedural_text_process.py /path/to/recipes /path/to/output --limit 100 --log-level INFO
```

## Diminishing

This script will take the outputs from the `procedural_text_processing.py`'s run (`all_entities.json`, `all_questions.json`) and help find a minimal subset of _frequently occurring_ entities that likely contributed to generating at least 100 questions per category. It includes a step for manual review.

### How It Works

1.  Load Data: Reads the entity frequencies (`all_entities.json`) and all the generated questions (`all_questions.json`).
2.  Sort Entities: Sorts entities by frequency in descending order.
3.  Iterative Selection: Starts with a small percentage of the most frequent entities.
4.  Verification (Heuristic): Checks how many questions _mention_ at least one entity from the current subset for each analysis type. **Important:** This is a heuristic. A question might be generated based on relationships _between_ entities, or lack thereof (like concurrency), and might not explicitly mention the _key_ entity in its text. However, given the 5x oversampling, it's a strong indicator. We'll use case-insensitive matching with word boundaries (`\b`) to improve accuracy.
5.  Expand Subset: If the target count (100) isn't met for all categories, it adds the next batch of most frequent entities and re-verifies.
6.  Propose Subset: Once a subset is found that meets the target counts (based on the heuristic), it saves this list (`proposed_entity_subset.txt`) for your review.
7.  Manual Review: You manually edit `proposed_entity_subset.txt`, removing any entities you deem incorrect or too noisy.
8.  Final Verification: You re-run the script with a specific flag (`--verify-manual-list`) pointing to your edited file. The script then calculates the final question counts based _only_ on your curated list.

### How To Use

1.  Ensure Inputs: Make sure you have the `all_entities.json` and `all_questions.json` files from your original run in a known input folder (here `path/to/output`).
2.  Run - Step 1 (Propose Subset):
    ```bash
    uv run python code/diminish.py outs/outs_1/ outs/outs_1_subset_analysis/
    ```
    - The script will run and create `subset_analysis/proposed_entity_subset.txt` and `subset_analysis/subset_finding.log` within `outs_1_sub/`.
3.  Manual Review:
    - Open `subset_analysis/proposed_entity_subset.txt`.
    - Delete any lines containing entities that seem like parsing errors (e.g., "medium heat", "minutes", single letters, verbs, etc.) or are too generic ("mixture", "batter" might be okay depending on your goal, but "ingredients" probably isn't).
    - Save the edited file. You can overwrite it or save it as, for example, `subset_analysis/manual_entity_subset.txt`.
4.  Run - Step 2 (Verify Manual List):
    ```bash
    python code/diminish.py outs/outs_1 outs/outs_1_subset_analysis --verify-manual-list outs/outs_1_subset_analysis/manual_entity_subset.txt
    ```
    - Make sure the path after `--verify-manual-list` points to the _edited_ file.
    - The script will now use _only_ the entities in your edited list to count the relevant questions.
    - It will output the final counts and save `subset_analysis/final_verification_report.json` and `subset_analysis/final_entity_subset.txt`.
5.  Analyze Results: Check the `final_verification_report.json` and the console output. If all counts are >= 100, your manually curated list is sufficient according to the heuristic. If not, you might need to be less strict in your manual filtering or accept the lower counts.

## Final Dataset Generation

After the previous two script, we have a curated list of trusted entities (e.g. `final_entity_subset.txt`). Now, we also want to use the _original_ pool of generated questions (`all_questions.json` or equivalent data from the processed recipe files) and select exactly `target_count` (e.g., 100) questions for each analysis category. The selection should prioritize questions coming from recipes that involve your curated entities and ideally contribute questions to multiple analysis types. The final output should be separate CSV files per category, each containing the prompt (recipe + question) and the answer.

### Script Logic

1.  **Load Inputs:**
    - The curated entity list (`final_entity_subset.txt`).
    - All _processed_ recipe JSON files from the _original_ output folder (e.g., `outs/outs_6/`). These contain the recipe text, identified entities, and the originally generated questions for that recipe. This is better than `all_questions.json` as it directly links questions to recipes and their _identified_ entities.
    - The target count per category.
2.  **Filter Recipes and Questions:**
    - Iterate through each processed recipe JSON.
    - Check if the recipe's identified entities (`"entities"` keys in the JSON) have any overlap with the curated entity list.
    - If there's an overlap, consider this recipe "relevant". Collect all questions associated with this relevant recipe. Store them along with their type and source recipe title.
3.  **Score Recipes:**
    - For each relevant recipe, count how many _different_ analysis types its questions cover. This is the recipe's score. Higher scores mean the recipe is more diverse in terms of the analysis it triggers.
4.  **Select Questions:**
    - Group all collected questions by their analysis type.
    - For each analysis type:
      - Get the list of questions for this type.
      - Sort this list:
        - Primarily by the score of their source recipe (descending).
        - Secondarily by recipe title (alphabetical, for determinism).
        - Tertiarily by the question text (alphabetical, for determinism).
      - Select the top `target_count` questions from the sorted list. If fewer than `target_count` are available, select all available ones and log a warning.
5.  **Format and Output:**
    - For each selected question:
      - Retrieve the full instructions for its source recipe.
      - Format the prompt string (Goal, Steps, Question).
      - Store the prompt and the answer.
    - Create a pandas DataFrame for each analysis type (`prompt`, `answer`).
    - Save each DataFrame to a separate CSV file in the specified output folder.

### How to Use

1.  **Ensure Inputs:**
    - Have the folder containing the _original_ processed recipe JSONs (e.g., `outs/outs_6/`).
    - Have your final curated entity list (e.g., `outs/outs_6_sub/final_entity_subset.txt`).
    - Decide on an output folder for the CSVs (e.g., `final_csv_datasets`).
2.  **Run the Script:**
    ```bash
    uv run python code/generate_final_dataset.py outs/outs_1/ outs/outs_1_subset_analysis/final_entity_subset.txt outs/outs_1_subsetted_final_datasets
    ```
3.  **Check Output:** The script will create the `final_csv_datasets` folder (or whatever you named it) and populate it with CSV files, one for each analysis category (e.g., `Reaching_Definitions_questions.csv`, `Taint_Analysis_questions.csv`, etc.). Each CSV will contain exactly 100 rows (or fewer if not enough eligible questions were found for that category), with `prompt` and `answer` columns.
