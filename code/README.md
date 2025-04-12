# Automated Question Generation for Procedural Text Analysis

This codebase provides a framework for analyzing procedural texts (like recipes or instructions) using concepts from program analysis. It extracts entities, their relationships, and dependencies between steps, then generates questions that test understanding of the procedural flow.

## Core Components

### 1. `Entity` Class
- Represents objects or ingredients mentioned in the text
- Tracks where entities are introduced, used, defined, and consumed
- Maintains state information across different steps

### 2. `ProceduralText` Class
- Main class that parses and represents the procedural text
- Contains:
  - The goal of the procedure
  - A list of steps
  - Extracted entities
  - Two graphs:
    - `step_dependencies`: Shows dependencies between steps
    - `entity_flow_graph`: Tracks how entities flow between steps

### 3. `QuestionGenerator` Class
- Generates questions based on program analysis concepts:
  - **Reaching Definitions**: Tracks if entities defined in one step reach later steps
  - **Very Busy Expressions**: Identifies entities used multiple times after definition
  - **Available Expressions**: Checks if entities remain valid across steps
  - **Live Variable Analysis**: Determines if entities are needed in future steps
  - **Interval Analysis**: Examines time constraints in steps
  - **Type-State Analysis**: Verifies entities are used in valid states
  - **Taint Analysis**: Identifies potential contamination or safety concerns
  - **Concurrency Analysis**: Determines which steps can run in parallel

## How It Works

1. The code parses each step using spaCy NLP to extract entities and their roles
2. It builds dependency graphs to represent relationships between steps and entities
3. For each analysis type, it identifies relevant patterns in the procedural text
4. It generates natural language questions with ground truth answers
5. These questions test understanding of the procedural flow and entity relationships


## Running

An example of processing up to 100 recipes from the input folder, save the results to the output folder, and log at the INFO level.

```bash
uv run python recipe_processor.py /path/to/recipes /path/to/output --limit 100 --log-level INFO
```
