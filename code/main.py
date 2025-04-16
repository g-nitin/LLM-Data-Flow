import argparse
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pprint import pformat
from shutil import rmtree
from typing import Any, Dict, List, Optional, Set, Tuple

import loguru
import matplotlib.pyplot as plt
import networkx as nx
import spacy
from tqdm import tqdm

# Load spaCy model for NLP processing
spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")  # Smaller model for faster processing
# nlp = spacy.load("en_core_web_trf")  # Larger model for better accuracy


@dataclass
class Entity:
    name: str
    step_introduced: int
    # step_id -> state
    states: Dict = field(default_factory=dict)
    # steps where this entity is used
    used_in: Set = field(default_factory=set)
    # steps where this entity is defined/modified
    defined_in: Set = field(default_factory=set)
    # steps where this entity is consumed
    consumed_in: Set = field(default_factory=set)


class ProceduralText:
    def __init__(self, goal: str, steps: List[str]):
        self.goal = goal
        self.steps = steps
        self.entities = {}  # name -> Entity
        self.step_dependencies = nx.DiGraph()  # Directed graph for step dependencies
        self.entity_flow_graph = nx.DiGraph()  # Graph for entity flow

        # Initialize step dependencies graph
        for i in range(len(steps)):
            self.step_dependencies.add_node(i)
            if i > 0:
                self.step_dependencies.add_edge(
                    i - 1, i
                )  # Default sequential dependency

        # Parse steps to extract entities and their relationships
        self._parse_steps()
        self.build_entity_flow_graph()

        # Log all variables
        loguru.logger.info(
            f"\n{len(self.steps)} steps:\n{pformat(list(zip(range(1, len(self.steps)), self.steps)))}"
            f"\nParsed {len(self.entities)} entities:\n{pformat(self.entities)}"
            f"\nStep dependencies:\n{pformat(list(self.step_dependencies.edges()))}"
        )

    def _extract_entities_from_step(self, step_idx, step_text):
        """
        Extract entities from a step using NLP techniques with refined filtering for recipes.
        Focuses on ingredients and key intermediate products, filtering out containers,
        tools, generic terms, locations, and certain actions/qualities.
        """
        doc = nlp(step_text)

        # Filter Lists
        measurement_units = {
            "cup",
            "cups",
            "tablespoon",
            "tablespoons",
            "tbsp",
            "teaspoon",
            "teaspoons",
            "tsp",
            "ounce",
            "ounces",
            "oz",
            "pound",
            "pounds",
            "lb",
            "gram",
            "grams",
            "g",
            "kilogram",
            "kilograms",
            "kg",
            "milliliter",
            "milliliters",
            "ml",
            "liter",
            "liters",
            "l",
            "pint",
            "pints",
            "quart",
            "quarts",
            "gallon",
            "gallons",
            "inch",
            "inches",
            "minute",
            "minutes",
            "min",
            "hour",
            "hours",
            "second",
            "seconds",
            "sec",
            "degree",
            "degrees",
            "fahrenheit",
            "celsius",
            "f",
            "c",
        }

        # More structured generic terms
        containers_tools = {
            "bowl",
            "pan",
            "skillet",
            "pot",
            "saucepan",
            "dish",
            "sheet",
            "rack",
            "plate",
            "spoon",
            "fork",
            "knife",
            "oven",
            "whisk",
            "spatula",
            "grater",
            "peeler",
            "board",
            "container",
            "jar",
            "bottle",
            "processor",
            "blender",
            "mixer",
            "cooker",
            "grill",
            "microwave",
            "utensil",
            "tongs",
            "ladle",
            "colander",
            "sieve",
            "mandoline",
            "thermometer",
            "timer",
            "machine",
            "bag",
            "wrap",
            "foil",
            "paper",
            "towel",
            "lid",
            "cover",
        }

        locations_qualities = {
            "side",
            "top",
            "bottom",
            "middle",
            "center",
            "level",
            "surface",
            "edge",
            "point",
            "heat",
            "temperature",
            "medium",
            "high",
            "low",
            "small",
            "large",
            "hot",
            "cold",
            "warm",
            "cool",
            "room",
            "color",
            "consistency",
            "aroma",
            "flavor",
            "texture",
            "appearance",
        }

        abstract_generic = {
            "step",
            "time",
            "rest",
            "half",
            "third",
            "quarter",
            "part",
            "piece",
            "pieces",
            "amount",
            "bit",
            "way",
            "method",
            "technique",
            "process",
            "result",
            "use",
            "need",
            "addition",
            "removal",
            "remainder",
            "balance",
            "recipe",
            "instructions",
            "directions",
            "return",
            "place",
            "set",
            "transfer",
            "attention",
            "care",
            "ingredients",
            "mixture",
            "batter",
            "liquid",
            "solids",
            "everything",
            "anything",
            "something",
            "nothing",
            "water",  # Water is often implicit or tap water, less often a specific 'ingredient' unless qualified (e.g., 'coconut water')
        }

        # Combine all terms to exclude by direct match
        stop_entity_lemmas = (
            measurement_units.union(containers_tools)
            .union(locations_qualities)
            .union(abstract_generic)
        )

        # Patterns for filtering (e.g., numeric values, specific phrases)
        filter_patterns = [
            r"^\d+(\.\d+)?$",  # Pure numbers
            r"^\d+/\d+$",  # Fractions
            r"medium heat",
            r"high heat",
            r"low heat",  # Specific heat levels often not entities
            r"room temperature",
            r"baking soda",
            r"baking powder",  # Keep these as exceptions if needed later, but filter simple 'baking'/'powder'
        ]
        # Pre-compile regex for efficiency
        compiled_filter_patterns = [
            re.compile(p, re.IGNORECASE) for p in filter_patterns
        ]

        potential_entities = {}  # Use dict to store lemma -> original text mapping, helps deduplicate

        # --- Process Noun Chunks ---
        for chunk in doc.noun_chunks:
            # Clean the chunk text: lowercase, remove leading/trailing articles/possessives
            clean_chunk_text = chunk.text.lower()
            clean_chunk_text = re.sub(
                r"^(the|a|an|some|your|my|his|her|its|our|their)\s+",
                "",
                clean_chunk_text,
            ).strip()
            clean_chunk_text = re.sub(
                r"\s+'s$", "", clean_chunk_text
            ).strip()  # Remove trailing 's

            # Use the lemma of the root word for checking against stop lists
            root_lemma = chunk.root.lemma_.lower()

            # --- Apply Filters ---
            # 1. Check exact cleaned text against patterns
            if any(
                pattern.match(clean_chunk_text) for pattern in compiled_filter_patterns
            ):
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' due to pattern match on '{clean_chunk_text}'"
                )
                continue

            # 2. Check root lemma against stop lemmas
            if root_lemma in stop_entity_lemmas:
                # Allow exceptions? e.g., if root is 'pot' but text is 'instant pot'? For now, filter if root matches.
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' because root lemma '{root_lemma}' is in stop list."
                )
                continue

            # 3. Check full cleaned text against stop lemmas (for multi-word generics)
            if clean_chunk_text in stop_entity_lemmas:
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' because full text '{clean_chunk_text}' is in stop list."
                )
                continue

            # 4. Length check
            if len(clean_chunk_text) < 3:
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' due to short length after cleaning: '{clean_chunk_text}'"
                )
                continue

            # 5. Check if root is a verb/adjective (sometimes chunks can be misidentified)
            if chunk.root.pos_ not in {"NOUN", "PROPN"}:
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' because root POS is {chunk.root.pos_}"
                )
                continue

            # --- If passes filters, add it ---
            # Use lemma for deduplication key, store original cleaned text
            entity_lemma = root_lemma  # Or potentially lemmatize the whole clean_chunk_text if more complex
            if entity_lemma not in potential_entities:
                potential_entities[entity_lemma] = clean_chunk_text
                loguru.logger.debug(
                    f"Keeping chunk: '{chunk.text}' -> Cleaned: '{clean_chunk_text}', Lemma: '{entity_lemma}'"
                )

        # --- Process Individual Nouns (Optional: Catch things missed by chunks) ---
        # Be cautious here to avoid re-adding filtered items or parts of chunks
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                token_lemma = token.lemma_.lower()

                # Check if this token's lemma is already covered by a kept chunk
                if token_lemma in potential_entities:
                    continue

                # Check if this token was part of ANY noun chunk (even filtered ones)
                part_of_chunk = False
                for chunk in doc.noun_chunks:
                    if token.i >= chunk.start and token.i < chunk.end:
                        part_of_chunk = True
                        break
                if part_of_chunk:
                    continue

                # Apply similar filters as for chunks
                if token_lemma in stop_entity_lemmas:
                    loguru.logger.debug(
                        f"Skipping individual noun '{token.text}' because lemma '{token_lemma}' is in stop list."
                    )
                    continue
                if len(token_lemma) < 3:
                    loguru.logger.debug(
                        f"Skipping individual noun '{token.text}' due to short length."
                    )
                    continue
                if any(
                    pattern.match(token.text.lower())
                    for pattern in compiled_filter_patterns
                ):
                    loguru.logger.debug(
                        f"Skipping individual noun '{token.text}' due to pattern match."
                    )
                    continue

                # If it passes, add it
                if token_lemma not in potential_entities:
                    potential_entities[token_lemma] = token.text.lower()
                    loguru.logger.debug(
                        f"Keeping individual noun: '{token.text}', Lemma: '{token_lemma}'"
                    )

        # --- Final list of entity names (using the stored original text) ---
        filtered_entities = list(potential_entities.values())
        loguru.logger.info(
            f"Step {step_idx + 1}: Extracted entities: {filtered_entities}"
        )

        # --- Create or update entity objects (using the extracted names) ---
        # (This part remains largely the same as your original code, using the 'filtered_entities' list)
        for entity_name in filtered_entities:
            # Ensure we use the *extracted name* for consistency in the ProceduralText object
            if entity_name not in self.entities:
                self.entities[entity_name] = Entity(entity_name, step_idx)
                # Initial definition assumed when first extracted, refine based on verbs
                self.entities[entity_name].defined_in.add(step_idx)

            # Determine entity's role in this step using verbs
            is_used = False
            is_defined = False
            is_consumed = False

            # Define verb categories (can be refined)
            use_verbs = {
                "add",
                "use",
                "stir",
                "mix",
                "combine",
                "pour",
                "crack",
                "sprinkle",
                "top",
                "garnish",
                "serve",
                "incorporate",
                "fold",
                "whisk",
                "beat",
                "season",
            }
            define_verbs = {
                "create",
                "make",
                "prepare",
                "preheat",
                "measure",
                "warm",
                "peel",
                "mash",
                "chop",
                "dice",
                "slice",
                "mince",
                "cook",
                "bake",
                "boil",
                "simmer",
                "fry",
                "roast",
                "grate",
                "zest",
                "juice",
                "reduce",
            }
            consume_verbs = {
                "eat",
                "consume",
                "finish",
                "remove",
                "discard",
                "drain",
                "strain",
                "reserve",
            }  # Reserve might be definition? Context needed.

            # Check verbs acting on this entity (check if entity name is in the object of the verb)
            for token in doc:
                if token.pos_ == "VERB":
                    # Check direct objects (dobj) or objects of prepositions (pobj) linked to the verb
                    for child in token.children:
                        # Check if the entity name is reasonably contained within the child's text
                        # This check needs to be robust (e.g., "the large onions" should match "onions")
                        if entity_name in child.text.lower():
                            verb_lemma = token.lemma_
                            if verb_lemma in use_verbs:
                                is_used = True
                            elif verb_lemma in define_verbs:
                                is_defined = True
                                # If defined here, remove from initial assumption
                                self.entities[entity_name].defined_in.discard(step_idx)
                                self.entities[entity_name].defined_in.add(
                                    step_idx
                                )  # Re-add to be sure
                            elif verb_lemma in consume_verbs:
                                is_consumed = True

            # Update entity roles for this step
            if is_used:
                self.entities[entity_name].used_in.add(step_idx)
            if is_consumed:
                self.entities[entity_name].consumed_in.add(step_idx)

    def _parse_steps(self):
        """Parse steps to extract entities and their relationships"""
        # Process each step
        for step_idx, step_text in enumerate(self.steps):
            # Extract entities
            self._extract_entities_from_step(step_idx, step_text)

            # Extract temporal dependencies
            # The pattern represents time intervals like "5 to 10 minutes" or "2 hours"
            time_pattern = (
                r"(\d+)[-\s]?to[-\s]?(\d+)\s+(minutes|minute|min|seconds|hours|hour)"
            )
            matches = re.findall(time_pattern, step_text, re.IGNORECASE)
            if matches:
                for match in matches:
                    min_time, max_time, unit = match
                    # Store time interval information with the step
                    self.step_dependencies.nodes[step_idx]["time_interval"] = (
                        int(min_time),
                        int(max_time),
                        unit,
                    )

            # Look for exact time specifications
            # The pattern represents exact time like "5 minutes" or "2 hours"
            exact_time_pattern = r"(\d+)\s+(minutes|minute|min|seconds|hours|hour)"
            exact_matches = re.findall(exact_time_pattern, step_text, re.IGNORECASE)
            if exact_matches and not matches:  # Only if we didn't already find a range
                for match in exact_matches:
                    time_val, unit = match
                    # Store as a time interval with the same min and max
                    self.step_dependencies.nodes[step_idx]["time_interval"] = (
                        int(time_val),
                        int(time_val),
                        unit,
                    )

            # Look for concurrent steps
            if any(
                word in step_text.lower()
                for word in ["while", "during", "meanwhile", "at the same time"]
            ):
                # This indicates potential concurrency
                # For simplicity, we'll just note it in the node attributes
                self.step_dependencies.nodes[step_idx]["concurrent"] = True

                # Try to identify which steps can run concurrently
                if "while" in step_text.lower():
                    # Look for references to other steps
                    for i, other_step in enumerate(self.steps):
                        if i != step_idx and any(
                            entity in step_text.lower()
                            for entity in self.get_entities_at_step(i)
                        ):
                            # Add a concurrency relationship
                            self.step_dependencies.nodes[step_idx][
                                "concurrent_with"
                            ] = i
                            # Remove the direct sequential dependency if it exists
                            if self.step_dependencies.has_edge(i - 1, i):
                                self.step_dependencies.remove_edge(i - 1, i)

    def build_entity_flow_graph(self):
        """Build a graph representing entity flow between steps"""
        for entity_name, entity in self.entities.items():
            # Add nodes for each step where the entity is defined or used
            # all_steps = entity.defined_in.union(entity.used_in)

            # Add edges from definition to usage
            for def_step in entity.defined_in:
                for use_step in entity.used_in:
                    if def_step < use_step:  # Only if definition comes before usage
                        if not self.entity_flow_graph.has_edge(def_step, use_step):
                            self.entity_flow_graph.add_edge(
                                def_step, use_step, entities=set()
                            )
                        self.entity_flow_graph[def_step][use_step]["entities"].add(
                            entity_name
                        )

    def get_entities_at_step(self, step_idx: int) -> Set[str]:
        """Get all entities that are available at a given step"""
        available_entities = set()
        for entity_name, entity in self.entities.items():
            # Entity is available if it was defined in a previous step and not consumed yet
            if any(def_step < step_idx for def_step in entity.defined_in) and not any(
                cons_step < step_idx for cons_step in entity.consumed_in
            ):
                available_entities.add(entity_name)
        return available_entities

    def is_entity_live_after_step(self, entity_name: str, step_idx: int) -> bool:
        """
        Check if an entity is live after a given step.
        An entity is live if it's used in any step after the current one
        """
        if entity_name not in self.entities:
            return False

        entity = self.entities[entity_name]
        return any(use_step > step_idx for use_step in entity.used_in)

    def can_steps_run_concurrently(self, step1: int, step2: int) -> bool:
        """Check if two steps can run concurrently"""
        # Check if either step is marked as concurrent
        step1_concurrent = self.step_dependencies.nodes.get(step1, {}).get(
            "concurrent", False
        )
        step2_concurrent = self.step_dependencies.nodes.get(step2, {}).get(
            "concurrent", False
        )

        if not (step1_concurrent or step2_concurrent):
            # If neither is marked concurrent, check for path dependencies
            if nx.has_path(self.step_dependencies, step1, step2) or nx.has_path(
                self.step_dependencies, step2, step1
            ):
                return False

        # Check for shared mutable entities
        entities1 = set()
        entities2 = set()

        for entity_name, entity in self.entities.items():
            if step1 in entity.defined_in:
                entities1.add(entity_name)
            if step2 in entity.defined_in:
                entities2.add(entity_name)

        # If they modify the same entities, they can't run concurrently
        return len(entities1.intersection(entities2)) == 0


class QuestionGenerator:
    def __init__(self, procedural_text: ProceduralText):
        """
        Initialize the QuestionGenerator with a ProceduralText instance.

        :param procedural_text: An instance of ProceduralText containing the recipe steps and parsed entities.
        """
        self.text = procedural_text

    def generate_reaching_definitions_question(self) -> Tuple[str, Optional[bool]]:
        """
        Generate a question concerning reaching definitions in the procedural text.

        This method examines entities to find cases where an entity defined in one step is used in a later step
        without being redefined in the intervening steps.

        :return: A tuple containing the generated question as a string and a boolean ground truth answer.
                 If no valid reaching definitions are found, returns an explanatory message and None.
        """
        valid_entities = []

        for entity_name, entity in self.text.entities.items():
            for def_step in entity.defined_in:
                for use_step in entity.used_in:
                    if def_step < use_step:
                        # Check if there's no redefinition between def_step and use_step
                        no_redefinition = True
                        for intermediate_step in range(def_step + 1, use_step):
                            if intermediate_step in entity.defined_in:
                                no_redefinition = False
                                break

                        if no_redefinition:
                            valid_entities.append((entity_name, def_step, use_step))

        if not valid_entities:
            return "No valid reaching definitions found.", None

        # Choose a random entity and steps
        entity_name, source_step, target_step = random.choice(valid_entities)

        # Generate question
        question = f"In Step {target_step + 1}, is the {entity_name} from Step {source_step + 1} being used?"

        # Ground truth is True since we verified no redefinition
        return question, True

    def generate_very_busy_expressions_question(
        self,
    ) -> List[Tuple[str, bool]]:
        """
        Generate questions about very busy expressions in the procedural text.

        Identifies entities that, after being defined, are used in multiple future steps.
        Generates a question for EACH such instance found in the recipe.

        :return: A list of tuples, each containing a generated question and the boolean ground truth (True).
                 Returns an empty list if no busy expressions are found.
        """
        busy_entity_questions = []  # Initialize list to store questions
        busy_entities_found = []  # Store the found entities first

        for entity_name, entity in self.text.entities.items():
            for def_step in entity.defined_in:
                # Find future uses (strictly after definition)
                future_uses = [step for step in entity.used_in if step > def_step]

                # Check for redefinition between def_step and each future_use
                # This makes the check stricter and aligns with the definition
                valid_future_uses = []
                for use_step in future_uses:
                    is_redefined_before_use = False
                    for intermediate_step in range(def_step + 1, use_step):
                        if intermediate_step in entity.defined_in:
                            is_redefined_before_use = True
                            break
                    if not is_redefined_before_use:
                        valid_future_uses.append(use_step)

                # Check if used in multiple future steps WITHOUT intermediate redefinition
                if len(valid_future_uses) >= 2:
                    busy_entities_found.append(
                        (entity_name, def_step, valid_future_uses)
                    )  # Store uses too for clarity

        if not busy_entities_found:
            return []  # Return empty list if none found

        # Generate a question for each found busy expression
        for entity_name, step, use_steps in busy_entities_found:
            # Construct the question text
            # Mentioning the specific future steps might make the question too complex/long.
            # Keep the original question format for simplicity.
            question = f"Is {entity_name} from Step {step + 1} used in multiple future steps without being redefined?"
            # Ground truth is True because we found at least two valid future uses
            busy_entity_questions.append((question, True))

        return busy_entity_questions

    def generate_available_expressions_question(self) -> Tuple[str, Optional[bool]]:
        """
        Generate a question that verifies if an entity remains available across the steps.

        The method checks whether an entity defined in one step is still valid and unmodified in a later step.

        :return: A tuple with the generated question and a boolean ground truth.
                 If no available expressions are detected, returns an explanatory message and None.
        """
        available_entities = []

        for entity_name, entity in self.text.entities.items():
            for def_step in entity.defined_in:
                for later_step in range(def_step + 1, len(self.text.steps)):
                    # Check if entity is available (not redefined between def_step and later_step)
                    is_available = not any(
                        intermediate_step in entity.defined_in
                        for intermediate_step in range(def_step + 1, later_step)
                    )
                    if is_available:
                        available_entities.append((entity_name, def_step, later_step))

        if not available_entities:
            return "No available expressions found.", None

        # Choose a random entity and steps
        entity_name, source_step, target_step = random.choice(available_entities)

        # Generate question
        question = f"Is {entity_name} from Step {source_step + 1} still available in Step {target_step + 1}?"

        # Ground truth is True since we verified availability
        return question, True

    def generate_live_variable_question(self) -> Tuple[str, Optional[bool]]:
        """
        Generate a question related to live variable analysis in the procedural text.

        This method evaluates whether a given entity is "live" after a specified step,
        meaning that the entity is still used in subsequent steps.

        :return: A tuple containing the generated question and a boolean ground truth.
                 If no suitable entity is found for live variable analysis, returns an explanatory message and None.
        """
        live_entities = []
        non_live_entities = []

        for entity_name, entity in self.text.entities.items():
            for step in range(len(self.text.steps)):
                # Check if entity is used after this step
                is_live = any(use_step > step for use_step in entity.used_in)

                if is_live:
                    live_entities.append((entity_name, step))
                else:
                    non_live_entities.append((entity_name, step))

        # Choose between live and non-live entities
        if live_entities and (not non_live_entities or random.random() > 0.5):
            entity_name, step = random.choice(live_entities)
            ground_truth = True
        elif non_live_entities:
            entity_name, step = random.choice(non_live_entities)
            ground_truth = False
        else:
            return "No entities found for live variable analysis.", None

        # Generate question
        question = f"Is {entity_name} still needed after Step {step + 1}?"

        return question, ground_truth

    def generate_interval_analysis_question(self) -> Tuple[str, Optional[str]]:
        """
        Generate a question regarding time interval analysis in the procedural text.

        It selects a step that specifies a time interval (either as an exact value or a range)
        and forms a question asking for that specific time interval detail.

        :return: A tuple with the question and the ground truth interval (as a string).
                 If no time interval is associated with any step, returns an explanatory message and None.
        """
        steps_with_intervals = []

        for step_idx in self.text.step_dependencies.nodes():
            if "time_interval" in self.text.step_dependencies.nodes[step_idx]:
                steps_with_intervals.append(step_idx)

        if not steps_with_intervals:
            return "No time intervals found in the steps.", None

        # Choose a random step with a time interval
        step = random.choice(steps_with_intervals)
        interval = self.text.step_dependencies.nodes[step]["time_interval"]

        # Generate question
        question = f"What is the time interval applied at Step {step + 1}?"

        # Ground truth is the actual interval
        if interval[0] == interval[1]:
            ground_truth = f"{interval[0]} {interval[2]}"
        else:
            ground_truth = f"{interval[0]}-{interval[1]} {interval[2]}"

        return question, ground_truth

    def generate_type_state_question(self) -> Tuple[str, Optional[bool]]:
        """
        Generate a question focusing on type-state analysis.

        This method searches for pairs of steps where an entity is defined and later used.
        It then constructs a question about the validity of using the entity if its prerequisite definition step were omitted.

        :return: A tuple containing the generated question and a boolean ground truth.
                 If no prerequisite relationships are found, returns an explanatory message and None.
        """
        prerequisite_pairs = []

        # Look for pairs of steps where one step prepares an entity for another
        for entity_name, entity in self.text.entities.items():
            defined_steps = sorted(list(entity.defined_in))
            used_steps = sorted(list(entity.used_in))

            for def_step in defined_steps:
                for use_step in used_steps:
                    if def_step < use_step:
                        # This is a potential prerequisite relationship
                        prerequisite_pairs.append((entity_name, def_step, use_step))

        if not prerequisite_pairs:
            return "No prerequisite relationships found.", None

        # Choose a random prerequisite pair
        entity_name, prerequisite_step, dependent_step = random.choice(
            prerequisite_pairs
        )

        # Extract action from the dependent step if possible
        step_text = self.text.steps[dependent_step]
        doc = nlp(step_text)

        action = None
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ not in {"be", "have"}:
                action = token.lemma_
                break

        if not action:
            action = "use"  # Default fallback

        # Generate question
        question = f"If we skip Step {prerequisite_step + 1}, is it still valid to {action} the {entity_name} in Step {dependent_step + 1}?"

        # Ground truth - if prerequisite step is skipped, the dependent step is invalid
        ground_truth = False

        return question, ground_truth

    def generate_taint_analysis_question(self) -> Tuple[str, Optional[bool]]:
        """
        Generate a question focused on taint analysis of the procedural text.

        The method identifies potential "tainted" entities—often raw ingredients or those flagged by specific keywords—
        and asks whether their use in a particular step might introduce safety concerns.

        :return: A tuple with the generated question and a boolean ground truth.
                 If no potential taints are identified, returns an explanatory message and None.
        """
        potential_taints = []

        # Look for entities that might introduce "taint"
        taint_keywords = {
            "raw",
            "optional",
            "allergen",
            "substitute",
            "contaminate",
            "uncooked",
        }

        for step_idx, step_text in enumerate(self.text.steps):
            step_text_lower = step_text.lower()

            # Check for taint keywords in the step
            if any(keyword in step_text_lower for keyword in taint_keywords):
                # Find entities in this step
                for entity_name, entity in self.text.entities.items():
                    if step_idx in entity.defined_in or step_idx in entity.used_in:
                        if entity_name in step_text_lower:
                            potential_taints.append((entity_name, step_idx))

        if not potential_taints:
            # If no obvious taints, look for raw ingredients
            for entity_name, entity in self.text.entities.items():
                for step_idx in entity.defined_in:
                    step_text = self.text.steps[step_idx].lower()
                    if "raw" in step_text or "egg" in entity_name:
                        potential_taints.append((entity_name, step_idx))

        if not potential_taints:
            return "No potential taints found.", None

        # Choose a random potential taint
        entity_name, step = random.choice(potential_taints)

        # Generate question
        question = f"Does using {entity_name} in Step {step + 1} introduce a potential safety concern to the recipe?"

        # Determine ground truth based on keywords and context
        step_text = self.text.steps[step].lower()

        # Raw ingredients that aren't cooked later are a concern
        is_raw = "raw" in step_text or "egg" in entity_name
        is_cooked_later = any(
            "cook" in self.text.steps[s].lower() or "bake" in self.text.steps[s].lower()
            for s in range(step + 1, len(self.text.steps))
        )

        # Optional ingredients are generally not a concern
        is_optional = "optional" in step_text

        ground_truth = is_raw and not is_cooked_later and not is_optional

        return question, ground_truth

    def generate_concurrency_analysis_question(self) -> Tuple[str, Optional[bool]]:
        """
        Generate a question regarding concurrency analysis among steps.

        This method examines steps marked as being concurrent or those that can run concurrently based on
        dependency and shared entity modifications, then formulates a related question.

        :return: A tuple containing the concurrency question and a boolean ground truth.
                 If no potential concurrent step pairs are found, returns an explanatory message and None.
        """
        # Find pairs of steps that might be concurrent
        potential_concurrent_pairs = []

        # First check steps explicitly marked as concurrent
        concurrent_steps = []
        for step_idx, attrs in self.text.step_dependencies.nodes(data=True):
            if attrs.get("concurrent", False):
                concurrent_steps.append(step_idx)

                # If we know what step it's concurrent with
                concurrent_with = attrs.get("concurrent_with")
                if concurrent_with is not None:
                    potential_concurrent_pairs.append((step_idx, concurrent_with))

        # If we don't have explicit concurrent pairs, look for potential ones
        if not potential_concurrent_pairs:
            for step1 in range(len(self.text.steps)):
                for step2 in range(step1 + 1, len(self.text.steps)):
                    # Check if steps have no direct dependency and don't modify the same entities
                    if self.text.can_steps_run_concurrently(step1, step2):
                        potential_concurrent_pairs.append((step1, step2))

        if not potential_concurrent_pairs:
            return "No potential concurrent steps found.", None

        # Choose a random pair
        step1, step2 = random.choice(potential_concurrent_pairs)

        # Generate question
        step1_text = self.text.steps[step1]
        step2_text = self.text.steps[step2]

        # Extract key actions from steps if possible
        doc1 = nlp(step1_text)
        action1 = next(
            (
                token.lemma_
                for token in doc1
                if token.pos_ == "VERB" and token.lemma_ not in {"be", "have"}
            ),
            "doing",
        )

        doc2 = nlp(step2_text)
        action2 = next(
            (
                token.lemma_
                for token in doc2
                if token.pos_ == "VERB" and token.lemma_ not in {"be", "have"}
            ),
            "doing",
        )

        question = f"Can we {action1} (Step {step1 + 1}) and {action2} (Step {step2 + 1}) at the same time?"

        # Ground truth is True since we verified they can run concurrently
        return question, True

    def generate_all_questions(self, num_per_type=1):
        """
        Generate a collection of questions for each analysis type provided.

        This method aggregates questions from various analysis functions such as reaching definitions,
        very busy expressions, available expressions, live variable analysis, interval analysis,
        type-state analysis, taint analysis, and concurrency analysis.

        :param num_per_type: The number of questions to generate per analysis type. Defaults to 1.
        :return: A list of tuples. Each tuple consists of an analysis type string and another tuple
                 containing the question and corresponding ground truth.
        """
        questions = []

        generator_functions = {
            "Reaching Definitions": self.generate_reaching_definitions_question,
            "Very Busy Expressions": self.generate_very_busy_expressions_question,  # This one returns a list now
            "Available Expressions": self.generate_available_expressions_question,
            "Live Variable Analysis": self.generate_live_variable_question,
            "Interval Analysis": self.generate_interval_analysis_question,
            "Type-State Analysis": self.generate_type_state_question,
            "Taint Analysis": self.generate_taint_analysis_question,
            "Concurrency Analysis": self.generate_concurrency_analysis_question,
        }

        for _ in range(num_per_type):  # Loop num_per_type times
            for q_type, generator_func in generator_functions.items():
                result = generator_func()

                if isinstance(
                    result, list
                ):  # Handle list results (like from Very Busy Expressions)
                    # Check if the list is not empty and contains tuples (question, answer)
                    if result and isinstance(result[0], tuple) and len(result[0]) == 2:
                        for question_answer in result:
                            if question_answer[1] is not None:  # Check for valid answer
                                questions.append((q_type, question_answer))
                    # Handle empty list case implicitly (adds nothing)
                elif (
                    isinstance(result, tuple) and len(result) == 2
                ):  # Handle single tuple result
                    question, answer = result
                    if answer is not None:  # Check for valid answer
                        questions.append((q_type, (question, answer)))
                else:
                    # Handle unexpected return types or None answers if necessary
                    loguru.logger.warning(
                        f"Generator for {q_type} returned unexpected result: {result}"
                    )

        return questions


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
    proc_text = ProceduralText(title, instructions)

    # Generate questions
    question_gen = QuestionGenerator(proc_text)
    questions = question_gen.generate_all_questions(num_per_type=1)

    # Format instructions as a dictionary with step numbers
    formatted_instructions = {i + 1: step for i, step in enumerate(instructions)}

    # Format entities for output
    formatted_entities = {}
    for entity_name, entity in proc_text.entities.items():
        formatted_entities[entity_name] = {
            "step_introduced": entity.step_introduced + 1,  # Convert to 1-indexed
            "states": entity.states,
            "used_in": [step + 1 for step in entity.used_in],  # Convert to 1-indexed
            "defined_in": [
                step + 1 for step in entity.defined_in
            ],  # Convert to 1-indexed
            "consumed_in": [
                step + 1 for step in entity.consumed_in
            ],  # Convert to 1-indexed
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
                proc_text = ProceduralText(title, instructions)

                # Generate questions
                question_gen = QuestionGenerator(proc_text)
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
        help="The limit on the number of questions per type",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
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

    # Process the recipes
    process_recipe_folder(args.input_folder, args.output_folder, args.limit)

    loguru.logger.info("Recipe processing complete!")


if __name__ == "__main__":
    main()
