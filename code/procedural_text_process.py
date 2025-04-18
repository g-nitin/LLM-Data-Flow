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
        Focuses on ingredients and key intermediate products, filtering out containers, tools, generic terms, locations, actions/verbs, quantities, times, and dimensions.
        """
        doc = nlp(step_text)
        loguru.logger.debug(f"*** Processing Step {step_idx + 1}: '{step_text}' ***")

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
            "pinch",
            "dash",
            "sprig",
            "clove",
            "slice",
            "slices",
            "piece",
            "pieces",
            "head",
            "bunch",
            "can",
            "cans",
            "jar",
            "jars",
            "package",
            "packages",
            "box",
            "boxes",
            "stick",
            "sticks",
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
            "surface",
            "counter",
            "stove",
            "burner",
            "hob",
            "range",
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
            "golden",
            "brown",
            "tender",
            "crisp",
            "soft",
            "firm",
            "thick",
            "thin",
            "smooth",
            "lumpy",
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
            "water",  # Often implicit, filter unless qualified e.g. "coconut water"
            "grease",  # Often refers to generic action byproduct unless specified e.g. "bacon grease"
            "oil",  # Often generic unless specified e.g. "olive oil"
            "juice",  # Often generic unless specified e.g. "lemon juice"
            "stock",
            "broth",  # Often generic unless specified e.g. "chicken stock"
        }

        # Explicit list of common cooking verbs (lemmas) to filter out
        verb_stop_list = {
            "preheat",
            "heat",
            "warm",
            "cool",
            "chill",
            "freeze",
            "bake",
            "roast",
            "grill",
            "broil",
            "toast",
            "fry",
            "saute",
            "sear",
            "boil",
            "simmer",
            "poach",
            "steam",
            "reduce",
            "mix",
            "stir",
            "combine",
            "blend",
            "whisk",
            "fold",
            "beat",
            "whip",
            "add",
            "incorporate",
            "introduce",
            "remove",
            "discard",
            "take",
            "set",
            "aside",
            "reserve",
            "drain",
            "strain",
            "sift",
            "measure",
            "weigh",
            "chop",
            "dice",
            "slice",
            "mince",
            "grate",
            "zest",
            "peel",
            "core",
            "trim",
            "mash",
            "puree",
            "crush",
            "grease",
            "oil",
            "butter",  # Verbs, not the nouns (usually)
            "season",
            "salt",
            "pepper",
            "cover",
            "wrap",
            "seal",
            "soak",
            "marinate",
            "serve",
            "plate",
            "garnish",
            "cook",
            "prepare",
            "make",
            "create",
            "start",
            "begin",
            "continue",
            "finish",
            "let",
            "allow",
            "ensure",
            "check",
            "test",
            "adjust",
            "repeat",
            "follow",
            "place",
            "put",
            "transfer",
            "pour",
            "sprinkle",
            "drizzle",
            "spread",
            "layer",
            "turn",
            "flip",
            "rotate",
            "bring",  # e.g., "bring to a boil"
            "work",  # e.g., "work the dough"
            "use",  # Too generic
            "need",  # Too generic
            "require",  # Too generic
            "taste",  # Action
            "look",  # Action
            "feel",  # Action
            "smell",  # Action
            "watch",  # Action
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
            r"^\d+\s?/\s?\d+$",  # Fractions like 1/2 or 1 / 2
            r"^\d+\s?-\s?\d+$",  # Ranges like 10-15
            r"^\d+x\d+$",  # Dimensions like 9x13
            r"^\d+(\s?(to|or|-)\s?\d+)?\s+(minute|minutes|min|hour|hours|hr|second|seconds|sec)$",  # Time durations e.g., "5 minutes", "10-15 minutes"
            r"^\d+(\s?(to|or|-)\s?\d+)?\s+(degree|degrees|°)\s?[fc]?$",  # Temperatures e.g., "350 degrees F"
            r"medium heat",
            r"high heat",
            r"low heat",  # Specific heat levels often not entities
            r"room temperature",
            r"^(about|approx\.?|approximately|around|over|under|less than|more than|at least)\s+\d+",
            r"^(a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(cup|tablespoon|teaspoon|pinch|dash|clove|slice|piece|can|jar|package|box|stick)",  # Common small quantities
        ]

        # Pre-compile regex for efficiency
        compiled_filter_patterns = [
            re.compile(p, re.IGNORECASE) for p in filter_patterns
        ]

        potential_entities = {}  # Use dict to store lemma -> original text mapping, helps deduplicate

        # Process Noun Chunks
        for chunk in doc.noun_chunks:
            # Clean the chunk text: lowercase, remove leading/trailing articles/possessives
            clean_chunk_text = chunk.text.lower()
            clean_chunk_text = re.sub(
                r"^(the|a|an|some|your|my|his|her|its|our|their|of|for|with|in|on|at|to|about|approx\.?|approximately|around|over|under|less than|more than|at least)\s+",
                "",
                clean_chunk_text,
            ).strip()
            clean_chunk_text = re.sub(
                r"\s+'s$", "", clean_chunk_text
            ).strip()  # Remove trailing 's

            # Use the lemma of the root word for checking against stop lists
            root_lemma = chunk.root.lemma_.lower()

            # Apply Filters
            # 0. Check first token POS: Skip if it starts with a verb, adverb, number, etc.
            first_token_pos = chunk[0].pos_
            if first_token_pos in {
                "VERB",
                "AUX",
                "ADV",
                "ADP",
                "NUM",
                "PUNCT",
                "PART",
                "SCONJ",
                "CCONJ",
            }:
                # Allow ADP only if it's part of a known multi-word entity? (e.g. "cream of tartar") - complex, skip for now.
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' because first token POS is {first_token_pos}"
                )
                continue

            # 1. Check exact cleaned text against compiled regex patterns
            if any(
                pattern.match(clean_chunk_text) for pattern in compiled_filter_patterns
            ):
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' due to pattern match on '{clean_chunk_text}'"
                )
                continue

            # 2. Check root lemma against general stop lemmas
            if root_lemma in stop_entity_lemmas:
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' because root lemma '{root_lemma}' is in stop list."
                )
                continue

            # 3. Check root lemma against VERB stop list
            if root_lemma in verb_stop_list:
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' because root lemma '{root_lemma}' is in VERB stop list."
                )
                continue

            # 4. Check full cleaned text against general stop lemmas (for multi-word generics)
            if clean_chunk_text in stop_entity_lemmas:
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' because full text '{clean_chunk_text}' is in stop list."
                )
                continue

            # 5. Check full cleaned text against VERB stop list
            if clean_chunk_text in verb_stop_list:
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' because full text '{clean_chunk_text}' is in VERB stop list."
                )
                continue

            # 6. Length check (on cleaned text)
            if len(clean_chunk_text) < 3:
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' due to short length after cleaning: '{clean_chunk_text}'"
                )
                continue

            # 7. Check if root is a verb/adjective (redundant with check 3, but keep as safety)
            if chunk.root.pos_ not in {"NOUN", "PROPN"}:
                # Allow ADJ if it modifies a noun? e.g. "large onions" - chunk root might be 'large'
                # Let's be strict for now: root must be NOUN or PROPN
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' because root POS is {chunk.root.pos_}"
                )
                continue

            # 8. Check if the chunk seems to be just a quantity + unit (missed by patterns)
            if (
                len(chunk) == 2
                and chunk[0].like_num
                and chunk[1].lemma_ in measurement_units
            ):
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' as it looks like quantity+unit."
                )
                continue

            # If passes filters, add it
            # Use lemma for deduplication key, store original cleaned text
            entity_lemma = root_lemma  # Using root lemma as the key
            # Alternative: use the full cleaned text as key if root lemma is too ambiguous
            # entity_key = clean_chunk_text

            # Prefer longer chunks if lemmas collide (e.g., "olive oil" vs "oil")
            if entity_lemma not in potential_entities or len(clean_chunk_text) > len(
                potential_entities[entity_lemma]
            ):
                potential_entities[entity_lemma] = clean_chunk_text
                loguru.logger.info(
                    f"Keeping potential entity: '{chunk.text}' -> Cleaned: '{clean_chunk_text}', Key Lemma: '{entity_lemma}'"
                )
            else:
                loguru.logger.debug(
                    f"Skipping chunk '{chunk.text}' because lemma '{entity_lemma}' already exists with '{potential_entities[entity_lemma]}'"
                )

        # Final list of entity names (using the stored original text)
        filtered_entities = list(potential_entities.values())
        loguru.logger.info(
            f"Step {step_idx + 1}: Final extracted entities for this step: {filtered_entities}"
        )

        # Create or update entity objects (using the extracted names)
        step_entity_roles = {}  # Track roles assigned in this step: entity_name -> set of roles {'used', 'defined', 'consumed'}

        for entity_name in filtered_entities:
            # Ensure we use the *extracted name* for consistency in the ProceduralText object
            if entity_name not in self.entities:
                self.entities[entity_name] = Entity(entity_name, step_idx)
                # Initial definition assumed when first extracted, refine based on verbs
                self.entities[entity_name].defined_in.add(step_idx)
                loguru.logger.debug(
                    f"Adding new global entity: '{entity_name}' from step {step_idx + 1}"
                )

            # Determine entity's role in this step using verbs (existing logic)
            is_used = False
            is_defined = False
            is_consumed = False

            # Define verb categories
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
                "coat",
                "stuff",
                "fill",
                "layer",
                "spread",
                "drizzle",
                "simmer",
                "boil",
                "bake",
                "roast",
                "cook",
                "fry",
                "saute",
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
                "melt",
                "toast",
                "brown",
                "render",
                "whip",
                "beat",
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
            }
            # Verbs indicating transformation (Expanded)
            transformation_verbs = {
                "mash",
                "chop",
                "dice",
                "slice",
                "mince",
                "grate",
                "zest",
                "juice",
                "cook",
                "bake",
                "boil",
                "simmer",
                "fry",
                "roast",
                "melt",
                "toast",
                "brown",
                "whip",
                "beat",
                "reduce",
                "puree",  # Added puree
            }

            # Check verbs acting ON this entity (check dependency relations)
            entity_found_in_step = False
            verb_action_found = False  # Flag if a verb explicitly acted on the entity
            for token in doc:
                # Find occurrences of the entity name (or its root lemma) in the step
                # This needs to be robust: check token text, lemma, and potentially substring matches
                if (
                    entity_name in token.text.lower()
                    or entity_name == token.lemma_.lower()
                ):
                    entity_found_in_step = True
                    # Check the token's syntactic head (the verb governing it)
                    head = token.head
                    if head.pos_ == "VERB":
                        verb_lemma = head.lemma_.lower()
                        loguru.logger.debug(
                            f"Entity '{entity_name}' governed by verb '{verb_lemma}' ({head.text}) with relation '{token.dep_}'"
                        )

                        # Check if the entity is the object/subject/conjunct etc.
                        if token.dep_ in {
                            "dobj",
                            "pobj",
                            "attr",
                            "nsubj",
                            "nsubjpass",
                            "conj",
                            "appos",
                            "advcl",
                        }:
                            verb_action_found = (
                                True  # Mark that we found a direct verb action
                            )
                            governing_verb_lemma = (
                                verb_lemma  # Store for transformation check
                            )
                            if verb_lemma in use_verbs:
                                is_used = True
                                loguru.logger.debug(
                                    f"Marking '{entity_name}' as USED by verb '{verb_lemma}'"
                                )
                            # Use 'elif' to avoid double-counting if a verb is in multiple sets (define takes precedence over use if both match)
                            elif verb_lemma in define_verbs:
                                is_defined = True
                                loguru.logger.debug(
                                    f"Marking '{entity_name}' as DEFINED by verb '{verb_lemma}'"
                                )
                            elif verb_lemma in consume_verbs:
                                is_consumed = True
                                loguru.logger.debug(
                                    f"Marking '{entity_name}' as CONSUMED by verb '{verb_lemma}'"
                                )

            # If entity name appears but not clearly governed by a verb (e.g., just mentioned)
            # Default to 'used' if not defined/consumed? Or require explicit verb action?
            # Let's require explicit verb action for now to be stricter.
            # However, if an entity is mentioned first time, it's likely defined.
            if entity_name in self.entities:
                current_entity = self.entities[entity_name]
                roles_assigned = set()

                # Handle initial definition assumption
                if (
                    step_idx == current_entity.step_introduced
                    and not verb_action_found
                    and entity_found_in_step
                ):
                    # If it's the first mention and no clear verb action, assume definition (e.g., "2 cups flour")
                    is_defined = True
                    loguru.logger.debug(
                        f"Marking '{entity_name}' as DEFINED (first mention, no clear verb)"
                    )

                # Update entity roles for this step
                if is_used:
                    current_entity.used_in.add(step_idx)
                    roles_assigned.add("used")
                if is_defined:
                    # Ensure definition is marked correctly
                    current_entity.defined_in.discard(
                        step_idx
                    )  # Remove initial assumption if needed
                    current_entity.defined_in.add(step_idx)
                    roles_assigned.add("defined")
                    # If defined by transformation, also mark as used (consuming previous state)
                    if governing_verb_lemma in transformation_verbs:
                        current_entity.used_in.add(step_idx)
                        roles_assigned.add("used")  # Add 'used' role as well
                        loguru.logger.debug(
                            f"Also marking '{entity_name}' as USED due to transformation verb '{governing_verb_lemma}'"
                        )
                if is_consumed:
                    current_entity.consumed_in.add(step_idx)
                    current_entity.consumed_in.add(step_idx)

                # If the entity was found, wasn't its introduction step, and no role was assigned by verbs, assume 'used'.
                if (
                    entity_found_in_step
                    and not roles_assigned
                    and step_idx != current_entity.step_introduced
                ):
                    # Check if it was defined or used previously to justify assuming 'used' now
                    was_previously_active = any(
                        s < step_idx
                        for s in current_entity.defined_in.union(current_entity.used_in)
                    )
                    if was_previously_active:
                        loguru.logger.warning(
                            f"Entity '{entity_name}' found in step {step_idx + 1} but no clear verb role assigned. Applying fallback: marking as USED."
                        )
                        current_entity.used_in.add(step_idx)
                        roles_assigned.add("used")

                # Store assigned roles for this step (used for debugging/verification if needed)
                step_entity_roles[entity_name] = roles_assigned

        loguru.logger.debug(f"*** Finished Step {step_idx + 1} ***")

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
                if len(valid_future_uses) >= 1:
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

        # Consider entities that are actually defined or used at some point
        relevant_entities = {
            name: entity
            for name, entity in self.text.entities.items()
            if entity.defined_in or entity.used_in
        }

        if not relevant_entities:
            return "No relevant entities found for live variable analysis.", None

        for entity_name, entity in relevant_entities.items():
            # Check liveness after each step where the entity *could* be relevant
            # (i.e., from its introduction up to the second-to-last step)
            last_possible_step = len(self.text.steps) - 1
            start_step = entity.step_introduced  # Start checking after it's introduced

            for step in range(start_step, last_possible_step):
                # Check if entity is used in any step strictly after 'step'
                is_live = any(use_step > step for use_step in entity.used_in)

                # Avoid asking about liveness if the entity was consumed at or before this step
                consumed_before_or_at_step = any(
                    cons_step <= step for cons_step in entity.consumed_in
                )

                if not consumed_before_or_at_step:
                    if is_live:
                        live_entities.append((entity_name, step))
                    else:
                        # Only consider non-live if it wasn't consumed earlier
                        non_live_entities.append((entity_name, step))

        # Ensure we don't have duplicates (e.g., entity live after step 2 and step 3)
        live_entities = list(set(live_entities))
        non_live_entities = list(set(non_live_entities))

        # Choose between live and non-live entities to generate a mix of True/False questions
        chosen_entity = None
        ground_truth = None

        can_choose_live = bool(live_entities)
        can_choose_non_live = bool(non_live_entities)

        if can_choose_live and (not can_choose_non_live or random.random() > 0.5):
            chosen_entity = random.choice(live_entities)
            ground_truth = True
        elif can_choose_non_live:
            chosen_entity = random.choice(non_live_entities)
            ground_truth = False

        if chosen_entity is None:
            return (
                "Could not find a suitable scenario for live variable analysis question.",
                None,
            )

        entity_name, step = chosen_entity

        # Old: question = f"Is {entity_name} still needed after Step {step + 1}?"
        question = f"Is {entity_name} live after Step {step + 1}?"

        loguru.logger.debug(
            f"Generated Liveness Q: '{question}' -> {ground_truth}. Entity: {entity_name}, Step checked after: {step + 1}, Used in: {self.text.entities[entity_name].used_in}"
        )

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
        cooking_verbs = {
            "cook",
            "bake",
            "simmer",
            "boil",
            "broil",
            "fry",
            "roast",
            "grill",
            "steam",
            "saute",
        }
        is_raw = "raw" in step_text or "egg" in entity_name
        is_cooked_later = any(
            any(
                verb in self.text.steps[s].lower().split() for verb in cooking_verbs
            )  # Check if any cooking verb is present
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
            "Very Busy Expressions": self.generate_very_busy_expressions_question,
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
