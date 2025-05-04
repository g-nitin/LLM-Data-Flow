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
# nlp = spacy.load("en_core_web_sm")  # Smaller model for faster processing
nlp = spacy.load("en_core_web_trf")  # Larger model for better accuracy


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

        self.exclusive_resources = {
            "oven",
            "stove",
            "burner",
            "hob",
            "range",
            "mixer",
            "stand mixer",
            "blender",
            "food processor",
            "microwave",
            "grill",
            "smoker",
            "deep fryer",
            "skillet",
            "saucepan",
            "wok",
            "griddle",
            "air fryer",
            "pressure cooker",
            "slow cooker",
        }
        self.verb_to_resource_map = {
            # Oven Verbs
            "bake": "oven",
            "roast": "oven",
            "broil": "oven",
            "preheat": "oven",
            # Stove Verbs
            "boil": "stove",
            "simmer": "stove",
            "fry": "stove",
            "saute": "stove",
            "sear": "stove",
            "reduce": "stove",
            "melt": "stove",
            "toast (in pan)": "stove",
            "steam (on stove)": "stove",
            "cook": "stove",
            "stir-fry": "wok",
            "mix (with mixer)": "mixer",
            "beat (with mixer)": "mixer",
            "whip (with mixer)": "mixer",
            "blend": "blender",
            "puree (with blender/processor)": "blender",
            "process": "food processor",
            "microwave": "microwave",
            "grill": "grill",
            "smoke": "smoker",
            "deep-fry": "deep fryer",
            "air-fry": "air fryer",
        }

        # Define essential tools based on resources
        self.essential_tools = self.exclusive_resources.copy()  # Start with appliances
        # Add common non-appliance exclusives, e.g., specific pans
        self.essential_tools.update({"skillet", "saucepan", "wok", "griddle"})

        # Initialize step dependencies graph - ONLY ADD NODES
        for i in range(len(steps)):
            self.step_dependencies.add_node(i)

        # Parse steps to extract entities and their relationships
        self._parse_steps()  # Extract entities/roles
        self.build_entity_flow_graph()
        self._add_dataflow_dependencies()  # Add dataflow edges AFTER initial parsing

        # Log all variables
        loguru.logger.info(
            f"\n{len(self.steps)} steps:\n{pformat(list(zip(range(1, len(self.steps) + 1), self.steps)))}"
            f"\nParsed {len(self.entities)} entities:\n{pformat(self.entities)}"
            f"\nStep dependencies (Nodes: {self.step_dependencies.number_of_nodes()}, Edges: {self.step_dependencies.number_of_edges()}):\n{pformat(list(self.step_dependencies.edges()))}"
        )

    def _get_lemma(text: str, nlp_processor) -> str:
        """Get the lemma of the last noun in a potentially multi-word string."""
        doc = nlp_processor(text)
        # Find the last token that is a noun or proper noun
        last_noun = None
        for token in reversed(doc):
            if token.pos_ in {"NOUN", "PROPN"}:
                last_noun = token
                break
        # Fallback to the last token's lemma if no noun found
        return last_noun.lemma_.lower() if last_noun else doc[-1].lemma_.lower()

    def _extract_entities_from_step(self, step_idx, step_text):
        """
        Extract entities from a step using NLP techniques with refined filtering for recipes.
        Focuses on ingredients and key intermediate products, filtering out containers, tools, generic terms, locations, actions/verbs, quantities, times, and dimensions.
        Includes rule-based linking for derived entities.
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
            "pot",
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
            "liquid",
            "solids",
            "everything",
            "anything",
            "something",
            "nothing",
            "water",
            "grease",
            "oil",
            "juice",
            "stock",
            "broth",
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
            "butter",
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
            "bring",
            "work",
            "use",
            "need",
            "require",
            "taste",
            "look",
            "feel",
            "smell",
            "watch",
        }

        # Combine stop lemmas
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
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (first token POS: {first_token_pos})"
                )
                continue

            # 1. Check exact cleaned text against compiled regex patterns
            if any(
                pattern.match(clean_chunk_text) for pattern in compiled_filter_patterns
            ):
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (pattern match: '{clean_chunk_text}')"
                )
                continue

            # 2. Check root lemma against general stop lemmas
            is_essential = root_lemma in self.essential_tools
            if is_essential:
                loguru.logger.trace(
                    f"Keeping chunk '{chunk.text}' because root lemma '{root_lemma}' is an essential tool."
                )
                # Skip other stop list checks if it's essential
            else:
                # If not essential, apply normal stop list checks
                if root_lemma in stop_entity_lemmas:
                    loguru.logger.trace(
                        f"Skipping chunk '{chunk.text}' (root lemma '{root_lemma}' in stop list)"
                    )
                    continue

            # 3. Check root lemma against VERB stop list
            if root_lemma in verb_stop_list:
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (root lemma '{root_lemma}' in VERB stop list)"
                )
                continue

            # 4. Check full cleaned text against general stop lemmas (for multi-word generics)
            if clean_chunk_text in stop_entity_lemmas:
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (full text '{clean_chunk_text}' in stop list)"
                )
                continue

            # 5. Check full cleaned text against VERB stop list
            if clean_chunk_text in verb_stop_list:
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (full text '{clean_chunk_text}' in VERB stop list)"
                )
                continue

            # 6. Length check (on cleaned text)
            if len(clean_chunk_text) < 2:
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (short length: '{clean_chunk_text}')"
                )
                continue

            # 7. Check if root is a verb/adjective (redundant with check 3, but keep as safety)
            if chunk.root.pos_ not in {"NOUN", "PROPN"}:
                # Allow ADJ if it modifies a noun? e.g. "large onions" - chunk root might be 'large'
                # Let's be strict for now: root must be NOUN or PROPN
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (root POS: {chunk.root.pos_})"
                )
                continue

            # 8. Check if the chunk seems to be just a quantity + unit (missed by patterns)
            if (
                len(chunk) == 2
                and chunk[0].like_num
                and chunk[1].lemma_ in measurement_units
            ):
                loguru.logger.trace(f"Skipping chunk '{chunk.text}' (quantity+unit)")
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
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (lemma '{entity_lemma}' exists with '{potential_entities[entity_lemma]}')"
                )

        # Final list of entity names (using the stored original text)
        filtered_entities = list(potential_entities.values())
        loguru.logger.info(
            f"Step {step_idx + 1}: Initial extracted entities for this step: {filtered_entities}"
        )

        loguru.logger.debug(f"Step {step_idx + 1}: Applying rule-based linking...")
        # Define patterns for derived entities: suffix -> potential base extraction method
        # Method 'suffix': Extracts text before the last space and the suffix
        # Method 'modifier': Extracts the last word (assumed head noun)
        derived_entity_rules = {
            "mixture": "suffix",
            "dough": "suffix",
            "batter": "suffix",
            "filling": "suffix",
            "sauce": "suffix",
            "glaze": "suffix",
            "marinade": "suffix",
            "dressing": "suffix",
            "paste": "suffix",
            "crust": "modifier",  # e.g., "potato crust" -> "crust"
            "stock": "modifier",  # e.g., "chicken stock" -> "stock" (might be too generic, adjust if needed)
            "broth": "modifier",  # e.g., "beef broth" -> "broth"
            "juice": "modifier",  # e.g., "lemon juice" -> "juice"
            "oil": "modifier",  # e.g., "olive oil" -> "oil"
            "milk": "modifier",  # e.g., "almond milk" -> "milk"
            # Add more rules as needed
        }

        # Use noun chunks from the step text for checking derived forms
        for chunk in doc.noun_chunks:
            # Basic cleaning similar to entity extraction
            chunk_text = chunk.text.lower().strip()
            chunk_text = re.sub(
                r"^(the|a|an|some|any|its|the)\s+", "", chunk_text
            ).strip()
            if len(chunk_text.split()) < 2:  # Need at least two words for these rules
                continue

            for suffix, method in derived_entity_rules.items():
                base_entity_name = None
                if method == "suffix" and chunk_text.endswith(f" {suffix}"):
                    # Extract base part: text before the suffix
                    base_entity_name = chunk_text[: -(len(suffix) + 1)].strip()
                    loguru.logger.trace(
                        f"Link Check (Suffix): Chunk '{chunk_text}', Suffix '{suffix}', Potential Base '{base_entity_name}'"
                    )

                elif method == "modifier" and chunk_text.endswith(f" {suffix}"):
                    # Check if the last word matches the suffix rule key
                    # Extract base part: the suffix itself (assumed head noun)
                    base_entity_name = suffix
                    loguru.logger.trace(
                        f"Link Check (Modifier): Chunk '{chunk_text}', Suffix '{suffix}', Potential Base '{base_entity_name}'"
                    )

                # If we extracted a potential base entity name
                if (
                    base_entity_name and len(base_entity_name) > 1
                ):  # Avoid empty or single-letter bases
                    # Check if this potential base entity exists globally
                    # Try exact match first, then lemma match as fallback
                    target_entity = None
                    if base_entity_name in self.entities:
                        target_entity = self.entities[base_entity_name]
                    else:
                        # Fallback: Check if the lemma of the base matches an existing entity's lemma
                        # This requires storing lemmas or re-calculating. Let's skip lemma fallback for now to keep it simple.
                        # base_lemma = _get_lemma(base_entity_name, nlp)
                        # for existing_name, entity_obj in self.entities.items():
                        #     if _get_lemma(existing_name, nlp) == base_lemma:
                        #         target_entity = entity_obj
                        #         base_entity_name = existing_name # Use the actual key
                        #         loguru.logger.trace(f"Link Check: Found base '{base_entity_name}' via lemma '{base_lemma}'")
                        #         break
                        pass  # No lemma fallback currently

                    if target_entity:
                        # Check if the base entity was active (defined or used) before this step
                        was_active_before = any(
                            s < step_idx
                            for s in target_entity.defined_in.union(
                                target_entity.used_in
                            )
                        )
                        if was_active_before:
                            # Mark the BASE entity as USED in the CURRENT step
                            if step_idx not in target_entity.used_in:
                                loguru.info(
                                    f"Rule-based Link: Marking base entity '{base_entity_name}' as USED in step {step_idx + 1} due to derived form '{chunk_text}'"
                                )
                                target_entity.used_in.add(step_idx)
                            break  # Stop checking other rules for this chunk
                        else:
                            loguru.logger.trace(
                                f"Rule-based Link: Base '{base_entity_name}' found but not active before step {step_idx + 1}."
                            )
                    else:
                        loguru.logger.trace(
                            f"Rule-based Link: Potential base entity '{base_entity_name}' (from '{chunk_text}') not found in global entities."
                        )

            # if found_link: # Optimization: if we linked this chunk, maybe skip to next chunk?
            #     continue

        # Create or update entity objects (using the extracted names)
        step_entity_roles = {}  # Track roles assigned in this step: entity_name -> set of roles {'used', 'defined', 'consumed'}

        # Loop through initially filtered entities to assign roles based on verbs/dependencies
        for entity_name in filtered_entities:
            # Ensure we use the *extracted name* for consistency in the ProceduralText object
            if entity_name not in self.entities:
                # This check might be redundant if potential_entities logic already added them, but safe to keep
                self.entities[entity_name] = Entity(entity_name, step_idx)
                # Initial definition assumed when first extracted, refine based on verbs
                self.entities[entity_name].defined_in.add(step_idx)
                loguru.logger.debug(
                    f"Adding new global entity (during role assignment): '{entity_name}' from step {step_idx + 1}"
                )

            # Determine entity's role in this step using verbs (existing logic)
            is_used = False
            is_defined = False
            is_consumed = False
            governing_verb_lemma = None

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
                "puree",
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
                "puree",
            }

            # Check verbs acting ON this entity (check dependency relations)
            entity_found_in_step = False
            verb_action_found = False  # Flag if a verb explicitly acted on the entity
            # Re-use the doc processed earlier
            for token in doc:
                # Find occurrences of the entity name (or its root lemma) in the step
                # Be robust: check token text, lemma, and potentially substring matches
                # Use lowercase comparison for robustness
                token_text_lower = token.text.lower()
                token_lemma_lower = token.lemma_.lower()
                entity_name_lower = entity_name.lower()

                # Check if token is part of the entity name (simple substring check for now)
                # A better check would involve aligning tokens with noun chunks
                if (
                    entity_name_lower in token_text_lower
                    or entity_name_lower == token_lemma_lower
                ):
                    # More specific check: is this token the root of a chunk matching the entity?
                    is_entity_match = False
                    if hasattr(
                        token, "sent"
                    ):  # Check if token belongs to a sentence (spaCy structure)
                        for chunk in token.sent.noun_chunks:
                            # Compare cleaned chunk text with entity name
                            clean_chunk_text = chunk.text.lower().strip()
                            clean_chunk_text = re.sub(
                                r"^(the|a|an|some|your|my|his|her|its|our|their|of|for|with|in|on|at|to|about|approx\.?|approximately|around|over|under|less than|more than|at least)\s+",
                                "",
                                clean_chunk_text,
                            ).strip()
                            if clean_chunk_text == entity_name_lower and token in chunk:
                                is_entity_match = True
                                break
                    # Fallback to simple check if chunk matching fails or is complex
                    if not is_entity_match:
                        if (
                            entity_name_lower in token_text_lower
                            or entity_name_lower == token_lemma_lower
                        ):
                            is_entity_match = True  # Less precise fallback

                    if is_entity_match:
                        entity_found_in_step = True
                        # Check the token's syntactic head (the verb governing it)
                        head = token.head
                        if head.pos_ == "VERB":
                            verb_lemma = head.lemma_.lower()
                            loguru.logger.debug(
                                f"Entity '{entity_name}' potentially governed by verb '{verb_lemma}' ({head.text}) with relation '{token.dep_}'"
                            )

                            # Check if the entity is the object/subject/conjunct etc. of the verb
                            if (
                                token.dep_
                                in {
                                    "dobj",
                                    "pobj",
                                    "obj",  # Direct/prepositional/general object
                                    "attr",  # Attribute (e.g., "it is flour")
                                    "nsubj",
                                    "nsubjpass",  # Nominal subject (active/passive)
                                    "conj",  # Conjunct (e.g., "mix flour and sugar")
                                    "appos",  # Appositional modifier
                                    "advcl",  # Adverbial clause modifier (less common for direct action)
                                    "xcomp",  # Open clausal complement
                                    "agent",  # Agent in passive construction
                                }
                            ):
                                verb_action_found = True
                                governing_verb_lemma = (
                                    verb_lemma  # Store for transformation check
                                )
                                loguru.logger.trace(
                                    f"Entity '{entity_name}' governed by verb '{verb_lemma}' ({head.text}) dep '{token.dep_}'"
                                )

                                # Assign roles based on verb lists (use elif for precedence)
                                if verb_lemma in consume_verbs:
                                    is_consumed = True
                                    loguru.logger.debug(
                                        f"Marking '{entity_name}' as CONSUMED by verb '{verb_lemma}'"
                                    )
                                elif verb_lemma in define_verbs:
                                    is_defined = True
                                    loguru.logger.debug(
                                        f"Marking '{entity_name}' as DEFINED by verb '{verb_lemma}'"
                                    )
                                elif verb_lemma in use_verbs:
                                    is_used = True
                                    loguru.logger.debug(
                                        f"Marking '{entity_name}' as USED by verb '{verb_lemma}'"
                                    )
                                else:
                                    loguru.logger.trace(
                                        f"Verb '{verb_lemma}' acting on '{entity_name}' not in standard role lists."
                                    )
                                    # Default assumption if verb found but not categorized? Maybe 'used'?
                                    # Let's be 'conservative' and not assume role if verb unknown.

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
                    loguru.logger.trace(
                        f"Marking '{entity_name}' as DEFINED (first mention, no clear verb)"
                    )

                # Update entity roles for this step
                if is_consumed:  # Consume takes precedence
                    current_entity.consumed_in.add(step_idx)
                    roles_assigned.add("consumed")
                    # If consumed, it might also be used in the process (e.g., draining uses the liquid before discarding)
                    # Let's keep it simple: consume means it's gone after this.
                    # Remove from defined/used if it was marked earlier in this step?
                    current_entity.defined_in.discard(step_idx)
                    current_entity.used_in.discard(step_idx)
                    loguru.logger.debug(
                        f"Marking '{entity_name}' as CONSUMED in step {step_idx + 1}"
                    )
                elif is_defined:  # Define before Use
                    # Ensure definition is marked correctly
                    current_entity.defined_in.add(step_idx)
                    roles_assigned.add("defined")
                    loguru.logger.debug(
                        f"Marking '{entity_name}' as DEFINED in step {step_idx + 1}"
                    )
                    # If defined by transformation, also mark as used (consuming previous state)
                    if governing_verb_lemma in transformation_verbs:
                        if (
                            step_idx not in current_entity.used_in
                        ):  # Avoid redundant logging
                            loguru.logger.debug(
                                f"Also marking '{entity_name}' as USED in step {step_idx + 1} due to transformation verb '{governing_verb_lemma}'"
                            )
                        current_entity.used_in.add(step_idx)  # Ensure it's added
                        roles_assigned.add("used")  # Add 'used' role as well

                elif is_used:  # Use is the fallback if not consumed/defined by a verb
                    current_entity.used_in.add(step_idx)
                    roles_assigned.add("used")
                    loguru.logger.debug(
                        f"Marking '{entity_name}' as USED in step {step_idx + 1}"
                    )

                # Fallback: If entity found, not introduction, no verb action, assume 'used' if previously active
                if (
                    entity_found_in_step
                    and not roles_assigned
                    and step_idx != current_entity.step_introduced
                ):
                    was_previously_active = any(
                        s < step_idx
                        for s in current_entity.defined_in.union(current_entity.used_in)
                    )
                    if was_previously_active:
                        loguru.warning(
                            f"Entity '{entity_name}' found in step {step_idx + 1} but no clear verb role assigned. Applying fallback: marking as USED."
                        )
                        current_entity.used_in.add(step_idx)
                        roles_assigned.add("used")

                # Store assigned roles for this step (used for debugging/verification if needed)
                if roles_assigned:  # Only store if some role was determined
                    step_entity_roles[entity_name] = roles_assigned

        loguru.logger.debug(
            f"Step {step_idx + 1}: Roles assigned in this step: {step_entity_roles}"
        )
        loguru.logger.debug(f"*** Finished Step {step_idx + 1} ***")

    def _parse_steps(self):
        """Parse steps to extract entities, relationships, and concurrency hints"""
        # Process each step
        for step_idx, step_text in enumerate(self.steps):
            # Extract entities and determine their roles (defined, used)
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
            if exact_matches and not matches:
                for match in exact_matches:
                    time_val, unit = match
                    self.step_dependencies.nodes[step_idx]["time_interval"] = (
                        int(time_val),
                        int(time_val),
                        unit,
                    )

            # Look for concurrency keywords ("while", etc.) and mark the node
            # We are not modifying the graph structure here based on "while".
            # We just mark the node as potentially concurrent with *something*.
            if any(
                word in step_text.lower()
                for word in ["while", "during", "meanwhile", "at the same time"]
            ):
                self.step_dependencies.nodes[step_idx]["concurrent_hint"] = True
                loguru.logger.trace(
                    f"Step {step_idx + 1} marked with concurrent_hint=True."
                )
            else:
                self.step_dependencies.nodes[step_idx]["concurrent_hint"] = False

    def _add_dataflow_dependencies(self):
        """
        Adds edges to the step_dependencies graph based on data flow.
        An edge A -> B is added if step B uses an entity last defined in step A.
        This complements the default sequential edges.
        """
        loguru.logger.debug("Adding dataflow dependencies to the graph...")
        edges_added = 0
        for entity_name, entity in self.entities.items():
            # Sort definition and usage steps
            defined_steps = sorted(list(entity.defined_in))
            used_steps = sorted(list(entity.used_in))

            if not defined_steps or not used_steps:
                continue  # Entity needs to be defined and used for dataflow

            # For each usage step, find the latest definition step before it
            for use_step in used_steps:
                latest_def_step = -1
                for def_step in defined_steps:
                    if def_step < use_step:
                        latest_def_step = max(latest_def_step, def_step)
                    else:
                        # Since lists are sorted, no need to check further def_steps for this use_step
                        break

                # If a preceding definition exists, add the dataflow edge
                if latest_def_step != -1:
                    # Check if the entity was redefined between the latest_def_step and use_step
                    redefined_between = False
                    for intermediate_def_step in defined_steps:
                        if latest_def_step < intermediate_def_step < use_step:
                            redefined_between = True
                            break

                    # Only add edge if NOT redefined in between
                    if not redefined_between:
                        if not self.step_dependencies.has_edge(
                            latest_def_step, use_step
                        ):
                            self.step_dependencies.add_edge(latest_def_step, use_step)
                            loguru.logger.debug(
                                f"Added dataflow edge: {latest_def_step + 1} -> {use_step + 1} (for entity '{entity_name}')"
                            )
                            edges_added += 1
                        # else: Edge already exists (could be default or another dataflow)
                    else:
                        loguru.logger.trace(
                            f"Skipped dataflow edge {latest_def_step + 1} -> {use_step + 1} for '{entity_name}' (redefined between)."
                        )

        loguru.logger.info(f"Added {edges_added} dataflow dependency edges.")

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

    def _get_resources_used(self, step_idx: int) -> Set[str]:
        """
        Identifies exclusive resources potentially used by a step based on verbs, entities, and keywords.
        """
        resources = set()
        step_text = self.steps[step_idx].lower()
        doc = nlp(step_text)

        # 1. Check verbs mapped to resources
        for token in doc:
            if token.pos_ == "VERB":
                verb_lemma = token.lemma_
                if verb_lemma in self.verb_to_resource_map:
                    resource = self.verb_to_resource_map[verb_lemma]
                    if resource not in resources:
                        loguru.logger.trace(
                            f"Step {step_idx + 1}: Verb '{verb_lemma}' maps to resource '{resource}'"
                        )
                    resources.add(resource)

        # 2. Check entities mentioned that ARE resources themselves
        step_entities_lower = {
            e.lower()
            for e, ent in self.entities.items()
            if step_idx in ent.used_in or step_idx in ent.defined_in
        }
        for entity_lower in step_entities_lower:
            if entity_lower in self.exclusive_resources:
                if entity_lower not in resources:  # Avoid duplicate logging
                    loguru.logger.trace(
                        f"Step {step_idx + 1}: Entity '{entity_lower}' used/defined is an exclusive resource."
                    )
                resources.add(entity_lower)

        # 3. Simple keyword check as fallback
        for resource in self.exclusive_resources:
            if resource in step_text:
                if resource not in resources:
                    loguru.logger.trace(
                        f"Step {step_idx + 1}: Keyword '{resource}' found, adding as resource."
                    )
                resources.add(resource)

        loguru.logger.debug(f"Step {step_idx + 1} identified resources: {resources}")
        return resources

    def can_steps_run_concurrently(self, step1: int, step2: int) -> bool:
        """
        Check if two steps can run concurrently based on dependencies, data conflicts,
        and resource conflicts. Includes detailed logging for rejection reasons.

        The function determines if two steps (`s1`, `s2`, where `s1 < s2`) can run in parallel by checking for three types of conflicts:

        1.  **Path Dependency:** It checks if there's *any* directed path from `s1` to `s2` in the `step_dependencies` graph (`nx.has_path(self.step_dependencies, s1, s2)`).
            *   This graph initially assumes a basic sequence (0->1, 1->2, etc.).
            *   Crucially, the `_add_dataflow_dependencies` function adds *extra* edges. An edge `A -> B` is added if Step `B` uses an entity that was last defined/modified in Step `A`.
            *   **If a path exists, `s1` *must* happen before `s2`, so they cannot be concurrent.** This is the most common reason steps *cannot* run in parallel.

        2.  **Data Conflicts:** It checks if the steps interfere with each other by accessing the same data in conflicting ways:
            *   **Write-Write (WW):** Both steps define/modify the same entity.
            *   **Write-Read (WR):** `s1` defines/modifies an entity that `s2` uses.
            *   **Read-Write (RW):** `s1` uses an entity that `s2` defines/modifies.
            *   **If any data conflict exists, they cannot be concurrent.** (Note: A path dependency often implies a data conflict, so the path check might catch it first).

        3.  **Resource Conflicts:** It checks if both steps require the *same exclusive resource* (like "oven", "stove", "mixer") simultaneously, using the `_get_resources_used` helper function.
            *   **If they need the same exclusive resource, they cannot be concurrent.**

        **Concurrency is only possible (`True`) if *NONE* of these conditions (Path Dependency, Data Conflict, Resource Conflict) are met.**
        """
        # Ensure step1 < step2 for consistent checking, swap if needed
        s1, s2 = min(step1, step2), max(step1, step2)
        loguru.logger.debug(f"Checking concurrency for Steps {s1 + 1} and {s2 + 1}")

        # 1. Check for Path Dependency in the potentially augmented step_dependencies graph
        # If there's a path from s1 to s2, s1 must precede s2.
        if nx.has_path(self.step_dependencies, s1, s2):
            log_path_detail = ""
            try:
                path = nx.shortest_path(self.step_dependencies, s1, s2)
                log_path_detail = f" Path: {[p + 1 for p in path]}"
            except nx.NetworkXNoPath:
                pass  # Should not happen
            loguru.logger.debug(
                f"-> REJECTED (Steps {s1 + 1}, {s2 + 1}): Path exists from {s1 + 1} to {s2 + 1}.{log_path_detail}"
            )
            return False

        # Check reverse path (less likely but good sanity check)
        if nx.has_path(self.step_dependencies, s2, s1):
            loguru.logger.debug(
                f"-> REJECTED (Steps {s1 + 1}, {s2 + 1}): Path exists from {s2 + 1} to {s1 + 1} (unexpected?)."
            )
            return False

        # 2. Check for Data Conflicts
        entities_written1 = {
            e for e, ent in self.entities.items() if s1 in ent.defined_in
        }
        entities_read1 = {e for e, ent in self.entities.items() if s1 in ent.used_in}
        entities_written2 = {
            e for e, ent in self.entities.items() if s2 in ent.defined_in
        }
        entities_read2 = {e for e, ent in self.entities.items() if s2 in ent.used_in}

        # Write-Write conflict
        ww_conflict = entities_written1.intersection(entities_written2)
        if ww_conflict:
            loguru.logger.debug(
                f"-> REJECTED (Steps {s1 + 1}, {s2 + 1}): Write-Write conflict on {ww_conflict}"
            )
            return False

        # Write-Read conflict (s1 writes, s2 reads)
        wr_conflict = entities_written1.intersection(entities_read2)
        if wr_conflict:
            loguru.logger.debug(
                f"-> REJECTED (Steps {s1 + 1}, {s2 + 1}): Write-Read conflict (S{s1 + 1} writes, S{s2 + 1} reads) on {wr_conflict}"
            )
            return False

        # Read-Write conflict (s1 reads, s2 writes)
        rw_conflict = entities_read1.intersection(entities_written2)
        if rw_conflict:
            loguru.logger.debug(
                f"-> REJECTED (Steps {s1 + 1}, {s2 + 1}): Read-Write conflict (S{s1 + 1} reads, S{s2 + 1} writes) on {rw_conflict}"
            )
            return False

        # 3. Check for Resource Conflicts
        resources1 = self._get_resources_used(s1)
        resources2 = self._get_resources_used(s2)
        loguru.logger.trace(f"  Resources S{s1 + 1}: {resources1}")
        loguru.logger.trace(f"  Resources S{s2 + 1}: {resources2}")
        resource_conflict = resources1.intersection(resources2)
        if resource_conflict:
            loguru.logger.debug(
                f"-> REJECTED (Steps {s1 + 1}, {s2 + 1}): Resource conflict on {resource_conflict}"
            )
            return False

        # 4. If no conflicts, they can run concurrently
        loguru.logger.debug(
            f"-> ACCEPTED: Steps {s1 + 1} and {s2 + 1} can run concurrently."
        )
        return True


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

        What does it mean for an entity to be "live"?

        In the context of the procedural text framework, drawing analogy from software analysis: An entity (like an ingredient, tool, or intermediate product) is considered **"live"** after a specific step (say, Step `k`) if that entity **will be used** in at least one subsequent step (Step `j`, where `j > k`).

        - Future Use: Liveness is about whether the entity is needed *later* in the procedure.
        - Specific Point: Liveness is always relative to a point in the procedure (i.e., *after* a particular step).
        - Implication: If an entity is *not* live after Step `k`, it means it's no longer required for any remaining steps. It might have been fully consumed, transformed into something else entirely, or simply won't be referenced again. This is analogous to how a variable that's not live in software can potentially be discarded or its register reused.

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

        question = f"Are the {entity_name} from Step {step + 1} live at any later step?"
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
        entity_found_in_step = False

        for token in doc:
            # Check if token relates to the entity (simple substring/lemma check for illustration)
            if entity_name in token.text.lower() or entity_name == token.lemma_.lower():
                entity_found_in_step = True
                head = token.head
                # Check if the head is a verb and the entity is its object/subject
                if head.pos_ == "VERB" and token.dep_ in {
                    "dobj",
                    "pobj",
                    "nsubjpass",
                    "attr",
                    "conj",
                }:
                    # Exclude auxiliary verbs like 'be', 'have' unless they are the main verb
                    if head.lemma_ not in {"be", "have"}:
                        action = head.lemma_
                        loguru.logger.debug(
                            f"Found action '{action}' governing entity '{entity_name}' via dep '{token.dep_}' in step {dependent_step + 1}"
                        )
                        break  # Found the most likely action for this entity

        # Fallback if no specific action found governing the entity
        if not action:
            if entity_found_in_step:
                loguru.logger.warning(
                    f"Could not find specific verb governing '{entity_name}' in step {dependent_step + 1}. Falling back to 'use'."
                )
                action = "use"
            else:
                # If the entity wasn't even found in the step text (shouldn't happen if parsing was correct)
                loguru.logger.error(
                    f"Entity '{entity_name}' expected but not found in step {dependent_step + 1} text: '{step_text}'. Cannot generate question."
                )
                # Return None or raise error, as the premise is flawed
                return (
                    f"Could not find entity '{entity_name}' in dependent step text.",
                    None,
                )

        # Generate question using the found or fallback action
        question = f"If we skip Step {prerequisite_step + 1}, is it still valid to {action} the {entity_name} in Step {dependent_step + 1}?"

        # Ground truth - if prerequisite step is skipped, the dependent step is invalid
        ground_truth = False

        return question, ground_truth

    def generate_taint_analysis_question(self) -> Tuple[str, Optional[bool]]:
        """
        Generate a question focused on taint analysis of the procedural text.

        The method identifies potential "tainted" entities, raw ingredients or those flagged by specific keywords, and asks whether their use in a particular step might introduce safety concerns.

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
    log_file = os.path.join(args.output_folder, "processing.log")
    loguru.logger.add(
        log_file, level=args.log_level.upper(), rotation="10 MB", compression="zip"
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
