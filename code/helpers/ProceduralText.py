import re
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, List, Set

import loguru
import networkx as nx


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
    def __init__(self, goal: str, steps: List[str], nlp: object):
        self.goal = goal
        self.steps = steps
        self.entities = {}  # name -> Entity
        self.step_dependencies = nx.DiGraph()  # Directed graph for step dependencies
        self.entity_flow_graph = nx.DiGraph()  # Graph for entity flow
        self.nlp = nlp

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

        # Initialize step dependencies graph - ADD NODES AND DEFAULT SEQUENTIAL EDGES
        loguru.logger.debug(
            "Initializing step dependency graph with nodes and default sequential edges..."
        )
        nodes_added = 0
        default_edges_added = 0
        for i in range(len(steps)):
            self.step_dependencies.add_node(i)
            nodes_added += 1
            # Add default edge from previous step to current step
            if i > 0:
                # Check if edge already exists (unlikely here, but safe)
                if not self.step_dependencies.has_edge(i - 1, i):
                    self.step_dependencies.add_edge(i - 1, i)
                    default_edges_added += 1
        loguru.logger.info(
            f"Added {nodes_added} nodes and {default_edges_added} default sequential edges."
        )

        # Parse steps to extract entities and their relationships
        self._parse_steps()  # Extract entities/roles, potentially mark concurrency hints
        self.build_entity_flow_graph()  # Builds separate graph
        self._add_dataflow_dependencies()  # Add *additional* dataflow edges based on entity usage

        # Log all variables
        loguru.logger.info(
            f"\n{len(self.steps)} steps:\n{pformat(list(zip(range(1, len(self.steps) + 1), self.steps)))}"
            f"\nParsed {len(self.entities)} entities:\n{pformat(self.entities)}"
            f"\nStep dependencies (Nodes: {self.step_dependencies.number_of_nodes()}, Edges: {self.step_dependencies.number_of_edges()}):\n{pformat(sorted(list(self.step_dependencies.edges())))}"  # Sort edges for consistent logging
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
        doc = self.nlp(step_text)
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

            # Check exact cleaned text against compiled regex patterns
            if any(
                pattern.match(clean_chunk_text) for pattern in compiled_filter_patterns
            ):
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (pattern match: '{clean_chunk_text}')"
                )
                continue

            # Check root lemma against general stop lemmas
            is_essential = root_lemma in self.essential_tools
            if is_essential:
                loguru.logger.trace(
                    f"Keeping chunk '{chunk.text}' because root lemma '{root_lemma}' is an essential tool."
                )
                # Skip other stop list checks if it's essential
                pass  # Let it proceed to the potential_entities addition
            else:
                # If not essential, apply normal stop list checks
                # Check root lemma against general stop lemmas
                if root_lemma in stop_entity_lemmas:
                    loguru.logger.trace(
                        f"Skipping chunk '{chunk.text}' (root lemma '{root_lemma}' in stop list)"
                    )
                    continue  # Skip to next chunk
                # Check full cleaned text against general stop lemmas
                if clean_chunk_text in stop_entity_lemmas:
                    loguru.logger.trace(
                        f"Skipping chunk '{chunk.text}' (full text '{clean_chunk_text}' in stop list)"
                    )
                    continue  # Skip to next chunk
                # Check full cleaned text against VERB stop list
                if clean_chunk_text in verb_stop_list:
                    loguru.logger.trace(
                        f"Skipping chunk '{chunk.text}' (full text '{clean_chunk_text}' in VERB stop list)"
                    )
                    continue  # Skip to next chunk

            # Check if root is a verb/adjective (only if NOT essential)
            # Also add check for Noun/Propn before filtering based on verb list
            if not is_essential:
                if chunk.root.pos_ not in {"NOUN", "PROPN"}:
                    # If the root isn't a noun, THEN check if its lemma is a verb to filter actions
                    if root_lemma in verb_stop_list:
                        loguru.logger.trace(
                            f"Skipping chunk '{chunk.text}' (root POS {chunk.root.pos_}, lemma '{root_lemma}' in VERB stop list)"
                        )
                        continue
                    else:
                        # If root is not Noun/Propn and not a known verb, maybe still skip?
                        loguru.logger.trace(
                            f"Skipping chunk '{chunk.text}' (root POS is {chunk.root.pos_})"
                        )
                        continue
                else:
                    pass  # It's a noun/propn, let it proceed

            # Length check (on cleaned text)
            if len(clean_chunk_text) < 2:
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (short length: '{clean_chunk_text}')"
                )
                continue

            # Check if the chunk seems to be just a quantity + unit (missed by patterns)
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
            # Verbs indicating transformation
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
            # Cooking verbs
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
                "poach",
                "toast",
                "microwave",
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

                            if verb_lemma in cooking_verbs and token.dep_ in {
                                "dobj",
                                "nsubjpass",
                                "attr",
                            }:  # Check if entity is object/subject of cooking
                                entity_obj = self.entities[entity_name]
                                entity_obj.states[step_idx] = "cooked"
                                loguru.logger.info(
                                    f"State Change: Marked '{entity_name}' as 'cooked' in step {step_idx + 1} due to verb '{verb_lemma}'"
                                )
                                # Propagate 'cooked' state forward (simple version)
                                for future_step in range(step_idx + 1, len(self.steps)):
                                    # Only update if not already set to something else potentially
                                    if future_step not in entity_obj.states:
                                        entity_obj.states[future_step] = "cooked"
                                    # # Or, more robustly, only if the previous state was cooked
                                    # elif entity_obj.states.get(future_step -1) == 'cooked':
                                    #    entity_obj.states[future_step] = 'cooked'

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

                    if not redefined_between:
                        # Add edge ONLY IF IT DOESN'T EXIST. This prevents adding a duplicate
                        # if the default i -> i+1 edge already covers this dataflow.
                        # However, adding dataflow edges even if default exists can be useful
                        # for visualizing specific dependencies.
                        edge_exists = self.step_dependencies.has_edge(
                            latest_def_step, use_step
                        )
                        if not edge_exists:
                            self.step_dependencies.add_edge(latest_def_step, use_step)
                            loguru.logger.debug(
                                f"Added dataflow edge: {latest_def_step + 1} -> {use_step + 1} (for entity '{entity_name}')"
                            )
                            edges_added += 1
                    else:
                        loguru.logger.trace(
                            f"Dataflow dependency {latest_def_step + 1} -> {use_step + 1} (for entity '{entity_name}') already covered by existing edge."
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
        doc = self.nlp(step_text)

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

        Relies on the step_dependencies graph which includes default i->i+1 edges plus specific dataflow edges.
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
