import re
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, List, Optional, Set, Tuple

import loguru
import networkx as nx

from . import constants


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
    _COMPILED_TIME_PATTERNS_WITH_PRIORITY = [
        (re.compile(pattern_str, re.IGNORECASE), type_str, priority_val)
        for pattern_str, type_str, priority_val in constants.TIME_PATTERN_CONFIG
    ]

    def __init__(self, goal: str, steps: List[str], nlp: object):
        self.goal = goal
        self.steps = steps
        self.entities = {}  # name -> Entity
        self.step_dependencies = nx.DiGraph()  # Directed graph for step dependencies
        self.entity_flow_graph = nx.DiGraph()  # Graph for entity flow
        self.nlp = nlp

        self.exclusive_resources = constants.EXCLUSIVE_RESOURCES.copy()
        self.verb_to_resource_map = constants.VERB_TO_RESOURCE_MAP.copy()

        # Define essential tools based on resources
        self.essential_tools = constants.EXCLUSIVE_RESOURCES.union(
            constants.ADDITIONAL_ESSENTIAL_TOOLS
        )

        loguru.logger.debug("Initializing step dependency graph...")
        nodes_added = 0
        default_edges_added = 0
        for i in range(len(steps)):
            self.step_dependencies.add_node(i)
            nodes_added += 1
            # Add default edge from previous step to current step
            if i > 0 and not self.step_dependencies.has_edge(i - 1, i):
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

    def _canonical_unit(self, unit_str: str) -> str:
        """Converts various unit strings to a canonical plural form."""
        unit_str_lower = unit_str.lower().strip()
        if re.match(rf"^{constants.UNIT_HR_PATTERN_STR}$", unit_str_lower):
            return "hours"
        if re.match(rf"^{constants.UNIT_MIN_PATTERN_STR}$", unit_str_lower):
            return "minutes"
        if re.match(rf"^{constants.UNIT_SEC_PATTERN_STR}$", unit_str_lower):
            return "seconds"
        # Fallback if somehow a unit string doesn't match known patterns
        loguru.logger.warning(
            f"Unknown unit string encountered in _canonical_unit: '{unit_str}'"
        )
        return unit_str_lower

    def _parse_numeric_string(self, value_str: str) -> float:
        """Parses a string (integer, decimal, or fraction) into a float."""
        value_str = value_str.strip()
        if "/" in value_str:
            parts = value_str.split("/")
            if len(parts) == 2:
                try:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    if denominator == 0:
                        loguru.logger.error(
                            f"Denominator cannot be zero in fraction: {value_str}"
                        )
                        raise ValueError("Denominator cannot be zero in fraction.")
                    return numerator / denominator
                except ValueError as e:
                    loguru.logger.error(
                        f"Could not parse fraction parts: {value_str} ({e})"
                    )
                    raise ValueError(f"Could not parse fraction: {value_str}") from e
            else:
                loguru.logger.error(
                    f"Invalid fraction format (not two parts): {value_str}"
                )
                raise ValueError(f"Invalid fraction format: {value_str}")
        else:
            try:
                return float(value_str)
            except ValueError as e:
                loguru.logger.error(
                    f"Could not parse numeric string as float: {value_str} ({e})"
                )
                raise ValueError(
                    f"Could not parse numeric string as float: {value_str}"
                ) from e

    def _parse_time_value_from_match(
        self, match: re.Match, pattern_type: str
    ) -> Optional[Tuple[float, float, str]]:
        """Parses time values from a regex match object based on pattern type."""
        try:
            if pattern_type == "hr_min":
                hr = int(match.group(1))  # Hours are typically integers
                minute = int(match.group(2))  # Minutes are typically integers
                total_minutes = float(hr * 60 + minute)  # Result as float
                return total_minutes, total_minutes, "minutes"
            elif pattern_type == "min_sec":
                minute = int(match.group(1))  # Minutes are typically integers
                sec = int(match.group(2))  # Seconds are typically integers
                total_seconds = float(minute * 60 + sec)  # Result as float
                return total_seconds, total_seconds, "seconds"
            elif pattern_type == "range":
                val1_str = match.group(1)
                val2_str = match.group(2)
                val1 = self._parse_numeric_string(val1_str)
                val2 = self._parse_numeric_string(val2_str)
                unit_str = match.group(3)
                return val1, val2, self._canonical_unit(unit_str)
            elif pattern_type == "exact":
                val_str = match.group(1)
                val1 = self._parse_numeric_string(val_str)
                unit_str = match.group(2)
                return val1, val1, self._canonical_unit(unit_str)
        except (
            ValueError,
            IndexError,
        ) as e:  # ValueError can be raised by _parse_numeric_string
            loguru.logger.error(
                f"Error parsing time from match (type: {pattern_type}, match groups: {match.groups()}): {e}"
            )
            return None
        return None  # Should not be reached if all paths return

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

        # Combine stop lemmas
        stop_entity_lemmas = (
            constants.MEASUREMENT_UNITS.union(constants.CONTAINERS_TOOLS_FILTER)
            .union(constants.LOCATIONS_QUALITIES_FILTER)
            .union(constants.ABSTRACT_GENERIC_FILTER)
        )

        # Pre-compile regex for efficiency
        compiled_filter_patterns = [
            re.compile(p, re.IGNORECASE) for p in constants.ENTITY_FILTER_REGEX_PATTERNS
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
                "NUM",
                "PUNCT",
                "PART",
                "SCONJ",
                "CCONJ",
            }:
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (first token POS: {first_token_pos})"
                )
                continue

            # Check exact cleaned text against compiled regex patterns
            if any(
                pattern.fullmatch(clean_chunk_text)
                for pattern in compiled_filter_patterns
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
            else:
                # If not essential, apply normal stop list checks
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
                    continue
                if (
                    clean_chunk_text in constants.VERB_STOP_LIST_FOR_ENTITY_FILTERING
                ):  # Check against verbs if it's not an essential tool
                    loguru.logger.trace(
                        f"Skipping chunk '{chunk.text}' (full text '{clean_chunk_text}' in VERB stop list)"
                    )
                    continue  # Skip to next chunk

            # Check if root is a verb/adjective (only if NOT essential)
            # Also add check for Noun/Propn before filtering based on verb list
            if not is_essential:
                if chunk.root.pos_ not in {"NOUN", "PROPN"}:
                    # If the root isn't a noun, THEN check if its lemma is a verb to filter actions
                    if root_lemma in constants.VERB_STOP_LIST_FOR_ENTITY_FILTERING:
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
                and chunk[1].lemma_ in constants.MEASUREMENT_UNITS
            ):
                loguru.logger.trace(f"Skipping chunk '{chunk.text}' (quantity+unit)")
                continue

            # If passes filters, add it
            entity_lemma = root_lemma  # Using root lemma as the key

            # Canonical naming for essential tools
            if entity_lemma in self.essential_tools:
                # For essential tools, always use the canonical lemma as the entity name
                current_stored_name = potential_entities.get(entity_lemma)
                # We want the value to be entity_lemma (canonical name)
                # Only update if it's not already set to the canonical name, or if it's not present
                if (
                    current_stored_name != entity_lemma
                ):  # This covers not present (None != lemma) and different
                    potential_entities[entity_lemma] = (
                        entity_lemma  # Store/overwrite with canonical name
                    )
                    loguru.logger.info(
                        f"Storing/Updating ESSENTIAL TOOL: Chunk '{chunk.text}' (lemma '{root_lemma}') -> Canonical Name: '{entity_lemma}'. Was: '{current_stored_name if current_stored_name is not None else 'Not present'}'"
                    )
            elif (
                entity_lemma not in potential_entities
                or len(clean_chunk_text) > len(potential_entities[entity_lemma])
            ):  # Logic for non-essential tools (prefer longer chunk text for same lemma)
                potential_entities[entity_lemma] = clean_chunk_text
                loguru.logger.info(
                    f"Keeping potential entity: '{chunk.text}' -> Cleaned: '{clean_chunk_text}', Key Lemma: '{entity_lemma}'"
                )
            else:  # Existing entity_lemma for non-essential tool, and current clean_chunk_text is not longer
                loguru.logger.trace(
                    f"Skipping chunk '{chunk.text}' (lemma '{entity_lemma}' exists with '{potential_entities[entity_lemma]}' and new one is not longer, or it's an essential tool already correctly named)"
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

        # Use noun chunks from the step text for checking derived forms
        for chunk in doc.noun_chunks:
            # Basic cleaning similar to entity extraction
            chunk_text = chunk.text.lower().strip()
            chunk_text = re.sub(
                r"^(the|a|an|some|any|its|the)\s+", "", chunk_text
            ).strip()
            if len(chunk_text.split()) < 2:  # Need at least two words for these rules
                continue

            for suffix, method in constants.DERIVED_ENTITY_RULES.items():
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
                    target_entity = self.entities.get(base_entity_name)
                    if target_entity:
                        # Check if the base entity was active (defined or used) before this step
                        was_active_before = any(
                            s < step_idx
                            for s in target_entity.defined_in.union(
                                target_entity.used_in
                            )
                        )
                        if was_active_before and step_idx not in target_entity.used_in:
                            loguru.logger.info(
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
            is_used, is_defined, is_consumed = False, False, False
            governing_verb_lemma = None

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
                        for nc_chunk in token.sent.noun_chunks:
                            # Compare cleaned chunk text with entity name
                            clean_nc_text = nc_chunk.text.lower().strip()
                            clean_nc_text = re.sub(
                                r"^(the|a|an|some|your|my|his|her|its|our|their|of|for|with|in|on|at|to|about|approx\.?|approximately|around|over|under|less than|more than|at least)\s+",
                                "",
                                clean_nc_text,
                            ).strip()
                            if clean_nc_text == entity_name_lower and token in nc_chunk:
                                is_entity_match = True
                                break
                    # Fallback to simple check if chunk matching fails or is complex
                    if not is_entity_match and (
                        entity_name_lower in token_text_lower
                        or entity_name_lower == token_lemma_lower
                    ):
                        is_entity_match = True  # Less precise fallback

                    if is_entity_match:
                        entity_found_in_step = True

                        # Improved governing verb identification
                        actual_verb_for_entity = None
                        # Scenario 1: Entity is a direct dependent of a verb (e.g., dobj, nsubj)
                        # Valid direct dependencies for an entity to be acted upon by a verb:
                        if (
                            token.head.pos_ == "VERB"
                            and token.dep_
                            in constants.VALID_DIRECT_DEPENDENCIES_FOR_ENTITY_VERB
                        ):
                            actual_verb_for_entity = token.head
                        # Scenario 2: Entity is the object of a preposition, and the preposition is attached to a verb
                        elif (
                            token.dep_ == "pobj" and token.head.pos_ == "ADP"
                        ):  # token.head is the preposition
                            if token.head.head.pos_ == "VERB":
                                actual_verb_for_entity = token.head.head

                        if actual_verb_for_entity:
                            verb_action_found = True
                            governing_verb_lemma = actual_verb_for_entity.lemma_.lower()
                            loguru.logger.trace(
                                f"Entity '{entity_name}' governed by verb '{governing_verb_lemma}' ({actual_verb_for_entity.text}) (original entity dep: '{token.dep_}' to '{token.head.text}')"
                            )

                            # Assign roles based on verb lists
                            if (
                                governing_verb_lemma
                                in constants.CONSUME_VERBS_ROLE_ASSIGNMENT
                            ):
                                is_consumed = True
                                loguru.logger.debug(
                                    f"Marking '{entity_name}' as CONSUMED by verb '{governing_verb_lemma}'"
                                )
                            elif (
                                governing_verb_lemma
                                in constants.DEFINE_VERBS_ROLE_ASSIGNMENT
                            ):
                                is_defined = True
                                loguru.logger.debug(
                                    f"Marking '{entity_name}' as DEFINED by verb '{governing_verb_lemma}'"
                                )
                            elif (
                                governing_verb_lemma
                                in constants.USE_VERBS_ROLE_ASSIGNMENT
                            ):  # This elif ensures define_verbs takes precedence if a verb is in both
                                is_used = True
                                loguru.logger.debug(
                                    f"Marking '{entity_name}' as USED by verb '{governing_verb_lemma}'"
                                )
                            else:
                                loguru.logger.trace(
                                    f"Verb '{governing_verb_lemma}' acting on '{entity_name}' not in standard role lists."
                                )
                            #  Cooking state logic (applies to ingredients, not typically tools like oven)
                            # Check if the entity is a direct argument of the cooking verb

                            if (
                                governing_verb_lemma
                                in constants.COOKING_VERBS_ROLE_ASSIGNMENT
                                and (
                                    actual_verb_for_entity == token.head
                                    and token.dep_ in {"dobj", "nsubjpass", "attr"}
                                )
                            ):
                                entity_obj = self.entities[entity_name]
                                entity_obj.states[step_idx] = "cooked"
                                loguru.logger.info(
                                    f"State Change: Marked '{entity_name}' as 'cooked' in step {step_idx + 1} due to verb '{governing_verb_lemma}'"
                                )
                                for future_step in range(step_idx + 1, len(self.steps)):
                                    if future_step not in entity_obj.states:
                                        entity_obj.states[future_step] = "cooked"
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
                    if (
                        governing_verb_lemma
                        in constants.TRANSFORMATION_VERBS_ROLE_ASSIGNMENT
                    ):
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
                        loguru.logger.warning(
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
        for step_idx, step_text in enumerate(
            self.steps
        ):  # step_text has original casing
            # Extract entities and determine their roles (defined, used)
            self._extract_entities_from_step(step_idx, step_text)

            # Time Interval Extraction Logic
            found_intervals_in_step = []
            current_best_interval_details = None

            for (
                regexp,
                type_str,
                priority_val,
            ) in self._COMPILED_TIME_PATTERNS_WITH_PRIORITY:
                # Search on original step_text, rely on re.IGNORECASE in compiled regexp
                for match in regexp.finditer(step_text):
                    parsed_time = self._parse_time_value_from_match(match, type_str)
                    if parsed_time:
                        min_v, max_v, unit_c = parsed_time
                        # Capture original matched text (will have original casing)
                        original_match_text = match.group(0)

                        interval_detail = {
                            "min_val": min_v,  # float
                            "max_val": max_v,  # float
                            "unit": unit_c,
                            "original_text": original_match_text,
                            "end_pos": match.end(),
                            "priority": priority_val,
                            "start_pos": match.start(),
                        }
                        found_intervals_in_step.append(interval_detail)
                        loguru.logger.trace(
                            f"Step {step_idx + 1}: Found potential time interval: {interval_detail}"
                        )
            if found_intervals_in_step:
                # Select the best interval:
                # 1. Maximize end_pos (latest occurrence)
                # 2. Minimize priority (higher priority pattern)
                # (Optional: Maximize start_pos for same end_pos and priority - makes match longer, less likely needed)

                # Find max end_pos
                max_end_pos = -1
                for detail in found_intervals_in_step:
                    if detail["end_pos"] > max_end_pos:
                        max_end_pos = detail["end_pos"]

                # Filter for those with max_end_pos
                candidates_at_max_end_pos = [
                    d for d in found_intervals_in_step if d["end_pos"] == max_end_pos
                ]

                # Sort these candidates by priority (ascending, so lowest number is best)
                # and then by start_pos (descending, so longer/earlier starting match is preferred if all else equal)
                # This secondary sort on start_pos is mostly for determinism if multiple identical patterns match at same end_pos.
                candidates_at_max_end_pos.sort(
                    key=lambda x: (x["priority"], -x["start_pos"])
                )

                current_best_interval_details = candidates_at_max_end_pos[0]

                loguru.logger.debug(
                    f"Step {step_idx + 1}: Selected final time interval: {current_best_interval_details}"
                )
                self.step_dependencies.nodes[step_idx]["time_interval"] = (
                    current_best_interval_details["min_val"],
                    current_best_interval_details["max_val"],
                    current_best_interval_details["unit"],
                    current_best_interval_details["original_text"],
                )
            # else: current_best_interval_details remains None, no time_interval attribute set

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
            loguru.logger.warning(
                f"Entity '{entity_name}' not found for liveness check after step {step_idx + 1}."
            )
            return False

        entity = self.entities[entity_name]
        is_live = any(use_step > step_idx for use_step in entity.used_in)
        loguru.logger.trace(
            f"Liveness check for '{entity_name}' after step {step_idx + 1}: Used in {entity.used_in}. Live: {is_live}"
        )
        return is_live

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
