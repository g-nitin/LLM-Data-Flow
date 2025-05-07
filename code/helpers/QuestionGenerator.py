import random
import re
from typing import List, Optional, Tuple

import loguru

from . import constants
from .ProceduralText import ProceduralText


class QuestionGenerator:
    def __init__(self, procedural_text: ProceduralText, nlp: object):
        """
        Initialize the QuestionGenerator with a ProceduralText instance.

        :param procedural_text: An instance of ProceduralText containing the recipe steps and parsed entities.
        :param nlp: A spaCy NLP model instance.
        """
        self.text = procedural_text
        self.nlp = nlp

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
            busy_entity_questions.append(
                (
                    question,
                    True
                    if len(use_steps) > 1
                    else False
                    if len(use_steps) == 1
                    else False,
                )
            )

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

            for step_idx_checked_after in range(start_step, last_possible_step):
                # Check if entity is used in any step strictly after 'step_idx_checked_after'
                is_live = any(
                    use_step > step_idx_checked_after for use_step in entity.used_in
                )

                # Avoid asking about liveness if the entity was consumed at or before this step
                consumed_before_or_at_step = any(
                    cons_step <= step_idx_checked_after
                    for cons_step in entity.consumed_in
                )

                if not consumed_before_or_at_step:
                    if is_live:
                        live_entities.append((entity_name, step_idx_checked_after))
                    else:
                        # Only consider non-live if it wasn't consumed earlier
                        non_live_entities.append((entity_name, step_idx_checked_after))

        # Ensure we don't have duplicates (e.g., entity live after step 2 and step 3)
        live_entities = list(set(live_entities))
        non_live_entities = list(set(non_live_entities))

        # Choose between live and non-live entities to generate a mix of True/False questions
        chosen_entity_info = None
        ground_truth = None

        can_choose_live = bool(live_entities)
        can_choose_non_live = bool(non_live_entities)

        if can_choose_live and (not can_choose_non_live or random.random() > 0.5):
            chosen_entity_info = random.choice(live_entities)
            ground_truth = True
        elif can_choose_non_live:
            chosen_entity_info = random.choice(non_live_entities)
            ground_truth = False

        if chosen_entity_info is None:
            return (
                "Could not find a suitable scenario for live variable analysis question.",
                None,
            )

        entity_name, step_num_checked_after = chosen_entity_info

        question = f"Is {entity_name} live after Step {step_num_checked_after + 1}?"
        loguru.logger.debug(
            f"Generated Liveness Q: '{question}' -> {ground_truth}. Entity: {entity_name}, Step checked after: {step_num_checked_after + 1}, Used in: {self.text.entities[entity_name].used_in}"
        )

        return question, ground_truth

    def generate_interval_analysis_question(self) -> Tuple[str, Optional[str]]:
        """
        Generate a question regarding time interval analysis in the procedural text.

        It selects a step that specifies a time interval (either as an exact value or a range)
        and forms a question asking for that specific time interval detail.
        The ground truth will be the exact text matched from the step.

        :return: A tuple with the question and the ground truth interval (as a string from the step).
                 If no time interval is associated with any step, returns an explanatory message and None.
        """
        steps_with_intervals = []

        for step_idx_loop in self.text.step_dependencies.nodes():
            if "time_interval" in self.text.step_dependencies.nodes[step_idx_loop]:
                steps_with_intervals.append(step_idx_loop)

        if not steps_with_intervals:
            return "No time intervals found in the steps.", None

        # Choose a random step with a time interval
        chosen_step_idx = random.choice(steps_with_intervals)
        interval_data = self.text.step_dependencies.nodes[chosen_step_idx][
            "time_interval"
        ]
        ground_truth = interval_data[3]

        # Generate question
        question = (
            f"What is the last time interval specified in Step {chosen_step_idx + 1}?"
        )

        loguru.logger.debug(
            f"Generated Interval Q: '{question}' -> '{ground_truth}' from data {interval_data} for step {chosen_step_idx + 1}"
        )
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
        doc = self.nlp(step_text)
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
        Generate a question focused on taint analysis using entity state tracking.

        Identifies potentially "tainted" entities (raw ingredients) and asks
        if their use introduces a safety concern, considering if they are ever cooked.

        :return: A tuple with the generated question and a boolean ground truth.
                 Returns explanatory message and None if no suitable scenario found.
        """
        potential_concerns = []
        # This list contains lowercase entity names that are inherently unsafe if not cooked.

        for entity_name, entity in self.text.entities.items():
            # entity.name is expected to be lowercase from ProceduralText parsing.

            # Determine if the entity is potentially unsafe initially.
            is_potentially_unsafe_initially = False
            if entity_name in constants.QGEN_UNSAFE_ENTITY_NAMES_TAINT:
                is_potentially_unsafe_initially = True
            elif entity_name.startswith("raw "):  # e.g., entity name is "raw fish"
                is_potentially_unsafe_initially = True
            else:
                # Check if the introduction step text explicitly mentions "raw {entity_name}"
                intro_step_text_lower = self.text.steps[entity.step_introduced].lower()
                try:
                    # Pattern to find "raw" immediately preceding the entity_name, as whole words.
                    # re.escape handles entity names that might contain regex special characters.
                    pattern_raw_entity = rf"\braw\s+{re.escape(entity_name)}\b"
                    if re.search(pattern_raw_entity, intro_step_text_lower):
                        is_potentially_unsafe_initially = True
                except re.error as e:
                    loguru.logger.warning(
                        f"Regex error during taint analysis for entity '{entity_name}' in recipe '{self.text.goal}': {e}. Step text: '{intro_step_text_lower}'"
                    )

            if not is_potentially_unsafe_initially:
                continue  # Skip if not considered unsafe by the refined criteria

            # Check if this entity is ever marked as 'cooked'
            # This relies on the entity itself being the direct object of a cooking verb.
            is_ever_cooked = any(state == "cooked" for state in entity.states.values())

            # Find the first step index where this entity was marked as cooked, if any.
            first_cooked_step = -1
            if is_ever_cooked:
                # Ensure states dictionary is not empty and contains 'cooked'
                cooked_steps = [
                    step_idx_loop
                    for step_idx_loop, state in entity.states.items()
                    if state == "cooked"  # Renamed 'step'
                ]
                if cooked_steps:
                    first_cooked_step = min(cooked_steps)

            for use_step in entity.used_in:
                # Determine if the entity is used before it's cooked (or if it's never cooked)
                is_used_while_potentially_raw = (
                    first_cooked_step == -1 or use_step < first_cooked_step
                )

                if is_used_while_potentially_raw:
                    # The overall concern for the recipe exists if the unsafe entity is *never* cooked.
                    overall_concern_exists = not is_ever_cooked
                    potential_concerns.append(
                        (entity_name, use_step, overall_concern_exists)
                    )
                    loguru.logger.debug(
                        f"Taint Candidate: Entity '{entity_name}' (unsafe) used in step {use_step + 1} "
                        f"while potentially raw. Ever cooked (direct): {is_ever_cooked}. Overall recipe concern: {overall_concern_exists}"
                    )

        if not potential_concerns:
            return (
                "No potential safety concerns identified based on refined unsafe criteria and cooking state.",
                None,
            )

        # Choose a random potential concern scenario
        entity_name, step_of_use, ground_truth = random.choice(potential_concerns)

        question = f"Does using {entity_name} in Step {step_of_use + 1} introduce a potential safety concern to the recipe?"

        # The ground_truth reflects whether this raw usage leads to an *uncooked final product*
        # with respect to this specific entity being directly cooked.
        loguru.logger.info(
            f"Generated Taint Q: '{question}' -> {ground_truth} (Entity: {entity_name}, Used raw in step: {step_of_use + 1}, Entity ever directly cooked: {not ground_truth})"
        )

        return question, ground_truth

    def generate_concurrency_analysis_question(
        self, max_retries=10
    ) -> Tuple[str, Optional[bool]]:
        """
        Generate a question regarding concurrency analysis among steps.

        Randomly selects a pair of distinct steps and determines if they
        *can* run concurrently based on data and resource dependencies.
        Attempts to filter questions where simplified verbs suggest concurrency
        but the actual answer is False due to hidden dependencies within steps.

        :param max_retries: Maximum attempts to find a non-misleading question pair.
        :return: A tuple containing the concurrency question and a boolean ground truth.
            Returns explanatory message and None if fewer than 2 steps exist.
        """
        num_steps = len(self.text.steps)
        if num_steps < 2:
            loguru.logger.warning("Skipping concurrency question: Fewer than 2 steps.")
            return "Not enough steps for concurrency analysis.", None

        for attempt in range(max_retries):
            # Randomly select two distinct step indices
            try:
                step1_idx, step2_idx = random.sample(range(num_steps), 2)
            except ValueError:
                loguru.logger.error("Error sampling steps for concurrency question.")
                return "Error selecting steps for concurrency analysis.", None

            # Determine the ground truth using the ProceduralText method
            ground_truth = self.text.can_steps_run_concurrently(step1_idx, step2_idx)

            # Extract key actions for the question
            s1, s2 = min(step1_idx, step2_idx), max(step1_idx, step2_idx)
            step1_text = self.text.steps[s1]
            step2_text = self.text.steps[s2]

            action1 = "perform action"
            try:
                doc1 = self.nlp(step1_text)
                # Find first non-auxiliary verb lemma
                action1 = next(
                    (
                        token.lemma_
                        for token in doc1
                        if token.pos_ == "VERB" and token.lemma_ not in {"be", "have"}
                    ),
                    "perform action",
                )
            except Exception as e:
                loguru.logger.warning(
                    f"NLP processing failed for step {s1 + 1} text: '{step1_text}'. Error: {e}"
                )

            action2 = "perform action"
            try:
                doc2 = self.nlp(step2_text)
                # Find first non-auxiliary verb lemma
                action2 = next(
                    (
                        token.lemma_
                        for token in doc2
                        if token.pos_ == "VERB" and token.lemma_ not in {"be", "have"}
                    ),
                    "perform action",
                )
            except Exception as e:
                loguru.logger.warning(
                    f"NLP processing failed for step {s2 + 1} text: '{step2_text}'. Error: {e}"
                )

            # Filtering Logic for Potentially Misleading "No" Answers
            is_potentially_misleading = False
            if ground_truth is False:
                # Check if it's a background prep + active prep verb combination
                combo1 = (
                    action1 in constants.QGEN_BACKGROUND_PREP_VERBS
                    and action2 in constants.QGEN_ACTIVE_PREP_VERBS
                )
                combo2 = (
                    action2 in constants.QGEN_BACKGROUND_PREP_VERBS
                    and action1 in constants.QGEN_ACTIVE_PREP_VERBS
                )

                # Also check if both are background verbs (e.g., measure and sift) which might also be misleading if False
                combo3 = (
                    action1 in constants.QGEN_BACKGROUND_PREP_VERBS
                    and action2 in constants.QGEN_BACKGROUND_PREP_VERBS
                )

                if combo1 or combo2 or combo3:
                    # If the answer is False for these combinations, it's likely due to
                    # hidden dependencies (like needing a prepared pan) rather than
                    # a direct conflict between the *named actions*. Skip it.
                    is_potentially_misleading = True
                    loguru.logger.debug(
                        f"Skipping potentially misleading concurrency question (Attempt {attempt + 1}/{max_retries}): "
                        f"'Can we {action1} (S{s1 + 1}) and {action2} (S{s2 + 1})?' "
                        f"(Actual Answer: False, Verbs: '{action1}', '{action2}')"
                    )

            if not is_potentially_misleading:
                # Generate the question text
                question = f"Can we {action1} (Step {s1 + 1}) and {action2} (Step {s2 + 1}) at the same time?"
                loguru.logger.debug(
                    f"Generated Concurrency Q: '{question}' -> {ground_truth} (Steps checked: {step1_idx + 1}, {step2_idx + 1})"
                )
                return question, ground_truth  # Found a suitable question

        # If loop finishes without returning a question
        loguru.logger.warning(
            f"Could not generate a non-misleading concurrency question after {max_retries} attempts."
        )
        return (
            "Could not find suitable steps for non-misleading concurrency analysis.",
            None,
        )

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
