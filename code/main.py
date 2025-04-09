import random
import re
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, List, Optional, Set, Tuple

import loguru
import networkx as nx
import spacy

# Load spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en_core_web_trf")


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
        Extract entities from a step using NLP techniques with filtering for recipes
        Note that the method is specific to RECIPES.
        """
        doc = nlp(step_text)

        # Extract all potential entities (nouns and noun phrases)
        potential_entities = []

        # Add noun chunks
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop and chunk.root.pos_ != "DET":
                potential_entities.append(chunk.text.lower())

        # Add individual nouns that might not be in chunks
        for token in doc:
            if token.pos_ == "NOUN" and not token.is_stop:
                potential_entities.append(token.text.lower())

        # Define patterns to filter out
        measurement_units = {
            "cup",
            "cups",
            "tablespoon",
            "tablespoons",
            "teaspoon",
            "teaspoons",
            "ounce",
            "ounces",
            "pound",
            "pounds",
            "gram",
            "grams",
            "kilogram",
            "kilograms",
            "milliliter",
            "milliliters",
            "liter",
            "liters",
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
            "hour",
            "hours",
            "second",
            "seconds",
            "degree",
            "degrees",
            "f",
            "c",
            "fahrenheit",
            "celsius",
        }

        # Patterns for measurements with numbers
        measurement_patterns = [
            r"\d+\s*-*\s*\d*\s*(minute|minutes|min|mins|hour|hours|second|seconds|sec|secs)",
            r"\d+\s*(cup|cups|tablespoon|tbsp|teaspoon|tsp|ounce|oz|pound|lb|gram|g|kg|ml|l)",
            r"\d+\s*-*\s*\d*\s*(degree|degrees|°)\s*[fc]",
        ]

        # Process each potential entity
        for entity_name in potential_entities:
            # Skip very common words, short entities, or measurement units
            if (
                len(entity_name) < 3
                or entity_name in measurement_units
                or any(word in entity_name for word in measurement_units)
                or any(
                    re.search(pattern, entity_name, re.IGNORECASE)
                    for pattern in measurement_patterns
                )
                or entity_name
                in {"step", "time", "texture", "pan", "mixture", "batter"}
            ):
                loguru.logger.debug(f"Skipping entity due to filtering: {entity_name}")
                continue

            # Skip entities that are just numbers or measurements
            if re.match(r"^\d+(\.\d+)?$", entity_name) or re.match(
                r"^\d+/\d+$", entity_name
            ):
                loguru.logger.debug(
                    f"Skipping entity due to being a number: {entity_name}"
                )
                continue

            # Skip temperature references
            if "degree" in entity_name or "°" in entity_name:
                loguru.logger.debug(
                    f"Skipping entity due to being a temperature: {entity_name}"
                )
                continue

            # Create or update entity
            if entity_name not in self.entities:
                self.entities[entity_name] = Entity(entity_name, step_idx)
                self.entities[entity_name].defined_in.add(step_idx)

            # Determine entity's role in this step
            is_used = False
            is_defined = False
            is_consumed = False

            # Check for verbs acting on this entity
            for token in doc:
                if token.pos_ == "VERB":
                    # Check if this verb acts on our entity
                    for child in token.children:
                        if entity_name in child.text.lower():
                            # Classify based on verb semantics
                            if token.lemma_ in {
                                "add",
                                "use",
                                "stir",
                                "mix",
                                "combine",
                                "pour",
                                "crack",
                            }:
                                is_used = True
                            elif token.lemma_ in {
                                "create",
                                "make",
                                "prepare",
                                "preheat",
                                "measure",
                                "warm",
                                "peel",
                                "mash",
                            }:
                                is_defined = True
                            elif token.lemma_ in {"eat", "consume", "finish", "remove"}:
                                is_consumed = True

            # Update entity with its role in this step
            if is_used:
                self.entities[entity_name].used_in.add(step_idx)
            if is_defined:
                self.entities[entity_name].defined_in.add(step_idx)
            if is_consumed:
                self.entities[entity_name].consumed_in.add(step_idx)

            # If no specific role was identified, assume it's being used
            if not (is_used or is_defined or is_consumed):
                self.entities[entity_name].used_in.add(step_idx)

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

    def generate_very_busy_expressions_question(self) -> Tuple[str, Optional[bool]]:
        """
        Generate a question about very busy expressions in the procedural text.

        This method identifies entities that, after being defined, are used in multiple future steps,
        indicating that they are "very busy" throughout the process.

        :return: A tuple containing the generated question and a boolean ground truth.
                 If no busy expressions are found, returns an explanatory message and None.
        """
        busy_entities = []

        for entity_name, entity in self.text.entities.items():
            for def_step in entity.defined_in:
                future_uses = [step for step in entity.used_in if step > def_step]
                if len(future_uses) >= 2:  # Used in multiple future steps
                    busy_entities.append((entity_name, def_step))

        if not busy_entities:
            return "No very busy expressions found.", None

        # Choose a random entity and step
        entity_name, step = random.choice(busy_entities)

        # Generate question
        question = f"Is {entity_name} from Step {step + 1} used in multiple future steps without being redefined?"

        # Ground truth is True since we verified multiple future uses
        return question, True

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

        for _ in range(num_per_type):
            questions.append(
                ("Reaching Definitions", self.generate_reaching_definitions_question())
            )
            questions.append(
                (
                    "Very Busy Expressions",
                    self.generate_very_busy_expressions_question(),
                )
            )
            questions.append(
                (
                    "Available Expressions",
                    self.generate_available_expressions_question(),
                )
            )
            questions.append(
                ("Live Variable Analysis", self.generate_live_variable_question())
            )
            questions.append(
                ("Interval Analysis", self.generate_interval_analysis_question())
            )
            questions.append(
                ("Type-State Analysis", self.generate_type_state_question())
            )
            questions.append(
                ("Taint Analysis", self.generate_taint_analysis_question())
            )
            questions.append(
                ("Concurrency Analysis", self.generate_concurrency_analysis_question())
            )

        return questions


# Example usage
def analyze_recipe(recipe_name, steps, goal):
    loguru.logger.info(f"\n----- Analyzing Recipe: {recipe_name} -----")

    # Create procedural text
    proc_text = ProceduralText(goal, steps)

    # Generate questions
    question_gen = QuestionGenerator(proc_text)
    questions = question_gen.generate_all_questions(num_per_type=1)

    # Print questions and answers
    for q_type, (question, answer) in questions:
        logging_str = f"\n{q_type}:\nQ: {question}\nA: {answer}\n"
        if answer is None:
            loguru.logger.warning(logging_str)
        else:
            loguru.logger.info(logging_str)


def main():
    # Setup logging
    # loguru.logger.add("code/debug.log", level="DEBUG")

    # Read in json file from cl
    # parser = argparse.ArgumentParser(description="Analyze procedural text.")
    # parser.add_argument(
    #     "--json_file",
    #     type=str,
    #     help="Path to the JSON file containing the procedural text.",
    # )
    # args = parser.parse_args()

    # Test with sample recipes
    recipes = [
        {
            "name": "Bake Banana Bread",
            "goal": "Bake a delicious banana bread",
            "steps": [
                "Preheat the oven to 350 degree F (this takes about 5 minutes).",  # 1
                "Peel two ripe bananas.",  # 2
                "Mash the peeled bananas in a bowl.",  # 3
                "(Optional) Warm 1/2 cup of milk for a moister texture.",  # 4
                "While the oven is preheating, measure 1 cup of sugar and 2 cups of flour.",  # 5
                "Combine the mashed bananas, sugar, flour, and the optional milk if used.",  # 6
                "Crack 1 raw egg into the mixture and stir thoroughly.",  # 7
                "Pour the batter into a greased loaf pan and bake for 30--35 minutes.",  # 8
                "Let the baked loaf cool for 10 minutes before slicing.",  # 9
            ],
        },
    ]

    # Run analysis on each recipe
    for recipe in recipes:
        analyze_recipe(recipe["name"], recipe["steps"], recipe["goal"])


if __name__ == "__main__":
    main()
