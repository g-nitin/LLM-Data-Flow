# Regex String Patterns (for ProceduralText)
# Unit patterns
UNIT_HR_PATTERN_STR = r"(?:hours?|hrs?\.?)"
UNIT_MIN_PATTERN_STR = r"(?:minutes?|mins?\.?)"
UNIT_SEC_PATTERN_STR = r"(?:seconds?|secs?\.?)"
GENERAL_UNIT_PATTERN_STR = (
    rf"(?:{UNIT_HR_PATTERN_STR}|{UNIT_MIN_PATTERN_STR}|{UNIT_SEC_PATTERN_STR})"
)

# Regex for parsing numbers: fractions, decimals, integers
# Order: fraction, then decimal (with optional integer part), then integer
NUMBER_PATTERN_STR = r"(?:\d+\s*/\s*\d+|\d*\.\d+|\d+)"

# Time parsing patterns
PAT_HR_MIN_RE_STR = (
    # Hours and minutes are typically integers in "X hours Y minutes" format
    rf"(\d+)\s*{UNIT_HR_PATTERN_STR}\s*(?:and\s*)?(\d+)\s*{UNIT_MIN_PATTERN_STR}"
)
PAT_MIN_SEC_RE_STR = (
    # Minutes and seconds are typically integers in "X minutes Y seconds" format
    rf"(\d+)\s*{UNIT_MIN_PATTERN_STR}\s*(?:and\s*)?(\d+)\s*{UNIT_SEC_PATTERN_STR}"
)
# Ensure _GENERAL_UNIT_PATTERN_STR is captured for parsing unit_str from group(3)
PAT_RANGE_RE_STR = (
    # Use _NUMBER_PATTERN_STR for values in a range
    rf"({NUMBER_PATTERN_STR})\s*(?:to|-)\s*({NUMBER_PATTERN_STR})\s*({GENERAL_UNIT_PATTERN_STR})"
)
# Ensure _GENERAL_UNIT_PATTERN_STR is captured for parsing unit_str from group(2)
# Use _NUMBER_PATTERN_STR for exact values
PAT_EXACT_RE_STR = rf"({NUMBER_PATTERN_STR})\s*({GENERAL_UNIT_PATTERN_STR})"

# Configuration for compiled time patterns (used in ProceduralText)
# ProceduralText will compile these.
TIME_PATTERN_CONFIG = [
    (PAT_HR_MIN_RE_STR, "hr_min", 1),
    (PAT_MIN_SEC_RE_STR, "min_sec", 2),
    (PAT_RANGE_RE_STR, "range", 3),
    (PAT_EXACT_RE_STR, "exact", 4),
]

# Resource Definitions (for ProceduralText)
EXCLUSIVE_RESOURCES = {
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
VERB_TO_RESOURCE_MAP = {
    "bake": "oven",
    "roast": "oven",
    "broil": "oven",
    "preheat": "oven",
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
# For essential_tools, it's often a combination or subset.
# If it's EXCLUSIVE_RESOURCES + a few more:
ADDITIONAL_ESSENTIAL_TOOLS = {
    "skillet",
    "saucepan",
    "wok",
    "griddle",
}  # Example if these are not always exclusive but are essential

# Entity Extraction Filter Lists & Rules (for ProceduralText)
MEASUREMENT_UNITS = {
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
CONTAINERS_TOOLS_FILTER = {
    "bowl",
    "pan",
    "pot",
    "skillet",
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

LOCATIONS_QUALITIES_FILTER = {
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
ABSTRACT_GENERIC_FILTER = {
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
# This VERB_STOP_LIST is used for filtering out entities that are actually verbs
VERB_STOP_LIST_FOR_ENTITY_FILTERING = {
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

# Patterns for filtering (e.g., numeric values, specific phrases)
ENTITY_FILTER_REGEX_PATTERNS = [
    r"^\d+(\.\d+)?$",  # Pure numbers
    r"^\d+\s?/\s?\d+$",  # Fractions like 1/2 or 1 / 2
    r"^\d+\s?-\s?\d+$",  # Ranges like 10-15
    r"^\d+x\d+$",  # Dimensions like 9x13
    r"^\d+(?:\s*(?:to|-)\s*\d+)?\s+(?:minute|minutes|min\.?|hour|hours|hr\.?|second|seconds|sec\.?)$",
    r"^\d+(?:\s*(?:to|-)\s*\d+)?\s+(?:degree|degrees|°)\s?[FC]?$",  # Temperatures e.g., "350 degrees F"
    r"medium heat",
    r"high heat",
    r"low heat",  # Specific heat levels often not entities
    r"room temperature",
    r"^(about|approx\.?|approximately|around|over|under|less than|more than|at least)\s+\d+",
    r"^(a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:cup|cups|tablespoon|tablespoons|tbsp\.?|teaspoon|teaspoons|tsp\.?|pinch|dash|clove|slice|slices|piece|pieces|can|cans|jar|jars|package|packages|box|boxes|stick|sticks)$",
]

DERIVED_ENTITY_RULES = {
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
}

# Entity Role Assignment Verbs (for ProceduralText)
USE_VERBS_ROLE_ASSIGNMENT = {
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
DEFINE_VERBS_ROLE_ASSIGNMENT = {
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
# Define verb categories
CONSUME_VERBS_ROLE_ASSIGNMENT = {
    "eat",
    "consume",
    "finish",
    "remove",
    "discard",
    "drain",
    "strain",
    "reserve",
}
TRANSFORMATION_VERBS_ROLE_ASSIGNMENT = {
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
    "preheat",
}
COOKING_VERBS_ROLE_ASSIGNMENT = {
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

VALID_DIRECT_DEPENDENCIES_FOR_ENTITY_VERB = {
    "dobj",
    "obj",
    "attr",
    "nsubj",
    "nsubjpass",
    "conj",
    "appos",
    "xcomp",
    "agent",
}


# Question Generation Constants (for QuestionGenerator)
# These verb lists are specific to the concurrency question filtering logic
QGEN_BACKGROUND_PREP_VERBS = {
    "preheat",
    "heat",
    "warm",
    "cool",
    "chill",
    "freeze",
    "grease",
    "oil",
    "butter",
    "flour",
    "line",
    "prepare",
    "boil",
    "melt",
    "toast",
    "set",
    "start",
    "begin",
    "measure",
    "weigh",
    "sift",
    "drain",
    "rinse",
    "soak",
    "reserve",
    "get",
    "gather",
    "have",
    "ensure",
    "check",
    "adjust",
    "allow",
    "let",
    "place",
    "put",
    "remove",
    "take",
    "discard",
    "cover",
    "combine",
}

QGEN_ACTIVE_PREP_VERBS = {
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
    "season",
    "sprinkle",
    "pour",
    "layer",
    "spread",
    "make",
    "create",
    "cook",
    "continue",
    "work",
    "use",
    "follow",
    "repeat",
    "taste",
    "look",
    "feel",
    "smell",
    "watch",
    "dissolve",
    "cream",
    "knead",
    "roll",
    "cut",
    "shape",
    "form",
}

QGEN_UNSAFE_ENTITY_NAMES_TAINT = {"egg", "eggs", "raw chicken", "raw beef", "raw pork"}


# Analysis Types & Script Configuration
ANALYSIS_TYPES = [  # Used for consistent naming and iteration
    "Reaching Definitions",
    "Very Busy Expressions",
    "Available Expressions",
    "Live Variable Analysis",
    "Interval Analysis",
    "Type-State Analysis",
    "Taint Analysis",
    "Concurrency Analysis",
]


# For generate_final_dataset.py

# Define regex patterns to extract the primary entity phrase for each relevant type
# The patterns capture the group assumed to be the (potentially malformed) entity name
FINAL_DATASET_QUESTION_TYPE_REGEX_PATTERNS = {
    "Reaching Definitions": r"In Step \d+, is the (.*?) from Step \d+ being used\?",
    "Very Busy Expressions": r"Is (.*?) from Step \d+ used in multiple future steps.*?",
    "Available Expressions": r"Is (.*?) from Step \d+ still available in Step \d+\?",
    "Live Variable Analysis": r"Is (.*?) live after Step \d+\?",
    "Type-State Analysis": r"If we skip Step \d+, is it still valid to .+ the (.*?) in Step \d+\?",
    "Taint Analysis": r"Does using (.*?) in Step \d+ introduce.*?",
}
# Files to ignore when scanning for recipe JSONs
NON_RECIPE_JSON_FILENAMES = {
    "all_entities.json",
    "all_questions.json",
    "question_counts.json",
    "processing_stats.json",
    "proposed_entity_subset.txt",
    "subset_finding.log",
    "final_verification_report.json",
    "final_entity_subset.txt",
    "processing.log",
    "csv_generation.log",
    "dataset_statistics.json",
}

# For post_processing.py
POST_PROCESSING_STAT_TITLES = {
    "total_procedures": "Total Procedures",
    "avg_steps_per_procedure": "Avg. Steps per Procedure",
    "min_steps": "Min Steps",
    "max_steps": "Max Steps",
    "total_qa_pairs_generated": "Total QA Pairs Generated",
    "qa_pairs_used": "QA Pairs Used (in final dataset)",
}
BOOLEAN_TO_YES_NO_MAP = {True: "Yes", False: "No"}


# General Application Constants
IGNORE_TYPES = {"Interval Analysis", "Concurrency Analysis"}

NLP_MODEL_SM = "en_core_web_sm"
NLP_MODEL_TRF = "en_core_web_trf"  # Default used in procedural_text_process.py

DEFAULT_LOG_LEVEL = "INFO"
MIN_RECIPE_STEPS_FOR_PROCESSING = 3

# Plotting constants
PLOT_TOP_N_ENTITIES_MAIN = 20
PLOT_TOP_N_ENTITIES_EXTENDED = 50
FILENAME_ENTITY_FREQ_PLOT = "entity_frequency.png"
FILENAME_ALL_ENTITIES_JSON = "all_entities.json"
FILENAME_ALL_QUESTIONS_JSON = "all_questions.json"
FILENAME_QUESTION_COUNTS_JSON = "question_counts.json"
FILENAME_PROCESSING_STATS_JSON = "processing_stats.json"
FILENAME_PROCESSING_LOG = "processing.log"

# diminish.py output filenames
FILENAME_PROPOSED_SUBSET_TXT = "proposed_entity_subset.txt"
FILENAME_SUBSET_FINDING_LOG = "subset_finding.log"
FILENAME_FINAL_VERIFICATION_REPORT_JSON = "final_verification_report.json"
FILENAME_FINAL_ENTITY_SUBSET_TXT = "final_entity_subset.txt"

# generate_final_dataset.py output filenames
FILENAME_CSV_GENERATION_LOG = "csv_generation.log"

# post_processing.py output filenames
FILENAME_DATASET_STATS_JSON = "dataset_statistics.json"

RANDOM_SEED = 13
