"""Prompts for the Data Science Agent.

This module contains all prompt templates used by the Data Science Agent
for approach generation, code generation, debugging, and result interpretation.
"""

from typing import Any, Dict, Final, List
import jinja2


# =============================================================================
# Prompt rendering utility
# =============================================================================

def render_prompt(template: str, **kwargs) -> str:
    """Render a Jinja2 template with the given variables."""
    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        undefined=jinja2.StrictUndefined,
    )
    jinja_template = env.from_string(template)
    return jinja_template.render(**kwargs)


# =============================================================================
# Rubric for approach evaluation
# =============================================================================

RUBRIC_QUESTIONS: Dict[str, Dict[str, str]] = {
    "data_answerable": {
        "answerable": (
            "Sometimes, the query will not be answerable given the data (e.g.,"
            " comparing the user's weight to the population but there is no"
            " weight related column in the population table). Assuming there is"
            " no missing data and we have the data for each table in the data"
            " schema for the past year, is this question answerable given the"
            " data?"
        ),
    },
    "timeframe": {
        "identify": (
            "Does the approach identify a *specific* timeframe for the question"
            " (e.g., looking at the data in the past X months)?"
        ),
        "identify_reasonableness": (
            "**If yes**, is the timeframe sufficiently reasonable?"
        ),
        "identify_clear": (
            "**If yes**, is the timeframe sufficiently clear (e.g., the"
            " timeframe is unambiguous.)? :red[**Do not make any extra"
            " assumptions!**]"
        ),
        "improvement": "Would a timeframe improve the answer a user gets?",
    },
    "data_transformations": {
        "appropriate_columns": (
            "Does the approach reference appropriate columns (if not explicitly"
            " referencing a column, it should be unambiguous what column and"
            " data table we are referring to) that exist for the approach?"
        ),
        "hardcoded_values": (
            "Besides for the timeframe, does the approach use any hardcoded"
            " values as a threshold or filter etc. (e.g., a heart rate value to"
            " reference tempo runs)?"
        ),
        "hardcoded_grounded": (
            "**If yes**, is the hardcoded value grounded based on the user's"
            " query or some accepted standard?"
        ),
        "made_transformations": (
            "Did the approach make any data transformations (separate from"
            " identifying the timeframe and calculating data statistics)?"
        ),
        "made_transformations_necessary": (
            "**If no**, are transformations necessary for the approach to"
            " answer the question?"
        ),
        "made_transformations_clear": (
            "**If yes**, is the approach sufficiently clear in how to do it"
            " (different data analysts can unambiguously get the same result"
            " from the calculation.)? :red[**Do not make any extra"
            " assumptions!**]"
        ),
        "made_transformations_logical_errors": (
            "**If yes**, did the transformations involve any logical errors"
            " (e.g., the query refers to how my steps are based on my sleep"
            " from the night before but the approach uses the sleep data on the"
            " day the steps are taken)?"
        ),
        "made_transformations_reference_nonexistant_columns": (
            "**If yes**, did the transformations reference columns that do not"
            " exist?"
        ),
    },
    "data_distribution_availability": {
        "columns_data_availability": (
            "If the approach calculates statistics or performs statistical"
            " tests, with respect to the **availability of data points** (rows)"
            " for the transformations and calculations, does the approach"
            " identify whether there is/is not sufficient data?"
        ),
        "columns_data_availability_reasonable": (
            "**If yes**, assuming we have data for all the columns needed for"
            " calculations, descriptive statistics and statistical tests, is"
            " the sufficiency check reasonable (is it too strict, not strict"
            " enough)?"
        ),
        "columns_specific_data_availability": (
            "If the approach calculates statistics or performs statistical"
            " tests, with respect to the **specific columns** necessary for"
            " calculations, descriptive statistics and statistical tests, does"
            " the approach identify whether there is/is not sufficient data"
            " (cells in those columns) ?"
        ),
        "columns_specific_data_availability_clear": (
            "**If yes**, is the sufficiency check clear (e.g., we know exactly"
            " what columns to look at or how to count/calculate whether there"
            " is/is not sufficient data)?"
        ),
        "columns_specific_data_availability_reasonable": (
            "**If yes**, is the sufficiency check reasonable (is it too strict,"
            " not strict enough)?"
        ),
        "columns_specific_data_availability_sufficient": (
            "**If yes**, is the sufficiency check comprehensive (all columns"
            " that are relevant are checked)?"
        ),
    },
    "data_statistics": {
        "summary_statistics": (
            "Does the approach calculate summary statistics (e.g., mean,"
            " median, mode, max, min, standard deviation, percentiles, count,"
            " sum, etc.)?"
        ),
        "summary_statistics_reasonable": (
            "**If yes**, is it reasonable to calculate these statistics?"
        ),
        "summary_statistics_necessary": (
            "**If no**, would summary statistics improve the data analysis"
            " approach to answer the question?"
        ),
        "summary_statistics_compared_to_hardcoded_values": (
            "**If yes**, does the approach compare the summary statistics to"
            " hardcoded values (e.g., a standard deviation of 1000 or less when"
            " specifying whether a person's steps have been consistent)?"
        ),
        "summary_statistics_compared_to_hardcoded_values_grounded": (
            "**If yes**, is the comparison to hardcoded values grounded based"
            " on the user's query or some accepted standard?"
        ),
        "summary_statistics_data_distribution": (
            "**If yes**, did the approach consider the distribution of the data"
            " when calculating the summary statistics (e.g., a median would"
            " make more sense then a mean for skewed data or a mean for"
            " categorical data would not make much sense)?"
        ),
        "summary_statistics_data_distribution_reasonable": (
            "**If yes**, is the consideration of the data distribution"
            " reasonable?"
        ),
        "summary_statistics_data_distribution_suitable": (
            "**If yes**, should the approach consider the data distribution"
            " when calculating the summary statistics?"
        ),
        "statistical_test": (
            "Does the approach perform any statistical test (e.g., T-test,"
            " ANOVA, correlation etc.)?"
        ),
        "statistical_test_reasonable": (
            "**If yes**, is it reasonable to perform this test?"
        ),
        "statistical_test_necessary": (
            "**If no**, would a statistical test improve the data analysis"
            " approach to answer the question?"
        ),
        "statistical_test_clear": (
            "**If yes**, is the approach sufficiently clear in how to do it"
            " (different data analysts can unambiguously get the same result"
            " from the calculation.)? :red[**Do not make any extra"
            " assumptions!**]"
        ),
        "statistical_test_data_distribution": (
            "**If yes**, does the approach consider the distribution of the"
            " data necessary to satisfy the assumptions of the statistical"
            " test?"
        ),
        "statistical_test_data_distribution_suitable": (
            "**If yes**, are these the right considerations for the statistical"
            " test?"
        ),
    },
    "summary": {
        "align_intents": (
            "Overall, does the approach align with the user's intent (i.e., the"
            " user will get a satisfactory answer from the analysis)?"
        ),
    },
}


def _serialize_rubric_questions(num_tabs: int = 1) -> str:
    """Serializes the approach rubric questions into a string."""
    sections = []
    indent = "\t" * num_tabs

    for section, questions in RUBRIC_QUESTIONS.items():
        if section == "data_answerable":
            continue
        section_text = f"\n{indent}**{section.replace('_', ' ').title()}**\n"
        for q_text in questions.values():
            section_text += f"\n{indent}- {q_text}"
        sections.append(section_text)

    return "\n".join(sections)


RUBRIC_QUESTIONS_STR: Final[str] = _serialize_rubric_questions()


# =============================================================================
# Few-shot examples (from few_shots/approach.py)
# =============================================================================

APPROACH_EXEMPLARS: List[Dict[str, str]] = [
    {
        "query": (
            "Am I a better cyclist than I was 6 months ago based on my app"
            " data?"
        ),
        "discussion": (
            "The term 'better' is ambiguous and requires clarification. One way"
            " to define it is by comparing average distance, speed, and heart"
            " rate over two time periods: 1 year ago to 6 months ago vs. the"
            " last 6 months. Because we are calculating summary statistics,"
            " sufficient data must be available for each period, and missing"
            " values should be checked."
        ),
        "approach": (
            "1. Define 'better' using distance per ride, speed, and heart rate."
            " 2. Filter cycling activities labeled 'OUTDOOR_BIKE' or 'BIKING'"
            " and separate them into past (1 year ago to 6 months ago) and"
            " recent (last 6 months) periods. 3. Ensure at least 5 activities"
            " per period; otherwise, return 'Insufficient data'. 4. Check for"
            " missing data and exclude metrics with >50% missing values. 5."
            " Compute summary statistics for available metrics. 6. Conduct"
            " Shapiro-Wilk test to determine normality. 7. Use t-test for"
            " normally distributed data or Mann-Whitney U test otherwise. 8."
            " Return summary statistics, missing data counts, and statistical"
            " test results, flagging any excluded metrics."
        ),
    },
    {
        "query": "How's my friend's heart rate compared to me?",
        "discussion": (
            "There is no data about my friend available. Therefore we cannot"
            " answer the query."
        ),
        "approach": (
            "1. Return 'There is no data about the user's friend available, we"
            " only have fitness data for the user and the general population.'"
        ),
    },
]


# =============================================================================
# APPROACH PROMPTS (from approach_preamble.py)
# =============================================================================

APPROACH_PREAMBLE_BASELINE: Final[str] = """\
You are an expert data scientist. Generate a step by step analysis approach based on given user query.
## Output Format
Your output should include _one_ discussion, _one_ approach field.

Begin!
Query: {{query}}
== DISCUSSION ==
"""

APPROACH_PREAMBLE_V3: Final[str] = """\
{#- Preamble: Instructions for generating the approach given the user's query -#}
You are an expert Python data analyst skilled in working with Fitbit time series health data. Your task is to devise a precise, step-by-step analysis plan based *only* on the provided user query and the available data schemas below. This plan will guide subsequent Python code generation but should **NOT** contain any code itself.

**Available Data Schemas:**
{{data_summary}}

**Task:**
Analyze the user's query below. Generate a **Discussion** section analyzing the query feasibility and operationalizing terms, followed by a step-by-step **Approach** section outlining the analysis plan.

**User Query:**
{{query}}

**Instructions for Generating the Plan:**
**A. Discussion Section:**
1.  **Analyze Feasibility:** Determine if the query can be answered using the provided data schemas.
2.  **Identify Ambiguity:** Pinpoint any vague terms in the query (e.g., "better," "active," "consistent," "often").
3.  **Operationalize Terms:** Define *exactly* how each ambiguous term will be measured using specific columns and calculations from the available data (e.g., "Define 'better cyclist' by comparing average speed (`distance`/`duration`) and average `active_zone_minutes` during cycling activities").
4.  **Outline Strategy:** Briefly summarize the overall plan.
5.  **Unanswerable Queries:** If the query cannot be answered (e.g., requires non-existent data like friend's metrics, uses undefined concepts), clearly state *why* in the discussion.

**B. Approach Section:**
* **If Query is Unanswerable:** Write a single step containing a helpful message explaining why it cannot be answered, based on the discussion (e.g., "1. Return 'Query cannot be answered because data for [missing element] is not available.'").
* **If Query is Answerable:** Provide a numbered list of precise, unambiguous steps:
    1.  **Timeframe:**
        * Specify the exact start and end dates/times for the analysis period.
        * If not specified in the query, default to the last 1 year of available data from the relevant DataFrame(s).
        * Ensure the chosen timeframe is reasonable and clearly stated.
    2.  **Data Selection & Filtering:**
        * Identify the necessary DataFrame(s) (e.g., `Summary DataFrame`, `Activities DataFrame`).
        * Specify the exact column(s) required from each DataFrame.
        * Detail any filtering criteria based on the operationalized terms or timeframe (e.g., "Filter `Activities DataFrame` for `activity_name` in ['OUTDOOR_BIKE', 'BIKING']", "Filter `Summary DataFrame` for `datetime` within the specified timeframe").
    3.  **Data Transformations:**
        * Clearly describe any required calculations or data manipulations (e.g., "Calculate speed as `distance` / (`duration` / 60) for each activity", "Aggregate daily `steps` from `Summary DataFrame` into weekly averages", "Lag `sleep_minutes` by one day to align with next day's `steps`").
        * Ensure transformations are logical and unambiguous. Specify units where relevant.
    4.  **Data Sufficiency Checks:**
        * Define and justify minimum data requirements (e.g., "Require at least 5 activities matching the criteria within each comparison period for analysis"). State this check as a step.
        * Specify checks for missing values in critical columns needed for calculations (e.g., "Check `resting_heart_rate` for nulls. If >50% are missing in a period, exclude this metric from analysis for that period and note it"). State this check as a step.
        * Clearly state what should happen if data is insufficient (e.g., "If insufficient data, return a message indicating this.").
    5.  **Statistical Analysis:**
        * Specify which summary statistics to calculate (e.g., mean, median, count, std dev, min, max). Justify why they are appropriate (consider data distribution - e.g., "Calculate median `steps` due to potential skewness").
        * If performing statistical tests (e.g., t-test, correlation, ANOVA):
            * State the exact test to be used and the variables involved.
            * Justify the choice of test.
            * Mention necessary assumptions (e.g., normality for t-test) and how they will be assessed (e.g., "Perform Shapiro-Wilk test on metric distributions within each group").
            * Ensure the test setup is unambiguous.
    6.  **Define Output:**
        * Clearly state what specific data, statistics, or test results the execution of this plan should produce (e.g., "Return the calculated median steps for each month", "Return the p-value from the t-test comparing average heart rates"). **Do not include interpretation or comparisons in the plan itself.**

**Key Requirements & Constraints:**

* **Precision:** Steps must be unambiguous. Another analyst using the same data must reach the exact same numerical result by following the plan. Use specific thresholds and column names.
* **No Assumptions:** Base the plan *only* on the query and provided schemas. **Do not make extra assumptions!** If a needed threshold isn't specified or inferable, state this as a limitation or use a clearly justified standard (e.g., population norms if applicable and available).
* **Relevance & Scope:** Use only the provided data. Address *only* the user's query. The plan should *only* return the data/statistics needed to answer the query, not interpretations.
* **NO CODE:** The output must be the plan (Discussion + Approach) only.

**Output Format:**
Your output must contain exactly one `== Discussion ==` section and one `== Approach ==` section, formatted as shown in the examples above.

Begin!
**Query:** {{query}}
== Discussion ==
"""


# =============================================================================
# CODEGEN PROMPTS
# =============================================================================

# DataFrame schemas embedded in prompt
DATAFRAME_SCHEMAS: Final[str] = """
# summary_df: Summary DataFrame
This is a summary of the user's activity, sleep, and personal health records data. Each row in the summary dataframe represents a single day.
The dataframe is indexed by the date of the record (datetime64[ns]), the index name is `datetime`.
The dataframe has the following columns:
- steps: The number of steps taken each day (float64).
- sleep_minutes: The total sleep time (in minutes) for each day (float64).
- bed_time: The time the user went to bed (datetime64[ns]).
- wake_up_time: The time the user woke up (datetime64[ns]).
- resting_heart_rate: The average resting heart rate (in beats per minute) for each day (float64).
- heart_rate_variability: The user's heart rate variability (float64).
- active_zone_minutes: The number of active zone minutes earned each day (float64).
- deep_sleep_minutes: The amount of deep sleep (in minutes) for each day (float64).
- rem_sleep_minutes: The amount of REM sleep (in minutes) for each day (float64).
- light_sleep_minutes: The amount of light sleep (in minutes) for each day (float64).
- awake_minutes: The amount of awake time (in minutes) for each day (float64).
- stress_management_score: The user's stress management score for each day. Higher value means better stress management and lower stress level (float64).
- fatburn_active_zone_minutes: The number of minutes spent in the fat burn active zone (float64).
- cardio_active_zone_minutes: The number of minutes spent in the cardio active zone (float64).
- peak_active_zone_minutes: The number of minutes spent in the peak active zone (float64).

# activities_df: Activities DataFrame
Each row in the activities dataframe represents a single activity.
The dataframe is indexed by the start time of the activity (datetime64[ns]), the index name is `start_time`.
The dataframe has the following columns:
- start_time: The start time of the activity (datetime64[ns]).
- end_time: The end time of the activity (datetime64[ns]).
- activity_name: The name of the activity (object). Possible values: "WALKING", "RUNNING", "OUTDOOR_BIKE", "STRENGTH_TRAINING", "WORKOUT", "ELLIPTICAL", "YOGA", "TREADMILL", "HIKING", "SWIMMING".
- distance: The distance covered during the activity (float64) in miles.
- duration: The duration of the activity (float64) in minutes.
- calories: The number of calories burned during the activity (float64).
- steps: The number of steps taken during the activity (float64).
- active_zone_minutes: The number of active zone minutes earned during the activity (float64).

# profile_df: User Profile Dataframe
The profile dataframe is a single row dataframe that contains the following information:
- age: The age of the user as an integer (e.g., 34).
- gender: The gender of the user as a lowercase string ("male" or "female").
- height_cm: The height of the user in centimeters (int).
- weight_kg: The weight of the user in kilograms (int).
- averageDailySteps: The user's average daily step count (int).
- cluster: The user's activity profile cluster label (object).
- elderly: Whether the user is classified as elderly (bool).

# population_df: Population DataFrame
This DataFrame represents simulated population health and activity metrics, broken down by demographic groups and percentile ranks.
Note: To compare a user's profile age (integer) with population data, call the pre-loaded `age_to_age_group(age)` helper function (already available in the execution environment — do NOT redefine it). It converts an integer age to the appropriate age bracket string (e.g., 34 → "25-34").
The dataframe has the following columns:
- percentile: An integer value ranging from 1 to 100.
- age: A string representing the age bracket (e.g., "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Under 18").
- gender: A string indicating the gender ("male" or "female").
- resting_heart_rate: Resting heart rate (beats per minute) for the percentile (float64).
- heart_rate_variability: Heart rate variability for the percentile (float64).
- steps: Number of steps for the percentile (float64).
- fatburn_active_zone_minutes: Fat burn active zone minutes for the percentile (float64).
- cardio_active_zone_minutes: Cardio active zone minutes for the percentile (float64).
- peak_active_zone_minutes: Peak active zone minutes for the percentile (float64).
- rem_sleep_minutes: REM sleep minutes for the percentile (float64).
- deep_sleep_minutes: Deep sleep minutes for the percentile (float64).
- light_sleep_minutes: Light sleep minutes for the percentile (float64).
- stress_management_score: Stress management score for the percentile (float64).
"""

# V2 codegen prompt
CODEGEN_PREAMBLE_V2: Final[str] = """
# Role and Goal
You are an expert Python data scientist specializing in analyzing Fitbit health and fitness time series data using the pandas library. Your goal is to write the **functional body** of a Python script that performs a specific analysis task.

# Inputs Provided
You will receive:
1.  **DataFrame Schemas:** Detailed descriptions of the pandas DataFrames available for use. Pay close attention to column names, data types, and the index. Crucially, `datetime` information is the **index** for these DataFrames, not a standard column.
2.  **Approach:** A step-by-step description outlining the specific analysis required. It describes the objective, potentially mentions the input DataFrames (by conceptual name or description), and the expected output or result of the analysis (answering the user query within).

<Dataframe schemas>
""" + DATAFRAME_SCHEMAS + """
</Dataframe schemas>

<Approach>
{{approach}}
</Approach>

# Instructions for Generating Python Code
Based strictly on the provided **schemas** and **approach**, generate only the Python code block that performs the requested analysis:

1.  **Core Logic:** Implement the data manipulation and analysis steps necessary to fulfill the requirements described in the `<Approach>`.
2.  **Assume Inputs:** Assume the DataFrames described in `<Dataframe schemas>` are available as pandas DataFrame variables (`summary_df`, `activities_df`, `profile_df`, `population_df`).
3.  **Use Index:** Remember to leverage the `datetime` index for time-based operations, filtering, or aggregations. Do not treat `datetime` as a column unless explicitly stated otherwise in the schema.
4.  **Imports:** Start the code block with all necessary import statements. Include the standard ones provided below, and add any others required by your generated analysis code (e.g., `numpy`).
5.  **Output:** The code should result in the final requested output (e.g., a processed DataFrame, a calculated value, a list of results). This might involve assigning the final result to a variable (e.g., `result_df` or `final_value`) as the last step, simulating what a function might return.
6.  **Function:** The code should be wrapped in a function named `analysis` that takes the DataFrames as arguments.
7.  **No Extras:**
    * Do **NOT** include the docstring itself in the output.
    * Do **NOT** include any comments (`# ...`) in the generated code.
8.  **Code Quality:** Write clean, efficient, and idiomatic pandas code. Use vectorized operations where appropriate.
9.  **Pre-loaded Helpers:** The execution environment provides `age_to_age_group(age: int) -> str` which converts an integer age to an age bracket string (e.g., 34 → "25-34"). Do NOT redefine this function — just call it directly.
10. **Conciseness:** Keep the analysis function body under 80 lines. Avoid verbose error handling or redundant data checks. If population data for a demographic is missing, return the user's own metrics without population comparison rather than raising an error.

Now generate the code.

```python
from typing import Any, Dict
import pandas as pd
import numpy as np
import datetime

def analysis(summary_df: pd.DataFrame | None = None,
    activities_df: pd.DataFrame | None = None,
    profile_df: pd.DataFrame | None = None,
    population_df: pd.DataFrame | None = None,) -> Dict[str, Any]:
"""

# V1 codegen prompt (not used by default)
CODEGEN_PREAMBLE: Final[str] = """\
{#- Preamble: Instructions for generating the code given the user's query and approach -#}
You are an expert python data programmer skilled in working with Fitbit time series health data and writing code given a specific analysis approach.

Here's a summary of the available data relevant to the user:
{{data_summary}}

You will be provided with an open-ended query related to the user's Fitbit data and an analysis approach to answer the query. Your task is to write the python code to implement the analysis approach.
Please keep the intent of the approach in mind when writing the code. Return all necessary calculations and contextual data to answer the query.

Important:
- In addition to built-in imports, please only use/import the following libraries:
    - matplotlib
    - pandas
    - numpy
    - scipy
- Anything that is printed to the screen will not be observed.
- ONLY the data structure stored in the last line of code will be used to \
answer the query. Store the answer and all relevant contextual data in a variable called result.


Begin!
Query: {{query}}
Approach: {{approach}}
The dataframes are stored in the following variables (DO NOT create new toy dataframes):
{{dfs_info}}
Code:
```python
"""


CODEGEN_PREAMBLE_BASELINE: Final[str] = """\
You are a Python expert with tons of experience with data anlysis on Fitbit health and fitness time series data.
You are provided with dataframe schemas that describes Dataframes available to use, and a docstring that describe how an `analysis` function should work to generate code that answers the user query listed in the docstring.
Here's a summary of the available data relevant to the user:
<Dataframe schemas>
""" + DATAFRAME_SCHEMAS + """
</Dataframe schemas>



Query: {{query}}
Approach: {{approach}}

Return only the fixed code (enclosed in ```python ... ``` block), without any explanation.

Generated Code:
```python
"""

DEBUG_CODEGEN_PREAMBLE_BASELINE: Final[str] = """\
{#- Preamble: Instructions for debugging code -#}
Please dubug and return the fixed code for the following code instructions, code, and error.

Code Instructions:
{{code_instructions}}
Code:
{{code}}
Error:
{{error}}

Return only the fixed code (enclosed in ```python ... ``` block), without any explanation.

Fixed Code:
```python
"""

CODE_REFINEMENT_PREAMBLE: Final[str] = """
You are an expert Python data programmer specializing in analyzing Fitbit time series health data.

**Task:** The code below executed successfully. Refine it based on the feedback provided. Preserve all correct logic — only change what the feedback explicitly requests.

**Guiding Principles:**
* Preserve all correct logic from the original code.
* Make only the changes described in the refinement feedback.
* Do not introduce unrelated changes, new analyses, or extra functionality.

**Structural Requirements (must be preserved exactly):**
* Wrap all code in the `analysis` function with the exact signature shown at the end of this prompt.
* No comments (`# ...`) anywhere in the code.
* No `try/except` wrappers unless the original had them.
* Keep the function body under 80 lines.
* Do not redefine `age_to_age_group` — it is pre-loaded in the execution environment.

**Input:**
Here's a summary of the available data:
<Dataframe schemas>
""" + DATAFRAME_SCHEMAS + """
</Dataframe schemas>

**1. Analysis goal:**
{{code_instructions}}

**2. Current code (executed successfully):**
{{code}}

**3. Execution output:**
{{execution_results}}

**4. Refinement feedback:**
{{code_update_feedback}}

Return only the refined code (enclosed in ```python ... ``` block), without any explanation.

Refined Python code:
```python
"""


DEBUG_CODEGEN_PREAMBLE: Final[str] = """\
{#- Preamble: Instructions for generating the code given the user's query and approach -#}
You are an expert python data programmer skilled in working with Fitbit time series health data and writing and debugging code given a specific analysis approach.
Given the user query, analysis approach, code, and a traceback error message from executing the code, your task is to edit the code to fix the error.

Here's a summary of the available data relevant to the user:
{{data_summary}}

Please keep the intent of the original code in mind when writing the code.
Return the entire code with the error fixed.

Important:
- The necessary data to answer the query will be stored on the last line of code and should not be the last line of a function.
- In addition to built-in imports, please only use/import the following libraries:
    - matplotlib
    - pandas
    - numpy
    - scipy
- Anything that is printed to the screen will not be observed.
- ONLY the data structure stored in the last line of code will be used to answer the query. Store this in a variable called result.
- Keep the code concise — focus ONLY on fixing the specific error. Do NOT add extra analyses, visualizations, or functionality that was not in the original code.

Begin!
Code Instructions: 
{{code_instructions}}
Code: 
{{code}}
Error: 
{{error}}
Fixed Code:
```python
"""


DEBUG_CODEGEN_PREAMBLE_V2: Final[str] = """
You are an expert Python data programmer specializing in analyzing and debugging Fitbit time series health data according to specific analysis requirements.

**Task:** Review the provided code generation prompt, the generated Python code, and the results of executing that code (which may include errors, tracebacks, or unexpected output). Identify the error(s) or discrepancies, fix the code accordingly, and return *only* the complete, corrected Python code block.

**Guiding Principles:**
* The fixed code must accurately implement the analysis described in the 'Code generation prompt'.
* The fix should directly address the error(s) or issues indicated in the 'Code execution results'.
* Preserve the original intent and overall logic described in the 'Code generation prompt'.
* Maintain the original coding style and structure where possible, unless changes are necessary for the fix.
* Do not introduce significant unrelated logic changes.
* **IMPORTANT: Focus ONLY on fixing the specific error. Do NOT add extra analyses, visualizations, try/except wrappers, or functionality that was not in the original code. If the original code was already long, simplify where possible while preserving correctness. The goal is a minimal, targeted fix.**

**Input:**

**1. Code generation prompt (Original Goal):**
{{code_instructions}}

**2. Generated code (Contains Errors/Issues):**
{{code}}
**3. Code execution results (Error, Traceback, or Output):**
{{code_result}}

**4. Code Update Feedback (Optional):**
Code Update Feedback:
{{code_update_feedback}}
**Output:**
Fixed Python code:
```python

"""


# =============================================================================
# INTERPRETATION PROMPT (from interpretation_preamble.py)
# =============================================================================

INTERPRETATION_PREAMBLE: Final[str] = """\
{#- Preamble: Instructions for interpreting the result of the data generation -#}

You are an expert data analyst specializing in Fitbit time-series health data.\
Your goal is to analyze execution results, identify issues, and communicate \
findings in a clear and user-friendly manner.

You will be provided a user query, an analysis approach, code, and the \
execution results from executing the code.

Your response should follow these guidelines:
- If the execution results are valid (i.e., the analysis is correct and no code \
changes are needed), summarize the values within the context of the original query. \
The user won't be able to see the execution results, so you will need to \
summarize them in a way that is understandable to the user. \
Do not make any interpretations for the user. YOU are not a health expert.
    - Start your response with a "[SUMMARY]" tag.
- If the execution results appear to be invalid due to a logical error in the \
code (e.g., there are nan values when there should not be any), explain why the \
result is not valid and suggest an update to the code (in english).
    - Start your response with a "[CODE UPDATE]" tag.
    
Begin!
Query: {{query}}
Approach:
{{approach}}
Code: 
{{code}}
Execution Results:
{{execution_results}}
Response:
"""


# =============================================================================
# Prompt generation functions (simplified from build_prompt.py)
# =============================================================================

def generate_approach_prompt_baseline(
    query: str,
) -> str:
    """Create the prompt for generating the approach (baseline / eval).

    Used by the codegen eval harness when the harness wants a minimal,
    schema-free approach generation step.
    """
    return render_prompt(
        APPROACH_PREAMBLE_BASELINE,
        query=query,
    )


def generate_approach_prompt(
    query: str,
    dfs_str: str,
) -> str:
    """Create the prompt for generating the approach."""
    return render_prompt(
        APPROACH_PREAMBLE_V3,
        query=query,
        data_summary=dfs_str,
        rubric=RUBRIC_QUESTIONS_STR,
    )


def generate_codegen_prompt(
    query: str,
    approach_str: str,
    dfs_str: str,
    dfs_info_str: str,
) -> str:
    """Create the prompt for generating the code.

    Uses V2 prompt which requires def analysis() function wrapper.
    The V2 prompt has DataFrame schemas embedded, so we only need approach.
    """
    return render_prompt(
        CODEGEN_PREAMBLE_V2,
        approach=approach_str,
    )


def generate_codegen_prompt_baseline(
    query: str,
    approach_str: str,
) -> str:
    """Create the prompt for generating the code (baseline / eval).

    Used when *approach_str* is a Python function stub (docstring_prompt
    format) — the baseline preamble treats the stub as the authoritative
    spec rather than re-imposing V2's return-type/signature constraints.
    """
    return render_prompt(
        CODEGEN_PREAMBLE_BASELINE,
        query=query,
        approach=approach_str,
    )


def generate_debug_prompt(
    approach_str: str,
    code: str,
    error: str,
    dfs_str: str,
) -> str:
    """Create the prompt for debugging the code."""
    return render_prompt(
        DEBUG_CODEGEN_PREAMBLE,
        code_instructions=approach_str,
        data_summary=dfs_str,
        code=code,
        error=error,
    )


def generate_debug_prompt_baseline(
    approach_str: str,
    code: str,
    error: str,
    dfs_str: str,
) -> str:
    """Create the prompt for debugging the code (baseline / eval).

    Neutral debug prompt that shows the stub, the code, and the error
    without imposing a competing function signature.
    """
    return render_prompt(
        DEBUG_CODEGEN_PREAMBLE_BASELINE,
        code_instructions=approach_str,
        data_summary=dfs_str,
        code=code,
        error=error,
    )


def generate_debug_prompt_v2(
    approach_str: str,
    code: str,
    code_result: str,
    code_update_feedback: str = "",
) -> str:
    """Create the prompt for debugging the code (v2)."""
    return render_prompt(
        DEBUG_CODEGEN_PREAMBLE_V2,
        code_instructions=approach_str,
        code=code,
        code_result=code_result,
        code_update_feedback=code_update_feedback,
    )


def generate_code_refinement_prompt(
    approach_str: str,
    code: str,
    execution_results: str,
    code_update_feedback: str,
) -> str:
    """Create the prompt for refining successfully-executing code.

    Used when code ran without error but the interpreter identified a logical
    issue. Unlike generate_debug_prompt_v2, this prompt correctly frames the
    task as refinement rather than error-fixing.
    """
    return render_prompt(
        CODE_REFINEMENT_PREAMBLE,
        code_instructions=approach_str,
        code=code,
        execution_results=execution_results,
        code_update_feedback=code_update_feedback,
    )


def generate_result_communication_prompt(
    query: str,
    approach_str: str,
    code: str,
    execution_results: str,
) -> str:
    """Create the prompt for generating the result communication."""
    return render_prompt(
        INTERPRETATION_PREAMBLE,
        query=query,
        approach=approach_str,
        code=code,
        execution_results=execution_results,
    )
