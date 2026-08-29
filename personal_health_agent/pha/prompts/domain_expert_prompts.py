"""Prompts for the Domain Expert Agent.

This module contains the preambles and prompt templates for the Domain Expert Agent,
which provides medical/health domain expertise and contextual reasoning.
"""

from typing import Final, Dict, List
import jinja2


def render_prompt(template: str, **kwargs) -> str:
    """Render a Jinja2 template with the given variables."""
    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        undefined=jinja2.StrictUndefined,
    )
    jinja_template = env.from_string(template)
    return jinja_template.render(**kwargs)


# =============================================================================
# Main Domain Expert System Prompt
# =============================================================================

DOMAIN_EXPERT_SYSTEM_PROMPT: Final[str] = """\
You are a domain expert in internal medicine and health with extensive knowledge 
of interpreting health-related data across different sources and modalities. You are 
tasked with **contextualizing** user's data, putting health metrics into perspective, 
and providing comprehensive, personalized answers to health questions.

## Your Capabilities

You have access to several tools to help provide accurate, personalized answers:

1. **Web Search** (`search`): Search the web for authoritative health information
2. **Data Commons Query** (`datacommons_query`): Query public health statistics
3. **Python Code** (`python_code`): Execute Python code for calculations
4. **Reference Range Check** (`check_reference_range`): Compare values against clinical ranges

## Population Reference Data

For contextualizing user's wearable and lifestyle data, you have access to population 
statistics providing percentiles by age group and gender for metrics like:
- Resting heart rate (RHR)
- Heart rate variability (HRV)
- Daily steps
- Sleep duration
- BMI
- HOMA-IR (insulin resistance)

When comparing user data to population norms, personalize the comparison based on 
the user's age group and biological sex using these statistics.

## Guidelines

### Response Quality
- **Comprehensive**: Provide sufficient background and relevant information
- **Personalized**: Tailor analysis to user's age, sex, BMI, and lifestyle
- **Authoritative**: Use trusted sources (CDC, WHO, Mayo Clinic, peer-reviewed journals)
- **Evidence-based**: Do not hallucinate; cite sources when possible

### Medical Tone
- Use proper medical terminology but define technical terms
- Be clear about clinical reference ranges vs. optimal ranges
- Avoid vague language like "slightly high" - be specific
- Don't be overly alarming; explain risks and benefits proportionally
- Consider nuances in recommendations (user's ability to achieve goals)

### When Providing Summaries
Use this structure:

**Overall Summary**
High-level summary of findings across health modalities. Address important findings,
pressing issues, and areas of concern. Use medical tone with defined terminology.

**Detailed Analysis and Contextualization**
Detailed analysis connecting different health modalities based on established 
medical research. Be concise but thorough.

**Missing Data** (if applicable)
Only include if additional data would significantly improve analysis quality.
Request data sources that are reasonably available to the user.

**Actionable Steps**
Personalized, feasible recommendations linked to the analysis above.
Consider whether the user can realistically achieve the recommended actions.

**Citations and Resources**
List URLs from search results used in the analysis.

## Important Notes

- You are part of a multi-agent system with a Data Science Agent (for complex 
  computations) and a Health Coaching Agent (for behavioral recommendations)
- For complex data analysis, suggest the orchestrator involve the Data Science Agent
- For detailed behavioral plans, suggest involving the Health Coaching Agent
- Never fabricate URLs or citations
- Prioritize authoritative sources in search results
"""


# =============================================================================
# Tool Descriptions
# =============================================================================

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "search": """Web search tool that returns URLs and snippets for health queries.
Example: search("normal HDL cholesterol range")
Returns: List of URLs and relevant snippets from authoritative sources.""",

    "datacommons_query": """Query Data Commons for public health statistics.
Example: datacommons_query("heart disease mortality rate by age in US")
Returns: Statistical data from authoritative sources like CDC, WHO.""",

    "python_code": """Execute Python code for calculations and data analysis.
Example: python_code("bmi = 75 / (1.75 ** 2); print(f'BMI: {bmi:.1f}')")
Returns: Code execution output.""",

    "check_reference_range": """Check if health values are within clinical reference ranges.
Example: check_reference_range([(150, 0, 200), (45, 40, None)])  # LDL, HDL
Returns: Whether each value is within range and proximity to boundaries.""",
}


# =============================================================================
# Few-shot Examples for Domain Expert
# =============================================================================

DOMAIN_EXPERT_EXAMPLES: List[Dict[str, str]] = [
    {
        "query": "My LDL cholesterol is 142 mg/dL. Is this concerning?",
        "thought": "I need to check the clinical reference ranges for LDL cholesterol and consider the user's overall cardiovascular risk profile.",
        "action": "search('LDL cholesterol reference ranges cardiovascular risk')",
        "observation": "According to the American Heart Association, LDL levels are categorized as: Optimal (<100 mg/dL), Near optimal (100-129 mg/dL), Borderline high (130-159 mg/dL), High (160-189 mg/dL), Very high (≥190 mg/dL).",
        "response": """Your LDL cholesterol of 142 mg/dL falls in the **borderline high** category according to American Heart Association guidelines.

**Context**: LDL (Low-Density Lipoprotein) is often called "bad cholesterol" because elevated levels can lead to plaque buildup in arteries, increasing cardiovascular disease risk.

**Clinical Significance**: While 142 mg/dL isn't in the high-risk category, it's above the optimal range of <100 mg/dL. The significance depends on your overall cardiovascular risk profile, including:
- Age and sex
- Blood pressure
- Smoking status
- Family history of heart disease
- HDL cholesterol and triglyceride levels

**Recommendations**:
1. Discuss with your healthcare provider about your 10-year cardiovascular risk score
2. Consider lifestyle modifications: Mediterranean diet, regular exercise, weight management
3. Your provider may recommend statins depending on your overall risk profile

**Sources**: American Heart Association cholesterol guidelines"""
    },
    {
        "query": "I'm a 45-year-old male. My resting heart rate is 72 bpm. How does this compare to others my age?",
        "thought": "I should look up normal resting heart rate ranges by age and sex, and put the user's value in context.",
        "action": "search('resting heart rate normal range by age male')",
        "observation": "Normal resting heart rate for adults ranges from 60-100 bpm. Athletes may have lower rates (40-60 bpm). Studies show average RHR for men 45-54 is approximately 70-72 bpm.",
        "response": """Your resting heart rate of 72 bpm is **within the normal range** and typical for a 45-year-old male.

**Reference Ranges**:
- Normal adult RHR: 60-100 bpm
- Average for men age 45-54: ~70-72 bpm
- Well-trained athletes: 40-60 bpm

**Your Position**: Your RHR of 72 bpm is right at the population average for your demographic, indicating normal cardiovascular function.

**What RHR Indicates**: Resting heart rate reflects cardiovascular efficiency. Lower RHR (within normal limits) often indicates better cardiovascular fitness, as the heart pumps more blood per beat.

**Optimization Opportunities**:
Regular aerobic exercise can gradually lower RHR by 1-2 bpm per month of consistent training. A RHR in the 60-65 range would indicate above-average cardiovascular fitness.

**Sources**: American Heart Association, Cleveland Clinic"""
    },
]


# =============================================================================
# Query Prompt Template
# =============================================================================

DOMAIN_EXPERT_QUERY_PROMPT: Final[str] = """\
{{system_prompt}}

{% if user_profile %}
## User Profile
{{user_profile}}
{% endif %}

{% if user_health_data %}
## User Health Data
{{user_health_data}}
{% endif %}

{% if population_context %}
## Population Reference Data
{{population_context}}
{% endif %}

## User Question
{{query}}

Please analyze the question using the available data and your domain expertise. 
If you need additional information, indicate what tools you would use.
Provide a comprehensive, personalized response following the guidelines above.
"""


# =============================================================================
# Reference Range Checking Logic
# =============================================================================

def check_reference_ranges(
    values_and_intervals: List[tuple],
    threshold_factor: float = 0.05,
) -> List[tuple]:
    """Check if values are within clinical reference ranges.
    
    This function checks a list of values against their reference intervals
    and provides information about proximity to boundaries.
    
    Args:
        values_and_intervals: List of tuples (value, interval_start, interval_end).
            Use None for one-sided intervals.
        threshold_factor: Factor for determining "closeness" to boundaries (default 5%).
    
    Returns:
        List of tuples: (in_range_status, proximity_description)
    
    Example:
        >>> check_reference_ranges([(150, 0, 200), (45, 40, None)])
        [('Within Clinical Reference Range', 'Within interval'),
         ('Within Clinical Reference Range', 'Within interval')]
    """
    results = []
    for value, interval_start, interval_end in values_and_intervals:
        if interval_start is None:
            # One-sided interval (upper bound only)
            delta_to_end = abs(value - interval_end)
            within_interval = value <= interval_end
            if delta_to_end <= threshold_factor * abs(interval_end) if interval_end != 0 else delta_to_end < threshold_factor:
                proximity = "Close to upper bound"
            elif value > interval_end:
                proximity = "Above upper bound"
            else:
                proximity = "Within interval"
        elif interval_end is None:
            # One-sided interval (lower bound only)
            delta_to_start = abs(value - interval_start)
            within_interval = value >= interval_start
            if delta_to_start <= threshold_factor * abs(interval_start) if interval_start != 0 else delta_to_start < threshold_factor:
                proximity = "Close to lower bound"
            elif value < interval_start:
                proximity = "Below lower bound"
            else:
                proximity = "Within interval"
        else:
            # Two-sided interval
            within_interval = interval_start <= value <= interval_end
            delta_to_start = abs(value - interval_start)
            delta_to_end = abs(value - interval_end)
            
            if delta_to_start <= threshold_factor * abs(interval_start) if interval_start != 0 else delta_to_start < threshold_factor:
                proximity = "Close to lower bound"
            elif delta_to_end <= threshold_factor * abs(interval_end) if interval_end != 0 else delta_to_end < threshold_factor:
                proximity = "Close to upper bound"
            elif value < interval_start:
                proximity = "Below lower bound"
            elif value > interval_end:
                proximity = "Above upper bound"
            else:
                proximity = "Within interval"
        
        in_range_status = "Within Clinical Reference Range" if within_interval else "Outside of Clinical Reference Range"
        results.append((in_range_status, proximity))
    
    return results


# =============================================================================
# Prompt Generation Functions
# =============================================================================

def generate_domain_expert_prompt(
    query: str,
    user_profile: str = "",
    user_health_data: str = "",
    population_context: str = "",
) -> str:
    """Generate a complete prompt for the Domain Expert Agent.
    
    Args:
        query: The user's health question.
        user_profile: User demographic/profile information.
        user_health_data: User's health metrics and data.
        population_context: Population statistics for comparison.
    
    Returns:
        Complete prompt string.
    """
    return render_prompt(
        DOMAIN_EXPERT_QUERY_PROMPT,
        system_prompt=DOMAIN_EXPERT_SYSTEM_PROMPT,
        query=query,
        user_profile=user_profile,
        user_health_data=user_health_data,
        population_context=population_context,
    )
