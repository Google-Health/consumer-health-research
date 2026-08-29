"""Prompt templates for PHA agents."""

from .data_science_prompts import (
    render_prompt,
    RUBRIC_QUESTIONS,
    RUBRIC_QUESTIONS_STR,
    APPROACH_EXEMPLARS,
    APPROACH_PREAMBLE_BASELINE,
    APPROACH_PREAMBLE_V3,
    CODEGEN_PREAMBLE,
    CODEGEN_PREAMBLE_BASELINE,
    DEBUG_CODEGEN_PREAMBLE,
    DEBUG_CODEGEN_PREAMBLE_BASELINE,
    DEBUG_CODEGEN_PREAMBLE_V2,
    CODE_REFINEMENT_PREAMBLE,
    INTERPRETATION_PREAMBLE,
    generate_approach_prompt,
    generate_approach_prompt_baseline,
    generate_codegen_prompt,
    generate_codegen_prompt_baseline,
    generate_debug_prompt,
    generate_debug_prompt_baseline,
    generate_debug_prompt_v2,
    generate_code_refinement_prompt,
    generate_result_communication_prompt,
)

from .domain_expert_prompts import (
    DOMAIN_EXPERT_SYSTEM_PROMPT,
    DOMAIN_EXPERT_EXAMPLES,
    TOOL_DESCRIPTIONS,
    generate_domain_expert_prompt,
    check_reference_ranges,
)

from .health_coach_prompts import (
    HEALTH_COACH_SYSTEM_PROMPT,
    HEALTH_COACH_SIMPLE_PROMPT,
    GOAL_PROMPT,
    REC_PROMPT,
    FINISH_PROMPT,
    SYNTHESIS_PROMPT,
)

from .orchestrator_prompts import (
    MULTI_AGENT_COLLABORATION_EXAMPLES,
    DATA_SCIENCE_AGENT_DATA_ACCESS_PROMPT,
    TEAM_STRUCTURE_PROMPT,
    AGENT_NAME_DATA_SCIENCE,
    AGENT_NAME_DOMAIN_EXPERT,
    AGENT_NAME_HEALTH_COACH,
)

__all__ = [
    # Data Science prompts
    "render_prompt",
    "RUBRIC_QUESTIONS",
    "RUBRIC_QUESTIONS_STR",
    "APPROACH_EXEMPLARS",
    "APPROACH_PREAMBLE_BASELINE",
    "APPROACH_PREAMBLE_V3",
    "CODEGEN_PREAMBLE",
    "CODEGEN_PREAMBLE_BASELINE",
    "DEBUG_CODEGEN_PREAMBLE",
    "DEBUG_CODEGEN_PREAMBLE_BASELINE",
    "DEBUG_CODEGEN_PREAMBLE_V2",
    "CODE_REFINEMENT_PREAMBLE",
    "INTERPRETATION_PREAMBLE",
    "generate_approach_prompt",
    "generate_approach_prompt_baseline",
    "generate_codegen_prompt",
    "generate_codegen_prompt_baseline",
    "generate_debug_prompt",
    "generate_debug_prompt_baseline",
    "generate_debug_prompt_v2",
    "generate_code_refinement_prompt",
    "generate_result_communication_prompt",
    # Domain Expert prompts
    "DOMAIN_EXPERT_SYSTEM_PROMPT",
    "DOMAIN_EXPERT_EXAMPLES",
    "TOOL_DESCRIPTIONS",
    "generate_domain_expert_prompt",
    "check_reference_ranges",
    # Health Coach prompts
    "HEALTH_COACH_SYSTEM_PROMPT",
    "HEALTH_COACH_SIMPLE_PROMPT",
    "GOAL_PROMPT",
    "REC_PROMPT",
    "FINISH_PROMPT",
    "SYNTHESIS_PROMPT",
    # Orchestrator prompts
    "MULTI_AGENT_COLLABORATION_EXAMPLES",
    "DATA_SCIENCE_AGENT_DATA_ACCESS_PROMPT",
    "TEAM_STRUCTURE_PROMPT",
    "AGENT_NAME_DATA_SCIENCE",
    "AGENT_NAME_DOMAIN_EXPERT",
    "AGENT_NAME_HEALTH_COACH",
]
