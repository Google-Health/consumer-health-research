"""Utility modules for PHA."""

from .data_schemas import (
    DType,
    ColumnInfo,
    DataFrameInfo,
    SUMMARY_SCHEMA,
    ACTIVITIES_SCHEMA,
    POPULATION_SCHEMA,
    PROFILE_SCHEMA,
    DFS_INFO,
    PhdaDataFrame,
    enforce_persona_schema,
)
from .data_loader import load_persona, localize_to_date
from .serialization import (
    DataSerializer,
    DataSerializerBase,
    serialize_dataframes,
    round_nested_floats,
)
from .parsing import parse_code_output
from .model_discovery import (
    ModelInfo,
    get_available_models,
    get_gemini_models,
    get_openai_models,
    get_anthropic_models,
    list_all_models,
    get_default_model,
    get_recommended_models,
    KNOWN_GEMINI_MODELS,
    KNOWN_OPENAI_MODELS,
    KNOWN_ANTHROPIC_MODELS,
)
from .personas import (
    PersonaInfo,
    discover_personas,
    list_personas,
    get_persona_ids,
    get_default_persona,
    load_persona_data,
    validate_persona_directory,
    create_persona_template,
    copy_persona,
)

__all__ = [
    # Data schemas
    "DType",
    "ColumnInfo", 
    "DataFrameInfo",
    "SUMMARY_SCHEMA",
    "ACTIVITIES_SCHEMA",
    "POPULATION_SCHEMA",
    "PROFILE_SCHEMA",
    "DFS_INFO",
    "PhdaDataFrame",
    "enforce_persona_schema",
    # Data loading
    "load_persona",
    "localize_to_date",
    # Serialization
    "DataSerializer",
    "DataSerializerBase",
    "serialize_dataframes",
    "round_nested_floats",
    # Parsing
    "parse_code_output",
    # Model discovery
    "ModelInfo",
    "get_available_models",
    "get_gemini_models",
    "get_openai_models",
    "get_anthropic_models",
    "list_all_models",
    "get_default_model",
    "get_recommended_models",
    "KNOWN_GEMINI_MODELS",
    "KNOWN_OPENAI_MODELS",
    "KNOWN_ANTHROPIC_MODELS",
    # Personas
    "PersonaInfo",
    "discover_personas",
    "list_personas",
    "get_persona_ids",
    "get_default_persona",
    "load_persona_data",
    "validate_persona_directory",
    "create_persona_template",
    "copy_persona",
]
