"""Utility for parsing temporal intent from natural language queries."""

import datetime
import os
import re
from typing import Optional

from google import genai
import pydantic

from ns_agent_adk.config import config as config_module


class TimeRange(pydantic.BaseModel):
  """Represents a time range extracted from a user query."""

  start: Optional[datetime.datetime] = None
  end: Optional[datetime.datetime] = None
  query_type: Optional[str] = (
      None  # e.g., 'recent', 'exact_date', 'range', 'since', 'before'
  )
  raw_query_span: Optional[str] = None  # The part of the query indicating time
  anchor_event: Optional[str] = (
      None  # Event name for relative queries (e.g., 'surgery')
  )


def get_temporal_intent(query: str, config: config_module.Config, anchor_date: datetime.datetime | None = None) -> TimeRange:
  """Extracts the time intent from the user's query using an LLM call."""
  # Fast heuristic check to bypass LLM if no temporal markers are found.
  # Matches years (19XX, 20XX, 21XX), dates (MM/DD/YYYY, YYYY-MM-DD, Month DDth),
  # and common temporal keywords.
  temporal_pattern = re.compile(
      r'\b(19\d{2}|20\d{2}|21\d{2})\b|'  # Years 1900-2199
      r'\b(?:1[0-2]|0?[1-9])[-/](?:3[01]|[12][0-9]|0?[1-9])[-/](?:19|20|21)\d{2}\b|'  # MM/DD/YYYY
      r'\b(?:19|20|21)\d{2}[-/](?:1[0-2]|0?[1-9])[-/](?:3[01]|[12][0-9]|0?[1-9])\b|'  # YYYY-MM-DD
      r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*'
      r' \d{1,2}(?:st|nd|rd|th)?\b|'  # Jan 1st
      r'\b(since|before|after|recent|last|ago|past|from|to|between|onwards)\b',  # Keywords
      re.IGNORECASE,
  )

  if not temporal_pattern.search(query):
    return TimeRange()
  today = anchor_date if anchor_date else datetime.datetime.today()
  preamble = f"""
Extract the time intent from the user's query. Today's date is {today.strftime('%Y-%B-%d')}.
Identify the start date and end date implied by the query in YYYY-MM-DD format.
If the query implies a duration (e.g., 'last two years'), calculate the start and end dates.
If only a start date is implied (e.g., 'since 2022'), leave the end date as null.
If only an end date is implied (e.g., 'before the surgery'), leave the start date as null.
If the query is relative to an event (e.g., 'after the surgery'), extract the event name as the 'anchor_event'.
If no time intent is present, return None for all fields.
Classify the type of temporal query: 'recent', 'exact_date', 'range', 'since', 'before', 'relative', or 'none'.
Extract the text span from the query that indicates the time.

Examples:
Query: 'Creatinine levels since the surgery in 2022.' -> TimeRange(start='2022-01-01 00:00:00', end=None, query_type='since', raw_query_span='since the surgery in 2022', anchor_event='surgery')
Query: 'Medications in the last two years.' -> TimeRange(start='{(today - datetime.timedelta(days=730)).strftime('%Y-%m-%d 00:00:00')}', end='{today.strftime('%Y-%m-%d 23:59:59')}', query_type='range', raw_query_span='in the last two years')
Query: 'Show me recent health records.' -> TimeRange(start='{(today - datetime.timedelta(days=90)).strftime('%Y-%m-%d 00:00:00')}', end='{today.strftime('%Y-%m-%d 23:59:59')}', query_type='recent', raw_query_span='recent')
Query: 'What happened on Jan 15th, 2023?' -> TimeRange(start='2023-01-15 00:00:00', end='2023-01-15 23:59:59', query_type='exact_date', raw_query_span='on Jan 15th, 2023')
Query: 'How is the patient doing after the transplant?' -> TimeRange(start=None, end=None, query_type='relative', raw_query_span='after the transplant', anchor_event='transplant')
Query: 'Any allergies?' -> TimeRange(start=None, end=None, query_type='none', raw_query_span=None)

User Query: {query}
"""
  api_key = os.getenv('GENAI_API_KEY')
  if not api_key:
    raise ValueError('GENAI_API_KEY environment variable not set.')

  try:
    client = genai.Client(api_key=api_key.split(',')[0])
    response = client.models.generate_content(
        model=config.temporal_parser_model_name,
        contents=preamble,
        config=genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=TimeRange,
        ),
    )
    return response.parsed
  except Exception as e:  # pylint: disable=broad-except
    print(f'Error during temporal intent extraction: {e}')
    return TimeRange()  # Return default instance on error
