"""Logging utilities for the FHIR Retrieval Bench."""

import logging
import re


class LogSuppressor(logging.Filter):
  """Filters log messages based on filename and regex pattern."""

  def __init__(self, filename: str, message_regex: str | None) -> None:
    """Initializes the instance.

    Args:
      filename: The filename to match in the log record's pathname.
      message_regex: The regex pattern to match against the log message
    """
    super().__init__()
    self._filename = filename
    self._message_regex = None
    if message_regex:
      self._message_regex = re.compile(message_regex)

  def filter(self, record: logging.LogRecord) -> bool:
    """Filters log records.

    Args:
      record: The log record to check.

    Returns:
      False if the record should be suppressed, True otherwise.
    """
    is_info_level = record.levelno == logging.INFO
    is_target_file = self._filename in record.pathname

    if is_info_level and is_target_file:
      message = record.getMessage()
      if self._message_regex is None:
        return False  # Drop this log message
      elif self._message_regex.search(message):
        return False

    return True  # Keep all other log messages