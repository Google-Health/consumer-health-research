"""Constants for data table schemas.

This module defines the schemas for the DataFrames used in PHA,
as well as the custom PhdaDataFrame class with temporal filtering.
"""

import datetime
import re
from typing import Any, Dict, Literal, Optional, TypedDict

import pandas as pd


# Type definitions
DType = Literal["datetime64[ns]", "float64", "str", "int"]


class ColumnInfo(TypedDict):
  description: str
  dtype: DType


class DataFrameInfo(TypedDict):
  name: str
  description: str
  columns: Dict[str, ColumnInfo]
  variable_name: str


SUMMARY_SCHEMA: DataFrameInfo = {
    "name": "Summary DataFrame",
    "variable_name": "summary_df",
    "description": (
        "This is a summary of the user's activity, sleep, and personal health"
        " records data. Each row in the summary dataframe represents a single"
        " day."
    ),
    "columns": {
        "datetime": {
            "description": "The date of the record. This is the index column.",
            "dtype": "datetime64[ns]",
        },
        "steps": {
            "description": "The number of steps taken each day.",
            "dtype": "float64",
        },
        "sleep_minutes": {
            "description": "The total sleep time (in minutes) for each day.",
            "dtype": "float64",
        },
        "bed_time": {
            "description": "The time the user went to bed.",
            "dtype": "datetime64[ns]",
        },
        "wake_up_time": {
            "description": "The time the user woke up.",
            "dtype": "datetime64[ns]",
        },
        "resting_heart_rate": {
            "description": (
                "The average resting heart rate (in beats per minute) for each"
                " day."
            ),
            "dtype": "float64",
        },
        "heart_rate_variability": {
            "description": "The user's heart rate variability.",
            "dtype": "float64",
        },
        "active_zone_minutes": {
            "description": "The number of active zone minutes earned each day.",
            "dtype": "float64",
        },
        "deep_sleep_minutes": {
            "description": (
                "The amount of deep sleep (in minutes) for each day."
            ),
            "dtype": "float64",
        },
        "rem_sleep_minutes": {
            "description": "The amount of REM sleep (in minutes) for each day.",
            "dtype": "float64",
        },
        "light_sleep_minutes": {
            "description": (
                "The amount of light sleep (in minutes) for each day."
            ),
            "dtype": "float64",
        },
        "awake_minutes": {
            "description": (
                "The amount of awake time (in minutes) for each day."
            ),
            "dtype": "float64",
        },
        "stress_management_score": {
            "description": "The user's stress management score for each day.",
            "dtype": "float64",
        },
        "fatburn_active_zone_minutes": {
            "description": (
                "Part of the active zone minutes spent in the fat burn active"
                " zone which is calculated as (220 - age - resting_heart_rate)"
                " * 0.4 + resting_heart_rate."
            ),
            "dtype": "float64",
        },
        "cardio_active_zone_minutes": {
            "description": (
                "Part of the active zone minutes spent in the cardio active"
                " zone which is calculated as (220 - age - resting_heart_rate)"
                " * 0.6 + resting_heart_rate."
            ),
            "dtype": "float64",
        },
        "peak_active_zone_minutes": {
            "description": (
                "Part of the active zone minutes spent in the peak active zone"
                " which is calculated as (220 - age - resting_heart_rate) *"
                " 0.85 + resting_heart_rate."
            ),
            "dtype": "float64",
        },
    },
}

ACTIVITIES_SCHEMA: DataFrameInfo = {
    "name": "Activities DataFrame",
    "variable_name": "activities_df",
    "description": (
        "This is a table of the user's physical activities. Each row in the"
        " activities dataframe represents a single recorded activity."
    ),
    "columns": {
        "start_time": {
            "description": "The start time of the activity.",
            "dtype": "datetime64[ns]",
        },
        "end_time": {
            "description": "The end time of the activity.",
            "dtype": "datetime64[ns]",
        },
        "activity_name": {
            "description": (
                'The name of the activity. Possible values: "WALKING", "RUNNING",'
                ' "OUTDOOR_BIKE", "STRENGTH_TRAINING", "WORKOUT", "ELLIPTICAL",'
                ' "YOGA", "TREADMILL", "HIKING", "SWIMMING".'
            ),
            "dtype": "str",
        },
        "distance": {
            "description": "The distance covered during the activity in miles.",
            "dtype": "float64",
        },
        "duration": {
            "description": "The duration of the activity in minutes.",
            "dtype": "float64",
        },
        "calories": {
            "description": "The number of calories burned during the activity.",
            "dtype": "float64",
        },
        "steps": {
            "description": "The number of steps taken during the activity.",
            "dtype": "float64",
        },
        "active_zone_minutes": {
            "description": (
                "The number of active zone minutes earned during the activity."
            ),
            "dtype": "float64",
        },
    },
}


POPULATION_SCHEMA: DataFrameInfo = {
    "name": "Population DataFrame",
    "variable_name": "population_df",
    "description": (
        "The population dataframe represents data for each percentile of the"
        " population broken down by age and gender. Each row in the population"
        " dataframe represents a single percentile for a given age group and"
        " gender."
    ),
    "columns": {
        "percentile": {
            "description": "The percentile of the population.",
            "dtype": "int",
        },
        "age": {
            "description": (
                'The age group of the percentile, one of ["18-24", "25-34",'
                ' "35-44", "45-54", "55-64", "65+"].'
            ),
            "dtype": "str",
        },
        "gender": {
            "description": (
                'The gender of the percentile, one of ["male", "female"].'
            ),
            "dtype": "str",
        },
        "resting_heart_rate": {
            "description": "The resting heart rate for the percentile.",
            "dtype": "float64",
        },
        "heart_rate_variability": {
            "description": "The heart rate variability for the percentile.",
            "dtype": "float64",
        },
        "fatburn_active_zone_minutes": {
            "description": (
                "Part of the active zone minutes spent in the fat burn active"
                " zone for the percentile."
            ),
            "dtype": "float64",
        },
        "cardio_active_zone_minutes": {
            "description": (
                "Part of the active zone minutes spent in the cardio active"
                " zone for the percentile."
            ),
            "dtype": "float64",
        },
        "peak_active_zone_minutes": {
            "description": (
                "Part of the active zone minutes spent in the peak active"
                " zone for the percentile."
            ),
            "dtype": "float64",
        },
        "steps": {
            "description": (
                "The number of steps taken each day for the percentile."
            ),
            "dtype": "float64",
        },
        "rem_sleep_minutes": {
            "description": (
                "The amount of REM sleep (in minutes) for the percentile."
            ),
            "dtype": "float64",
        },
        "deep_sleep_minutes": {
            "description": (
                "The amount of deep sleep (in minutes) for the percentile."
            ),
            "dtype": "float64",
        },
        "light_sleep_minutes": {
            "description": (
                "The amount of light sleep (in minutes) for the percentile."
            ),
            "dtype": "float64",
        },
        "stress_management_score": {
            "description": "The stress management score for the percentile.",
            "dtype": "float64",
        },
    },
}


PROFILE_SCHEMA: DataFrameInfo = {
    "name": "Profile DataFrame",
    "variable_name": "profile_df",
    "description": "This is a table of the user's profile data.",
    "columns": {
        "age": {"description": "The age of the user.", "dtype": "int"},
        "gender": {"description": "The gender of the user.", "dtype": "str"},
        "height_cm": {
            "description": "The height of the user in centimeters.",
            "dtype": "int",
        },
        "weight_kg": {
            "description": "The weight of the user in kilograms.",
            "dtype": "int",
        },
        "averageDailySteps": {
            "description": "The user's average daily step count.",
            "dtype": "int",
        },
        "cluster": {
            "description": "The user's activity profile cluster label.",
            "dtype": "str",
        },
        "elderly": {
            "description": "Whether the user is classified as elderly.",
            "dtype": "bool",
        },
    },
}


DFS_INFO = [
    SUMMARY_SCHEMA,
    ACTIVITIES_SCHEMA,
    PROFILE_SCHEMA,
    POPULATION_SCHEMA,
]


class PhdaDataFrame(pd.DataFrame):
  """A DataFrame for DataAgent.

  It's provided to enable the `during` operation.
  """

  def during(
      self, time_expression: Any, last_days_alt: bool = False
  ) -> pd.DataFrame:
    """Returns a DataFrame filtered by the given time expression."""
    if not isinstance(self.index, pd.DatetimeIndex):
      print("Error: DataFrame index must be a DatetimeIndex")
      raise ValueError("DataFrame index must be a DatetimeIndex")

    if isinstance(time_expression, pd.Timestamp) and pd.isna(time_expression):
      print("Error: Time expression is an empty pd.Timestamp (NaT)")
      raise ValueError("Time expression is an empty pd.Timestamp (NaT)")

    if isinstance(time_expression, str):
      reference_date = self.index.max().date()
      start_date, end_date = self.get_date_range(
          time_expression, reference_date, last_days_alt=last_days_alt
      )
      start_date = (
          start_date.date() if hasattr(start_date, "date") else start_date
      )
      end_date = end_date.date() if hasattr(end_date, "date") else end_date
    elif isinstance(time_expression, pd.Series):
      if time_expression.empty:
        return PhdaDataFrame(columns=self.columns)
      start_date = time_expression.min().normalize().date()
      end_date = time_expression.max().normalize().date()
    elif isinstance(time_expression, pd.Timestamp):
      start_date = end_date = time_expression.normalize().date()
    else:
      print("Unsupported time expression type: ", time_expression)
      print(type(time_expression))
      raise ValueError("Unsupported time expression type")
    mask = (self.index.date >= start_date) & (self.index.date <= end_date)
    return self.loc[mask]

  def get_date_range(
      self,
      time_expression: str,
      reference_date: datetime.date,
      last_days_alt: bool = False,
  ) -> tuple[datetime.date, datetime.date]:
    """Gets the start and end dates for a given time expression."""

    # Match the 'last X days' pattern
    match = re.match(r"last (\d+) days", time_expression)

    if time_expression == "today":
      start_date = end_date = reference_date
    elif time_expression == "yesterday":
      start_date = end_date = reference_date - datetime.timedelta(days=1)
    elif match:
      days = int(match.group(1))
      if not last_days_alt:
        start_date = reference_date - datetime.timedelta(days=days)
        end_date = reference_date
      else:
        start_date = reference_date - datetime.timedelta(days=days - 1)
        end_date = reference_date
    else:
      raise ValueError(f"Unknown time expression: {time_expression}")

    return start_date, end_date


def enforce_persona_schema(
    summary: pd.DataFrame, activities: pd.DataFrame, population: pd.DataFrame
):
  """Remove useless columns and enforce datatypes for FitBit Advisor personas."""

  def _enforce(
      df,
      schema: Dict[str, ColumnInfo],
      index_col: Optional[str] = None,
  ):
    # Import here so that when we pull this into the sandbox it's available
    # # pylint: disable=g-import-not-at-top
    from pandas.api import types
    # pyformat: enable=g-import-not-at-top
    columns = []
    for col, column_info in schema.items():
      datatype = column_info["dtype"]
      if col in df.columns:
        if (
            types.is_datetime64_any_dtype(df[col])
            and datatype == "datetime64[ns]"
        ):
          # Pandas will throw a timezone warning if we do this naievely
          df[col] = df[col].dt.tz_localize(None)
          df[col] = df[col].astype("datetime64[ns]")
        else:
          df[col] = df[col].astype(datatype)
        columns.append(col)
    if index_col is not None:
      df.index = df[index_col]
    return df[columns]

  activities = _enforce(
      activities,
      ACTIVITIES_SCHEMA["columns"],
      index_col="start_time",
  )
  activities = PhdaDataFrame(activities)

  summary = _enforce(summary, SUMMARY_SCHEMA["columns"], index_col="datetime")
  summary = PhdaDataFrame(summary)

  population = _enforce(population, POPULATION_SCHEMA["columns"])
  population = PhdaDataFrame(population)

  return summary, activities, population
