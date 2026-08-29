"""Utility functions for loading and parsing data.

This is a set of functions for loading and parsing data
in the PHIA project.
"""

import collections
import datetime
import re
from typing import Any, Dict, List, Optional, Tuple, TypeVar
import pandas as pd

# To support type hints on methods returning an instance of the class
TChiaDataFrame = TypeVar('TChiaDataFrame', bound='ChiaDataFrame')

class ChiaDataFrame(pd.DataFrame):
    """
    A DataFrame subclass for CHIA-related data, adding time-based filtering.
    """

    def during(
        self, time_expression: Any, last_days_alt: bool = False
    ) -> TChiaDataFrame:
        """
        Returns a DataFrame filtered by the given time expression.

        Args:
            time_expression: A string (e.g., "today", "yesterday", "last 7 days"),
                             a pandas.Timestamp, or a pandas.Series of datetimes
                             to define the filtering range.
            last_days_alt: Boolean, alters 'last X days' to include the current day
                             as the end of the range.

        Returns:
            A filtered ChiaDataFrame.
        """
        if not isinstance(self.index, pd.DatetimeIndex):
            print("Error: DataFrame index must be a DatetimeIndex")
            raise ValueError("DataFrame index must be a DatetimeIndex")

        if isinstance(time_expression, pd.Timestamp) and pd.isna(time_expression):
            print("Error: Time expression is an empty pd.Timestamp (NaT)")
            raise ValueError("Time expression is an empty pd.Timestamp (NaT)")

        if self.index.empty:
            return self.copy()

        reference_date = self.index.max().date()

        if isinstance(time_expression, str):
            start_date, end_date = self._get_date_range(
                time_expression, reference_date, last_days_alt=last_days_alt
            )
            start_date = (
                start_date.date() if hasattr(start_date, "date") else start_date
            )
            end_date = end_date.date() if hasattr(end_date, "date") else end_date
        elif isinstance(time_expression, pd.Series):
            if time_expression.empty:
                return ChiaDataFrame(columns=self.columns)
            start_date = time_expression.min().normalize().date()
            end_date = time_expression.max().normalize().date()
        elif isinstance(time_expression, pd.Timestamp):
            start_date = end_date = time_expression.normalize().date()
        elif isinstance(time_expression, datetime.date):
            start_date = end_date = time_expression
        else:
            print(f"Unsupported time expression type: {type(time_expression)}")
            raise ValueError("Unsupported time expression type")

        mask = (self.index.date >= start_date) & (self.index.date <= end_date)
        return self.loc[mask]

    def _get_date_range(
        self,
        time_expression: str,
        reference_date: datetime.date,
        last_days_alt: bool = False,
    ) -> tuple[datetime.date, datetime.date]:
        """Gets the start and end dates for a given time expression."""
        time_expression = time_expression.lower().strip()
        match = re.match(r"last (\d+) days", time_expression)

        if time_expression == "today":
            start_date = end_date = reference_date
        elif time_expression == "yesterday":
            start_date = end_date = reference_date - datetime.timedelta(days=1)
        elif match:
            days = int(match.group(1))
            if days <= 0:
                raise ValueError("Number of days must be positive")
            if not last_days_alt:
                # e.g., "last 1 days" means "today"
                start_date = reference_date - datetime.timedelta(days=days - 1)
                end_date = reference_date
            else:
                # e.g., "last 1 days" means "yesterday"
                start_date = reference_date - datetime.timedelta(days=days)
                end_date = reference_date - datetime.timedelta(days=1)
        else:
            try:
                # Attempt to parse other date formats
                parsed_date = pd.to_datetime(time_expression).date()
                start_date = end_date = parsed_date
            except ValueError:
                raise ValueError(f"Unknown time expression: {time_expression}")

        return start_date, end_date

def unpack_nested_items(base_key, value, timestamp, dates):
    """Recursively unpacks a nested dict of items."""
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            full_key = f"{base_key}.{sub_key}" if base_key else sub_key
            unpack_nested_items(full_key, sub_value, timestamp, dates)
    else:
        dates[timestamp][base_key] = value

def unpack_daily_records(records: Dict[str, Any]) -> pd.DataFrame:
    """Unpacks daily records into a DataFrame."""
    dates = collections.defaultdict(dict)
    for k, stream in records.items():
        for item in stream:
            unpack_nested_items(k, item["value"], item["datetime"], dates)
    return pd.DataFrame(dates).T

def flatten_activity(data: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Flattens a nested activity dictionary."""
    items = {}
    for k, v in data.items():
        new_key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_activity(v, new_key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.update(flatten_activity(item, f"{new_key}_{i}"))
                else:
                    items[f"{new_key}_{i}"] = item
        else:
            items[new_key] = v
    return items

def unpack_activities(activities: List[Dict[str, Any]]) -> pd.DataFrame:
    """Unpacks a list of activities into a DataFrame."""
    flat_data = [flatten_activity(act) for act in activities]
    return pd.DataFrame(flat_data)

def _enforce_schema(df, schema):
    """Enforces a schema on a DataFrame."""
    columns = []
    df = df.copy()
    for col, dtype in schema["dtypes"].items():
        if col in df.columns:
            if df[col].isnull().all():
                if dtype == "datetime64[ns]":
                    # Pass  object dtype for all-NaN datetime columns
                    df[col] = df[col].astype("object")
                else:
                    df[col] = df[col].astype(dtype)
            elif dtype == "datetime64[ns]":
                try:
                    df[col] = pd.to_datetime(df[col])
                    if getattr(df[col].dt, 'tz', None) is not None:
                         # Make timezone-naive by stripping timezone info
                         print(f"Warning: Timezone found on column {col}. Making naive.")
                         df[col] = df[col].dt.tz_localize(None)
                except Exception as e:
                    print(f"Could not convert column {col} to datetime: {e}")
                    continue
            elif dtype == "str":
                 df[col] = df[col].astype(str)
            else:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Could not convert column {col} to type {dtype}: {e}")
                    continue
            columns.append(col)
        # else:
            # Optional: print(f"Warning: Column {col} not found in DataFrame for schema enforcement.")

    index_col = schema["index"]
    if index_col in df.columns:
        # Ensure index column is datetime before setting index
        if df[index_col].dtype != "datetime64[ns]":
             df[index_col] = pd.to_datetime(df[index_col])
        if getattr(df[index_col].dt, 'tz', None) is not None:
             df[index_col] = df[index_col].dt.tz_localize(None)
        df.index = df[index_col]
    elif df.index.name != index_col:
         print(f"Warning: Index column {index_col} not found to set index.")

    return df[columns]

def enforce_persona_schema(summary: pd.DataFrame, activities: pd.DataFrame) -> Tuple[ChiaDataFrame, ChiaDataFrame]:
    """Remove useless columns and enforce datatypes for Fitbit Advisor personas."""
    summary_schema = {
        "index": "datetime",
        "dtypes": {
            "datetime": "datetime64[ns]", 
            "resting_heart_rate": "float64", 
            "heart_rate_variability": "float64",
            "fatburn_active_zone_minutes": "float64", 
            "cardio_active_zone_minutes": "float64", 
            "peak_active_zone_minutes": "float64",
            "active_zone_minutes": "float64", 
            "steps": "float64", 
            "rem_sleep_minutes": "float64", 
            "deep_sleep_minutes": "float64",
            "awake_minutes": "float64", 
            "light_sleep_minutes": "float64", 
            "sleep_minutes": "float64", 
            "bed_time": "datetime64[ns]",
            "wake_up_time": "datetime64[ns]", 
            "stress_management_score": "float64",
            "deep_sleep_percent": "float64", 
            "rem_sleep_percent": "float64", 
            "awake_percent": "float64", 
            "light_sleep_percent": "float64",
        },
    }
    activity_schema = {
        "index": "startTime",
        "dtypes": {
            "startTime": "datetime64[ns]", 
            "endTime": "datetime64[ns]", 
            "activityName": "str",
            "distance": "float64", 
            "duration": "float64", 
            "elevationGain": "float64",
            "averageHeartRate": "float64", 
            "calories": "float64", 
            "steps": "float64", 
            "activeZoneMinutes": "float64",
            "speed": "float64",
        },
    }

    summary = _enforce_schema(summary, summary_schema)
    activities = _enforce_schema(activities, activity_schema)

    return ChiaDataFrame(summary), ChiaDataFrame(activities)

def localize_to_date(
    summary_df: pd.DataFrame,
    activities_df: pd.DataFrame,
    date: Optional[datetime.datetime | datetime.date | str] = "today",
) -> Tuple[ChiaDataFrame, ChiaDataFrame]:
    """Modifies timestamp columns so that the data appears to end on the given date."""
    summary_df = summary_df.copy()
    activities_df = activities_df.copy()

    if not isinstance(summary_df.index, pd.DatetimeIndex):
         summary_df.index = pd.to_datetime(summary_df.index)
    if not isinstance(activities_df.index, pd.DatetimeIndex):
         activities_df.index = pd.to_datetime(activities_df.index)

    if summary_df.index.empty or activities_df.index.empty:
        print("Warning: One or both DataFrames have an empty index. Skipping localization.")
        return ChiaDataFrame(summary_df), ChiaDataFrame(activities_df)

    latest_activity_date = activities_df.index.max().date()
    latest_summary_date = summary_df.index.max().date()
    latest_date = max(latest_activity_date, latest_summary_date)

    date_zero = pd.to_datetime(date).date()
    days_difference = (date_zero - latest_date).days
    delta = pd.Timedelta(days=days_difference)

    time_columns_summary = ["bed_time", "wake_up_time", "datetime"]
    for col in time_columns_summary:
        if col in summary_df.columns and summary_df[col].dtype == "datetime64[ns]":
            summary_df[col] = summary_df[col] + delta
    summary_df.index = summary_df.index + delta

    time_columns_activity = ["startTime", "endTime"]
    for col in time_columns_activity:
        if col in activities_df.columns and activities_df[col].dtype == "datetime64[ns]":
            activities_df.loc[:, col] = activities_df[col] + delta
    activities_df.index = activities_df.index + delta

    return ChiaDataFrame(summary_df), ChiaDataFrame(activities_df)

def load_persona(
    summary_path: str,
    activities_path: str,
    enforce_schema: bool = True,
    temporally_localize: bool | str = False,
) -> Tuple[ChiaDataFrame, ChiaDataFrame, pd.DataFrame]:
    """
    Loads persona data from CSV files.

    Args:
        summary_path: Path or URL to the summary CSV file.
        activities_path: Path or URL to the activities CSV file.
        enforce_schema: Whether to enforce the schema and data types.
        temporally_localize: False, True (for 'today'), or a date string
                             to shift the data to end on.

    Returns:
        Tuple of (summary_df, activities_df, profile_df).
    """
    try:
        summary = pd.read_csv(summary_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load summary CSV from {summary_path}: {e}")
    try:
        activities = pd.read_csv(activities_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load activities CSV from {activities_path}: {e}")

    # Extract Profile
    age = int(summary["age"].unique()[0]) if "age" in summary.columns and summary["age"].notna().any() else "Not Available"
    gender = summary["gender"].unique()[0] if "gender" in summary.columns and summary["gender"].notna().any() else "Not Available"
    mean_steps = int(summary["steps"].mean()) if "steps" in summary.columns and summary["steps"].notna().any() else "Not Available"
    elderly = "Not Available"
    if isinstance(age, int):
        elderly = "Yes" if age > 65 else "No"
    weight_kg = int(summary["weight_kg"].unique()[0]) if "weight_kg" in summary.columns and summary["weight_kg"].notna().any() else "Not Available"
    height_cm = int(summary["height_cm"].unique()[0]) if "height_cm" in summary.columns and summary["height_cm"].notna().any() else "Not Available"

    profile = pd.DataFrame([{
        "age": age, "gender": gender, "averageDailySteps": mean_steps,
        "elderly": elderly, "height_cm": height_cm, "weight_kg": weight_kg
    }])

    drop_cols = ["age", "gender", "weight_kg", "height_cm"]
    summary = summary.drop(columns=[col for col in drop_cols if col in summary.columns], errors='ignore')

    if enforce_schema:
        summary, activities = enforce_persona_schema(summary, activities)
    else:
        summary = ChiaDataFrame(summary)
        activities = ChiaDataFrame(activities)

    if temporally_localize:
        date_ref = "today" if isinstance(temporally_localize, bool) else temporally_localize
        summary, activities = localize_to_date(summary, activities, date_ref)

    return summary, activities, profile
