"""Serializing a dataframe into text to be used in a prompt."""

import abc
import json
from typing import Any, Dict, List, Optional, Union
import warnings

import pandas as pd
import yaml

from .data_schemas import DataFrameInfo


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


class DataSerializerBase(abc.ABC):

  @abc.abstractmethod
  def serialize(
      self,
      data: pd.DataFrame,
      df_info: Optional[DataFrameInfo] = None,
      use_yaml: bool = True,
      **kwargs,
  ) -> str:
    pass


class DataSerializer(DataSerializerBase):
  """DataSerializer is a class that serializes a pandas DataFrame into text to be used in a propmt.

  It provides methods to get properties of each column in a DataFrame,
  summarize data from a DataFrame or a file location, and serialize the data
  into YAML or JSON format.
  """

  def get_column_properties_df(
      self,
      data: pd.DataFrame,
      df_info: Optional[DataFrameInfo] = None,
      n_samples: int = 3,
  ) -> List[Dict[str, Any]]:
    """Get properties of each column in a pandas DataFrame."""
    properties_list = []
    for column in data.columns:
      if df_info is not None:
        if column not in df_info["columns"]:
          continue
        else:
          column_description = df_info["columns"][column].get("description", "")
      else:
        column_description = ""
      properties = self.get_column_properties_series(
          data[column], column_description, n_samples
      )
      properties_list.append({"column": column, "properties": properties})
    return properties_list

  def get_column_properties_series(
      self, series: pd.Series, column_description: str = "", n_samples: int = 3
  ) -> Dict[str, Any]:
    """Get properties of a pandas Series."""
    dtype = series.dtype
    properties = {}
    if dtype in [int, float, complex]:
      properties["dtype"] = "number"
      try:
        properties["std"] = series.std().tolist()
        properties["min"] = series.min().tolist()
        properties["max"] = series.max().tolist()
      except AttributeError:
        properties["std"] = series.std()
        properties["min"] = series.min()
        properties["max"] = series.max()
    elif dtype == bool:
      properties["dtype"] = "boolean"
    elif dtype == object:
      # Check if the string column can be cast to a valid datetime
      try:
        with warnings.catch_warnings():
          warnings.filterwarnings("ignore", category=Warning)
          pd.to_datetime(series, errors="raise")
        properties["dtype"] = "date"
      except (ValueError, TypeError):
        # Check if the string column has a limited number of values
        if series.nunique() / len(series) < 0.5:
          properties["dtype"] = "category"
        else:
          properties["dtype"] = "string"
    elif isinstance(series.dtype, pd.CategoricalDtype):
      properties["dtype"] = "category"
    elif pd.api.types.is_datetime64_any_dtype(series):
      properties["dtype"] = "date"
    else:
      properties["dtype"] = str(dtype)

    # add min max if dtype is date
    if properties["dtype"] == "date":
      try:
        properties["min"] = str(series.min())
        properties["max"] = str(series.max())
      except TypeError:
        cast_date_col = pd.to_datetime(series, errors="coerce")
        properties["min"] = str(cast_date_col.min())
        properties["max"] = str(cast_date_col.max())
    # Add additional properties to the output dictionary
    nunique = series.nunique()
    if "samples" not in properties:
      non_null_values = series[series.notnull()].unique()
      n_samples = min(n_samples, len(non_null_values))
      samples = (
          pd.Series(non_null_values).sample(n_samples, random_state=42).tolist()
      )
      properties["samples"] = samples
      if properties["dtype"] == "date":
        properties["samples"] = [str(s) for s in samples]
    properties["num_unique_values"] = nunique
    properties["description"] = column_description
    return properties

  def summarize(
      self,
      data: Union[pd.DataFrame, pd.Series],
      df_info: Optional[DataFrameInfo] = None,
      n_samples: int = 3,
  ) -> Dict[str, Any]:
    """Summarize data from a pandas DataFrame."""

    result = {}
    if isinstance(data, pd.DataFrame):
      data_properties = self.get_column_properties_df(
          data, df_info=df_info, n_samples=n_samples
      )
      result = {
          "dataset_description": (
              "" if df_info is None else df_info["description"]
          ),
          "fields": data_properties,
          "num_rows": len(data),
      }
      result["field_names"] = data.columns.tolist()
    elif isinstance(data, pd.Series):
      name = data.name
      column_description = (
          df_info["columns"][name].get("description", "")
          if df_info is not None
          else ""
      )
      result = self.get_column_properties_series(
          data, column_description, n_samples
      )
    return result

  def serialize(
      self,
      data: Union[pd.DataFrame, pd.Series],
      df_info: Optional[DataFrameInfo] = None,
      use_yaml: bool = True,
  ) -> str:
    """Serialize data from a pandas DataFrame or Series into YAML or JSON format.

    Args:
        data: The pandas DataFrame or Series to serialize.
        df_info: The DataFrameInfo object containing information about the
          DataFrame.
        use_yaml: Whether to use YAML or JSON format. Defaults to True.

    Returns:
        The serialized data in YAML or JSON format.
    """
    result = self.summarize(data, df_info=df_info)
    result = round_nested_floats(result)
    if use_yaml:
      return yaml.dump(result, indent=2)
    return json.dumps(result, indent=2)


def round_nested_floats(obj, decimals=3):
  if isinstance(obj, float):
    return round(obj, decimals)
  elif isinstance(obj, dict):
    return {k: round_nested_floats(v, decimals) for k, v in obj.items()}
  elif isinstance(obj, list):
    return [round_nested_floats(x, decimals) for x in obj]
  return obj


def serialize_dataframes(
    dfs: List[pd.DataFrame],
    dfs_info: List[DataFrameInfo],
    use_yaml: bool = True,
) -> str:
  """Serialize a list of dataframes into text to be used in a propmt.

  Args:
      dfs: The list of pandas DataFrames to serialize.
      dfs_info: The list of DataFrameInfo objects containing information about
        the DataFrames.
      use_yaml: Whether to use YAML or JSON format. Defaults to True.

  Returns:
      The serialized dataframes in string based on the format specified.
  """
  df_serializer = DataSerializer()
  s = ""
  for df, df_info in zip(dfs, dfs_info):
    s += f"#### {df_info['name']}\n"
    if len(df) <= 10:  # We will show the entire dataframe for small dataframes.
      s += f"\n{str(df)}"
    else:
      s += df_serializer.serialize(df, df_info, use_yaml)
    s += "\n\n"
  return s
