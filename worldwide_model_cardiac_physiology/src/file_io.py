# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filesystem adapter shared by every library and notebook of this package.

All reads/writes go through the helpers in this module. Files live under
``./data`` or the directory given by the ``WPM_DATA_DIR`` environment variable.

Directory layout under the data root::

    <root>/dummy_data.csv, dummy_data_hrv.csv   cohort inputs (public schema)
    <root>/extra_data/                          GeoJSON, UNSD reference tables
    <root>/results/<run_tag>/                   one modelling run
    <root>/results/paper/                       tables + figures (optional)
"""

import os
import shutil
from typing import Any

# Storage back-end; None = local filesystem.
_env: Any = None

_DATA_DIR_ENV_VAR = 'WPM_DATA_DIR'
_LOCAL_DATA_SUBDIR = 'data'

WEARABLE_SUBDIR = 'wearable_data'
EXTRA_SUBDIR = 'extra_data'
RESULTS_SUBDIR = 'results'
PAPER_RUN_TAG = 'paper'


# --- Environment -------------------------------------------------------------




def _resolve_local_data_dir() -> str:
  """Returns ``$WPM_DATA_DIR``, else ``./data`` if present, else ``<repo>/data``.

  ``<repo>`` is the parent of the ``src/`` directory holding this module, so
  notebooks opened from ``notebooks/`` and scripts run from the repository
  root resolve the same ``data/`` folder.
  """
  override = os.environ.get(_DATA_DIR_ENV_VAR)
  if override:
    return override
  cwd_candidate = os.path.join(os.getcwd(), _LOCAL_DATA_SUBDIR)
  if os.path.isdir(cwd_candidate):
    return cwd_candidate
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  return os.path.join(repo_root, _LOCAL_DATA_SUBDIR)


def get_data_dir(default_dir: str | None = None) -> str:
  """Returns the data root directory (no trailing slash).

  Args:
    default_dir: Optional explicit root; takes precedence over the environment
      default but not over the ``WPM_DATA_DIR`` environment variable.
  """
  if _env is not None:
    root = _env.get_data_dir(default_dir)
  else:
    root = default_dir or _resolve_local_data_dir()
  return root.rstrip('/')


def get_wearable_dir(data_dir: str | None = None) -> str:
  """Returns ``<root>/wearable_data``."""
  return os.path.join(data_dir or get_data_dir(), WEARABLE_SUBDIR)


def get_extra_dir(data_dir: str | None = None) -> str:
  """Returns ``<root>/extra_data``."""
  return os.path.join(data_dir or get_data_dir(), EXTRA_SUBDIR)


def get_results_dir(run_tag: str, data_dir: str | None = None) -> str:
  """Returns ``<root>/results/<run_tag>`` (not created)."""
  return os.path.join(data_dir or get_data_dir(), RESULTS_SUBDIR, run_tag)


def get_paper_dir(data_dir: str | None = None) -> str:
  """Returns ``<root>/results/paper`` (tables and figures written by 01/03)."""
  return get_results_dir(PAPER_RUN_TAG, data_dir)




# --- File operations ---------------------------------------------------------


def make_dirs(path: str) -> None:
  """Creates ``path`` and its parents (no-op if it exists or is empty)."""
  if not path:
    return
  if _env is not None:
    _env.make_dirs(path)
  else:
    os.makedirs(path, exist_ok=True)


def file_exists(path: str) -> bool:
  """Returns True if ``path`` (file or directory) exists."""
  if _env is not None:
    return _env.file_exists(path)
  return os.path.exists(path)


def open_file(path: str, mode: str = 'r') -> Any:
  """Opens ``path``; parent directories are created for write modes."""
  if any(flag in mode for flag in ('w', 'a', 'x')):
    make_dirs(os.path.dirname(path))
  if _env is not None:
    return _env.open_file(path, mode)
  return open(path, mode)


def list_dir(path: str) -> list[str]:
  """Returns the entry base names of directory ``path``."""
  if _env is not None:
    return _env.list_dir(path)
  return os.listdir(path)


def copy_file(src: str, dst: str, overwrite: bool = False) -> None:
  """Copies ``src`` to ``dst``."""
  if _env is not None:
    _env.copy_file(src, dst, overwrite=overwrite)
    return
  if not overwrite and os.path.exists(dst):
    raise FileExistsError(dst)
  make_dirs(os.path.dirname(dst))
  shutil.copyfile(src, dst)


def remove_file(path: str) -> None:
  """Removes file ``path`` (no-op if it does not exist)."""
  if _env is not None:
    _env.remove_file(path)
  elif os.path.exists(path):
    os.remove(path)
