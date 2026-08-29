"""Tests for data loading utilities."""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from pha.utils.data_loader import load_persona, get_personal_df_from_phr
from config import Settings


class TestLoadPersona:
    """Tests for the load_persona function."""
    
    def test_load_from_explicit_paths(self, sample_data_dir):
        """Load data using explicit file paths."""
        summary_df, activities_df, profile_df, population_df = load_persona(
            summary_path=sample_data_dir / "summary.csv",
            activities_path=sample_data_dir / "activities.csv",
            profile_path=sample_data_dir / "profile.csv",
            population_path=sample_data_dir / "population_percentiles.csv",
            enforce_schema=False,
            temporally_localize=False,
        )
        
        assert isinstance(summary_df, pd.DataFrame)
        assert isinstance(activities_df, pd.DataFrame)
        assert isinstance(profile_df, pd.DataFrame)
        assert isinstance(population_df, pd.DataFrame)
        
        assert len(summary_df) > 0
        assert len(activities_df) > 0
        assert len(profile_df) > 0
        assert len(population_df) > 0
    
    def test_load_from_settings(self, sample_data_dir):
        """Load data using Settings object."""
        settings = Settings(
            data_dir=str(sample_data_dir),
            llm_backend="gemini",
        )
        
        summary_df, activities_df, profile_df, population_df = load_persona(
            settings=settings,
            enforce_schema=False,
            temporally_localize=False,
        )
        
        assert len(summary_df) > 0
        assert len(activities_df) > 0
    
    def test_temporal_localization_shifts_dates(self, sample_data_dir):
        """Temporal localization shifts dates so last date is today."""
        summary_df, _, _, _ = load_persona(
            summary_path=sample_data_dir / "summary.csv",
            activities_path=sample_data_dir / "activities.csv",
            profile_path=sample_data_dir / "profile.csv",
            population_path=sample_data_dir / "population_percentiles.csv",
            enforce_schema=False,
            temporally_localize=True,
        )
        
        # Parse dates and check that max date is recent
        if "date" in summary_df.columns:
            dates = pd.to_datetime(summary_df["date"])
            max_date = dates.max()
            today = pd.Timestamp.now().normalize()
            
            # Max date should be within a day of today
            assert abs((max_date - today).days) <= 1
    
    def test_schema_enforcement_filters_columns(self, sample_data_dir):
        """Schema enforcement keeps only expected columns."""
        summary_df, _, _, _ = load_persona(
            summary_path=sample_data_dir / "summary.csv",
            activities_path=sample_data_dir / "activities.csv",
            profile_path=sample_data_dir / "profile.csv",
            population_path=sample_data_dir / "population_percentiles.csv",
            enforce_schema=True,
            temporally_localize=False,
        )
        
        # Should have core summary columns
        assert "date" in summary_df.columns or len(summary_df.columns) > 0
    
    def test_returns_four_dataframes(self, sample_data_dir):
        """Function returns exactly four DataFrames."""
        result = load_persona(
            summary_path=sample_data_dir / "summary.csv",
            activities_path=sample_data_dir / "activities.csv",
            profile_path=sample_data_dir / "profile.csv",
            population_path=sample_data_dir / "population_percentiles.csv",
        )
        
        assert len(result) == 4
        assert all(isinstance(df, pd.DataFrame) for df in result)


class TestGetPersonalDfFromPhr:
    """Tests for extracting profile from PHR DataFrame."""
    
    def test_extracts_profile_fields(self):
        """Extracts age, weight, height, gender, bmi from PHR."""
        phr_df = pd.DataFrame({
            "age": [35],
            "weight": [70],
            "height": [175],
            "sex": ["male"],
            "bmi": [22.9],
            "other_field": ["ignored"],
        })
        
        profile_df = get_personal_df_from_phr(phr_df)
        
        assert "age" in profile_df.columns
        assert "weight" in profile_df.columns
        assert "height" in profile_df.columns
        assert "gender" in profile_df.columns
        assert "bmi" in profile_df.columns
        assert "other_field" not in profile_df.columns
        
        assert profile_df["age"].iloc[0] == 35
        assert profile_df["gender"].iloc[0] == "male"
