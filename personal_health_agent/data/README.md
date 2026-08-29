# PHA Data Format Documentation

This directory contains health data used by the Personal Health Agent (PHA). The `sample/` subdirectory includes synthetic data for testing and demonstration.

## Overview

PHA expects four CSV files:

| File | Purpose | Rows |
|------|---------|------|
| `summary.csv` | Daily aggregated health metrics | One row per day |
| `activities.csv` | Individual exercise/activity records | One row per activity |
| `profile.csv` | User demographic information | Single row |
| `population_percentiles.csv` | Population reference data | One row per percentile × age group × gender |

## Data Schemas

### summary.csv (Daily Summary)

Daily health metrics aggregated from wearable devices. Each row represents one day.

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | datetime | Date of the record (used as index) |
| `steps` | float | Total steps taken that day |
| `sleep_minutes` | float | Total sleep duration in minutes |
| `bed_time` | datetime | Time user went to bed |
| `wake_up_time` | datetime | Time user woke up |
| `resting_heart_rate` | float | Average resting heart rate (bpm) |
| `heart_rate_variability` | float | Heart rate variability (ms) |
| `active_zone_minutes` | float | Total active zone minutes |
| `deep_sleep_minutes` | float | Time in deep sleep (minutes) |
| `rem_sleep_minutes` | float | Time in REM sleep (minutes) |
| `light_sleep_minutes` | float | Time in light sleep (minutes) |
| `awake_minutes` | float | Time awake during sleep period (minutes) |
| `stress_management_score` | float | Stress score (higher = better management) |
| `fatburn_active_zone_minutes` | float | Minutes in fat burn heart rate zone |
| `cardio_active_zone_minutes` | float | Minutes in cardio heart rate zone |
| `peak_active_zone_minutes` | float | Minutes in peak heart rate zone |
| `sleep_score` | float | Overall sleep quality score (optional) |
| `readiness_score` | float | Daily readiness score (optional) |
| `cardio_load_total` | float | Total cardio load in TRIMP (optional) |

**Example:**
```csv
datetime,steps,sleep_minutes,bed_time,wake_up_time,resting_heart_rate,heart_rate_variability
2024-01-15,8523,425,2024-01-14 23:15:00,2024-01-15 06:20:00,62,45
2024-01-16,10241,390,2024-01-15 23:45:00,2024-01-16 06:15:00,60,48
```

### activities.csv (Exercise Activities)

Individual exercise and activity records. Each row represents one activity session.

| Column | Type | Description |
|--------|------|-------------|
| `start_time` | datetime | Activity start time (used as index) |
| `end_time` | datetime | Activity end time |
| `activity_name` | string | Type of activity (e.g., "RUNNING", "WALKING", "CYCLING") |
| `duration` | float | Duration in minutes |
| `distance` | float | Distance covered in miles |
| `averageHeartRate` | float | Average heart rate during activity (bpm) |
| `calories` | float | Calories burned |
| `steps` | float | Steps during activity (if applicable) |
| `active_zone_minutes` | float | Active zone minutes earned |
| `speed` | float | Average speed in mph (optional) |
| `cardio_load` | float | Cardio load in TRIMP (optional) |
| `elevation_gain` | float | Elevation gain in meters (optional) |

**Supported Activity Names:**
`WALKING`, `RUNNING`, `OUTDOOR_BIKE`, `BIKING`, `CYCLING`, `WORKOUT`, `TREADMILL`, `SWIMMING`, `YOGA`, `STRENGTH_TRAINING`, `HIIT`, `ELLIPTICAL`, `ROWING_MACHINE`, `HIKING`, and others.

**Example:**
```csv
start_time,end_time,activity_name,duration,distance,averageHeartRate,calories,steps
2024-01-15 07:30:00,2024-01-15 08:00:00,RUNNING,30,3.1,145,320,4200
2024-01-15 18:00:00,2024-01-15 18:45:00,WALKING,45,2.2,95,180,4800
```

### profile.csv (User Profile)

User demographic and physical information. Single row.

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | User's age in years |
| `gender` | string | "male" or "female" |
| `height_cm` | int | Height in centimeters |
| `weight_kg` | int | Weight in kilograms |
| `averageDailySteps` | int | Historical average daily steps (optional) |
| `cluster` | string | User archetype cluster (optional) |
| `elderly` | string | "Yes" or "No" (optional) |

**Example:**
```csv
age,gender,height_cm,weight_kg,averageDailySteps
34,male,178,75,8156
```

### population_percentiles.csv (Population Reference)

Population-level statistics for comparing user metrics against demographic peers. Used by agents to contextualize user data (e.g., "Your resting heart rate is in the 75th percentile for your age group").

| Column | Type | Description |
|--------|------|-------------|
| `percentile` | int | Percentile rank (1-100) |
| `age_group` or `age` | string | Age bracket (e.g., "18-24", "25-34", "35-44", "45-54", "55-64", "65+") |
| `gender` | string | "male" or "female" |
| `resting_heart_rate` | float | RHR at this percentile |
| `heart_rate_variability` | float | HRV at this percentile |
| `steps` | float | Daily steps at this percentile |
| `rem_sleep_minutes` | float | REM sleep at this percentile |
| `deep_sleep_minutes` | float | Deep sleep at this percentile |
| `light_sleep_minutes` | float | Light sleep at this percentile |
| `stress_management_score` | float | Stress score at this percentile |
| `fatburn_active_zone_minutes` | float | Fat burn AZM at this percentile |
| `cardio_active_zone_minutes` | float | Cardio AZM at this percentile |
| `peak_active_zone_minutes` | float | Peak AZM at this percentile |

**Example:**
```csv
percentile,age_group,gender,resting_heart_rate,heart_rate_variability,steps
1,25-34,male,51,15,1200
50,25-34,male,68,32,7500
99,25-34,male,85,65,15000
```

## Using Your Own Data

### Option 1: Replace Sample Files

Replace the files in `data/sample/` with your own CSVs following the schemas above. The system will automatically load them.

### Option 2: Custom Data Directory

```python
from config.settings import Settings

settings = Settings(data_dir="/path/to/your/data")
```

### Option 3: Direct DataFrame Loading

```python
from pha.agents import DataScienceAgent
import pandas as pd

agent = DataScienceAgent()
agent.load_dataframes({
    'summary_df': pd.read_csv('my_summary.csv'),
    'activities_df': pd.read_csv('my_activities.csv'),
    'profile_df': pd.read_csv('my_profile.csv'),
    'population_df': pd.read_csv('my_population.csv'),
})
```

## Temporal Localization

By default, PHA shifts dates in the data so that the most recent record appears as "today". This allows sample data to feel current. Disable with:

```python
settings = Settings(temporally_localize=False)
```

## Sample Data

The `sample/` directory contains ~365 days of synthetic health data for a fictional 34-year-old male. This data is entirely fabricated for demonstration purposes and does not represent any real individual.

## Data Sources for Real Usage

To use PHA with real data, you can export from:
- **Fitbit**: Export via Google Takeout or Fitbit's data export feature
- **Apple Health**: Export via the Health app's "Export All Health Data" feature
- **Garmin**: Export via Garmin Connect
- **Other wearables**: Most devices offer data export; you may need to transform the format

Note: You may need to write preprocessing scripts to convert exported data into the expected CSV format.
