"""Feature engineering module for Forest Cover Type Classifier."""
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all 8 domain-informed feature engineering transformations.

    Args:
        df: Raw DataFrame with original Forest Cover Type features.

    Returns:
        DataFrame with 8 additional engineered features.
    """
    df = df.copy()

    df["euclidean_dist_to_hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    df["hillshade_mean"] = (
        df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]
    ) / 3

    df["hillshade_range"] = df["Hillshade_9am"] - df["Hillshade_3pm"]

    df["elevation_water_level"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    df["slope_hydrology_interaction"] = (
        df["Slope"] * df["Horizontal_Distance_To_Hydrology"]
    )

    df["human_impact_distance"] = (
        df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    )

    df["aspect_sin"] = np.sin(df["Aspect"] * np.pi / 180)
    df["aspect_cos"] = np.cos(df["Aspect"] * np.pi / 180)

    return df
