import pandas as pd
import numpy as np

# ==========================================
# INDUSTRIAL FEATURE ENGINEERING
# ==========================================

def build_features(df):

    print(
        "\nBUILDING INDUSTRIAL FEATURES...\n"
    )

    # ======================================
    # NUMERIC COLUMNS
    # ======================================

    numeric_columns = (

        df.select_dtypes(

            include=[
                'int64',
                'float64'
            ]
        )
        .columns
        .tolist()
    )

    # ======================================
    # DEMAND-INVENTORY RATIO
    # ======================================

    if (
        "historical_demand" in df.columns
        and
        "inventory_level" in df.columns
    ):

        df[
            "demand_inventory_ratio"
        ] = (

            df["historical_demand"]

            /

            (
                df["inventory_level"]
                + 1
            )
        )

    # ======================================
    # COST PER LABOR HOUR
    # ======================================

    if (
        "production_cost" in df.columns
        and
        "labor_hours" in df.columns
    ):

        df[
            "cost_per_labor_hour"
        ] = (

            df["production_cost"]

            /

            (
                df["labor_hours"]
                + 1
            )
        )

    # ======================================
    # MACHINE EFFICIENCY
    # ======================================

    if (
        "machine_utilization" in df.columns
        and
        "production_cost" in df.columns
    ):

        df[
            "machine_efficiency"
        ] = (

            df["machine_utilization"]

            /

            (
                df["production_cost"]
                + 1
            )
        )

    # ======================================
    # SUPPLIER PERFORMANCE
    # ======================================

    if (
        "supplier_rating" in df.columns
        and
        "lead_time" in df.columns
    ):

        df[
            "supplier_performance"
        ] = (

            df["supplier_rating"]

            /

            (
                df["lead_time"]
                + 1
            )
        )

    # ======================================
    # TEMPERATURE-PRESSURE INDEX
    # ======================================

    if (
        "temperature" in df.columns
        and
        "pressure" in df.columns
    ):

        df[
            "temp_pressure_index"
        ] = (

            df["temperature"]

            *
            df["pressure"]
        )

    # ======================================
    # LIMITED INTERACTION FEATURES
    # ======================================

    important_pairs = [

        (
            "historical_demand",
            "inventory_level"
        ),

        (
            "production_cost",
            "labor_hours"
        ),

        (
            "machine_utilization",
            "pressure"
        ),

        (
            "temperature",
            "pressure"
        )
    ]

    for col1, col2 in important_pairs:

        if (
            col1 in df.columns
            and
            col2 in df.columns
        ):

            feature_name = (

                f"{col1}_"
                f"{col2}_interaction"
            )

            df[feature_name] = (

                df[col1]
                *
                df[col2]
            )

    # ======================================
    # LOG FEATURES
    # ======================================

    for col in numeric_columns:

        if (

            col in df.columns

            and

            df[col].min() >= 0
        ):

            df[
                f"{col}_log"
            ] = np.log1p(
                df[col]
            )

    # ======================================
    # ROLLING MEAN FEATURES
    # ======================================

    for col in numeric_columns[:3]:

        if col in df.columns:

            df[
                f"{col}_rolling_mean"
            ] = (

                df[col]

                .rolling(
                    window=5,
                    min_periods=1
                )

                .mean()
            )

    print(
        "Feature Engineering Completed!"
    )

    print(
        f"Total Features: "
        f"{df.shape[1]}"
    )

    return df