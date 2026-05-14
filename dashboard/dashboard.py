import sys
import os

# ==========================================
# PROJECT ROOT PATH
# ==========================================

sys.path.append(

    os.path.abspath(

        os.path.join(

            os.path.dirname(__file__),

            ".."
        )
    )
)

# ==========================================
# IMPORTS
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import (

    mean_absolute_error,
    r2_score,
    accuracy_score
)

# ==========================================
# PREPROCESSING
# ==========================================

from preprocessing.detect_schema import (
    detect_schema
)

from preprocessing.clean_data import (
    clean_data
)

from preprocessing.outlier_handler import (
    remove_outliers
)

from preprocessing.feature_builder import (
    build_features
)

from preprocessing.encode_features import (
    encode_features
)

# ==========================================
# MODEL TRAINING
# ==========================================

from models.train_model import (
    train_model
)

# ==========================================
# VISUALIZATIONS
# ==========================================

from visualizations import (

    create_histogram,
    create_scatter,
    create_boxplot,

    create_pie_chart,

    create_feature_importance,

    create_actual_vs_predicted_chart,

    create_classification_comparison_chart,

    create_confusion_matrix_chart
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(

    page_title=
        "Industrial AI System",

    page_icon="🏭",

    layout="wide"
)

# ==========================================
# HEADER
# ==========================================

st.title(
    "🏭 Industrial AI Decision Intelligence System"
)

st.markdown("""

### AI-Based Industrial Analytics Platform

This platform supports:

- Demand Forecasting
- Cost Prediction
- Procurement Intelligence
- Safety Risk Analysis
- Industrial Decision Analytics

""")

# ==========================================
# CACHE DATA LOADING
# ==========================================

@st.cache_data

def load_datasets(uploaded_files):

    dataframes = []

    for uploaded_file in uploaded_files:

        # ----------------------------------
        # CSV
        # ----------------------------------

        if uploaded_file.name.endswith(".csv"):

            temp_df = pd.read_csv(
                uploaded_file
            )

        # ----------------------------------
        # EXCEL
        # ----------------------------------

        else:

            temp_df = pd.read_excel(
                uploaded_file
            )

        dataframes.append(temp_df)

    merged_df = pd.concat(

        dataframes,

        ignore_index=True
    )

    return merged_df

# ==========================================
# PREPROCESSING PIPELINE
# ==========================================

@st.cache_data

def preprocess_pipeline(df):

    schema = detect_schema(df)

    df = clean_data(df)

    df = remove_outliers(df)

    df = build_features(df)

    df, encoders = encode_features(df)

    return df, schema

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_files = st.file_uploader(

    "Upload CSV or Excel Files",

    type=["csv", "xlsx"],

    accept_multiple_files=True
)

# ==========================================
# MAIN PIPELINE
# ==========================================

if uploaded_files:

    # ======================================
    # LOAD DATA
    # ======================================

    df = load_datasets(
        uploaded_files
    )

    original_df = df.copy()

    # ======================================
    # LIGHTWEIGHT VISUALIZATION DATA
    # ======================================

    if len(original_df) > 1000:

        visualization_df = (

            original_df.sample(

                1000,

                random_state=42
            )
        )

    else:

        visualization_df = original_df

    # ======================================
    # KPI SECTION
    # ======================================

    st.success(
        f"{len(uploaded_files)} "
        f"dataset(s) uploaded successfully!"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        original_df.shape[0]
    )

    c2.metric(
        "Columns",
        original_df.shape[1]
    )

    c3.metric(
        "Files Uploaded",
        len(uploaded_files)
    )

    # ======================================
    # TABS
    # ======================================

    tab1, tab2, tab3, tab4 = st.tabs([

        "Dataset",

        "Preprocessing",

        "AI Training",

        "Analytics"
    ])

    # ======================================
    # TAB 1 — DATASET
    # ======================================

    with tab1:

        st.subheader(
            "Merged Industrial Dataset"
        )

        st.dataframe(

            original_df.head(10),

            use_container_width=True
        )

        st.write(
            f"Dataset Shape: "
            f"{original_df.shape}"
        )

    # ======================================
    # TAB 2 — PREPROCESSING
    # ======================================

    with tab2:

        with st.spinner(
            "Running preprocessing pipeline..."
        ):

            processed_df, schema = (

                preprocess_pipeline(df)
            )

            st.session_state[
                "processed_df"
            ] = processed_df

        st.success(
            "Preprocessing Completed!"
        )

        st.subheader(
            "Detected Schema"
        )

        st.json(schema)

        st.subheader(
            "Processed Dataset"
        )

        st.dataframe(

            processed_df.head(10),

            use_container_width=True
        )

    # ======================================
    # TAB 3 — AI TRAINING
    # ======================================

    with tab3:

        if (
            "processed_df"
            not in st.session_state
        ):

            st.warning(
                "Please preprocess "
                "the dataset first."
            )

        else:

            st.subheader(
                "Industrial AI Model"
            )

            target_column = st.selectbox(

                "Select Target Column",

                st.session_state[
                    "processed_df"
                ].columns
            )

            # ==================================
            # TRAINING SAMPLE
            # ==================================

            training_df = (

                st.session_state[
                    "processed_df"
                ]

                .sample(

                    min(

                        2000,

                        len(
                            st.session_state[
                                "processed_df"
                            ]
                        )
                    ),

                    random_state=42
                )
            )

            # ==================================
            # TRAIN BUTTON
            # ==================================

            if st.button(
                "Train AI Model"
            ):

                with st.spinner(
                    "Training Industrial AI Models..."
                ):

                    (
                        model,
                        y_test,
                        predictions,
                        problem_type,
                        best_model_name

                    ) = train_model(

                        training_df,

                        target_column
                    )

                st.success(
                    "Model Training Completed!"
                )

                st.info(
                    f"Best Selected Model: "
                    f"{best_model_name}"
                )

                # ==============================
                # PREDICTION DATAFRAME
                # ==============================

                prediction_df = pd.DataFrame({

                    "Actual":
                        y_test.values,

                    "Predicted":
                        predictions
                })

                # ==============================
                # REGRESSION
                # ==============================

                if problem_type == "regression":

                    mae = mean_absolute_error(

                        y_test,

                        predictions
                    )

                    r2 = r2_score(

                        y_test,

                        predictions
                    )

                    # ==========================
                    # KPI CARDS
                    # ==========================

                    k1, k2, k3, k4 = (
                        st.columns(4)
                    )

                    k1.metric(

                        "Average Actual",

                        f"{np.mean(y_test):.2f}"
                    )

                    k2.metric(

                        "Average Forecast",

                        f"{np.mean(predictions):.2f}"
                    )

                    k3.metric(

                        "Forecast Error",

                        f"{mae:.2f}"
                    )

                    k4.metric(

                        "R² Score",

                        f"{r2:.4f}"
                    )

                    # ==========================
                    # FORECAST CHART
                    # ==========================

                    st.subheader(
                        "Forecast Comparison"
                    )

                    forecast_chart = (

                        create_actual_vs_predicted_chart(

                            y_test,

                            predictions
                        )
                    )

                    st.plotly_chart(

                        forecast_chart,

                        use_container_width=True
                    )

                # ==============================
                # CLASSIFICATION
                # ==============================

                else:

                    accuracy = accuracy_score(

                        y_test,

                        predictions
                    )

                    st.metric(

                        "Classification Accuracy",

                        f"{accuracy:.4f}"
                    )

                    st.subheader(
                        "Classification Analysis"
                    )

                    classification_chart = (

                        create_classification_comparison_chart(

                            y_test,

                            predictions
                        )
                    )

                    st.plotly_chart(

                        classification_chart,

                        use_container_width=True
                    )

                    st.subheader(
                        "Confusion Matrix"
                    )

                    confusion_chart = (

                        create_confusion_matrix_chart(

                            y_test,

                            predictions
                        )
                    )

                    st.plotly_chart(

                        confusion_chart,

                        use_container_width=True
                    )

                # ==============================
                # FEATURE IMPORTANCE
                # ==============================

                st.subheader(
                    "Feature Importance"
                )

                feature_chart = (

                    create_feature_importance(

                        model,

                        training_df
                        .drop(
                            columns=[target_column]
                        )
                        .columns
                    )
                )

                if feature_chart:

                    st.plotly_chart(

                        feature_chart,

                        use_container_width=True
                    )

                # ==============================
                # RESULTS TABLE
                # ==============================

                st.subheader(
                    "Prediction Results"
                )

                st.dataframe(

                    prediction_df.head(20),

                    use_container_width=True
                )

                # ==============================
                # DOWNLOAD BUTTON
                # ==============================

                prediction_csv = (

                    prediction_df

                    .to_csv(index=False)

                    .encode("utf-8")
                )

                st.download_button(

                    label=
                        "Download Predicted Dataset",

                    data=prediction_csv,

                    file_name=
                        "predicted_results.csv",

                    mime="text/csv"
                )

    # ======================================
    # TAB 4 — ANALYTICS
    # ======================================

    with tab4:

        st.subheader(
            "Industrial Analytics"
        )

        numeric_columns = (

            visualization_df

            .select_dtypes(

                include=[
                    'int64',
                    'float64'
                ]
            )

            .columns

            .tolist()
        )

        # ==================================
        # HISTOGRAM
        # ==================================

        histogram_column = st.selectbox(

            "Distribution Analysis",

            numeric_columns
        )

        histogram = create_histogram(

            visualization_df,

            histogram_column
        )

        st.plotly_chart(

            histogram,

            use_container_width=True
        )

        # ==================================
        # PIE CHART
        # ==================================

        categorical_columns = (

            visualization_df

            .select_dtypes(
                include=['object']
            )

            .columns

            .tolist()
        )

        if categorical_columns:

            pie_column = st.selectbox(

                "Categorical Analysis",

                categorical_columns
            )

            pie_chart = create_pie_chart(

                visualization_df,

                pie_column
            )

            st.plotly_chart(

                pie_chart,

                use_container_width=True
            )

        # ==================================
        # SCATTER PLOT
        # ==================================

        if len(numeric_columns) >= 2:

            x_col = st.selectbox(

                "X Axis",

                numeric_columns,

                key="x"
            )

            y_col = st.selectbox(

                "Y Axis",

                numeric_columns,

                key="y"
            )

            scatter_df = (

                visualization_df.sample(

                    min(
                        300,
                        len(
                            visualization_df
                        )
                    ),

                    random_state=42
                )
            )

            scatter = create_scatter(

                scatter_df,

                x_col,

                y_col
            )

            st.plotly_chart(

                scatter,

                use_container_width=True
            )

        # ==================================
        # BOXPLOT
        # ==================================

        box_col = st.selectbox(

            "Boxplot Column",

            numeric_columns,

            key="box"
        )

        boxplot = create_boxplot(

            visualization_df,

            box_col
        )

        st.plotly_chart(

            boxplot,

            use_container_width=True
        )

    # ======================================
    # DOWNLOAD ORIGINAL DATASET
    # ======================================

    csv = (

        original_df

        .to_csv(index=False)

        .encode("utf-8")
    )

    st.download_button(

        label=
            "Download Original Dataset",

        data=csv,

        file_name=
            "industrial_dataset.csv",

        mime="text/csv"
    )