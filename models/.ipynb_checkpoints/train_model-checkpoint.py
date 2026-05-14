from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (

    mean_absolute_error,
    r2_score,

    accuracy_score
)

from models.model_selector import (
    select_best_model
)

# ==========================================
# TRAIN INDUSTRIAL AI MODEL
# ==========================================

def train_model(

    df,
    target_column
):

    print(
        "\nTRAINING INDUSTRIAL AI MODEL...\n"
    )

    # ======================================
    # FEATURES + TARGET
    # ======================================

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # ======================================
    # DETECT PROBLEM TYPE
    # ======================================

    if (

        str(y.dtype) == "object"

        or

        y.nunique() <= 10
    ):

        problem_type = (
            "classification"
        )

    else:

        problem_type = (
            "regression"
        )

    print(
        f"Detected Problem Type: "
        f"{problem_type}"
    )

    # ======================================
    # TRAIN TEST SPLIT
    # ======================================

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42
        )
    )

    # ======================================
    # AUTO MODEL SELECTION
    # ======================================

    (
        model,
        best_model_name

    ) = select_best_model(

        X_train,
        X_test,

        y_train,
        y_test,

        problem_type
    )

    # ======================================
    # FINAL PREDICTIONS
    # ======================================

    predictions = model.predict(
        X_test
    )

    # ======================================
    # REGRESSION METRICS
    # ======================================

    if problem_type == "regression":

        mae = mean_absolute_error(

            y_test,

            predictions
        )

        r2 = r2_score(

            y_test,

            predictions
        )

        print(
            f"\nBest Model: "
            f"{best_model_name}"
        )

        print(
            f"Forecast Error (MAE): "
            f"{mae:.2f}"
        )

        print(
            f"R² Score: "
            f"{r2:.4f}"
        )

    # ======================================
    # CLASSIFICATION METRICS
    # ======================================

    else:

        accuracy = accuracy_score(

            y_test,

            predictions
        )

        print(
            f"\nBest Model: "
            f"{best_model_name}"
        )

        print(
            f"Classification Accuracy: "
            f"{accuracy:.4f}"
        )

    print(
        "\nMODEL TRAINING COMPLETED!\n"
    )

    return (

        model,

        y_test,

        predictions,

        problem_type,

        best_model_name
    )