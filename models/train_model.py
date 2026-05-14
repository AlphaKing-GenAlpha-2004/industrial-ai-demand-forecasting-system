from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (

    mean_absolute_error,
    r2_score,

    accuracy_score
)

from model_selector import (
    select_best_model
)

# ==========================================
# TRAIN INDUSTRIAL AI MODEL
# ==========================================

def train_model(

    df,
    target_column
):

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

    if y.dtype == "object":

        problem_type = "classification"

    else:

        problem_type = "regression"

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

    model = select_best_model(

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
    # METRICS
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
            f"\nForecast Error (MAE): {mae:.2f}"
        )

        print(
            f"R2 Score: {r2:.4f}"
        )

    else:

        accuracy = accuracy_score(

            y_test,

            predictions
        )

        print(
            f"\nClassification Accuracy: "
            f"{accuracy:.4f}"
        )

    return (

        model,

        y_test,

        predictions,

        problem_type
    )