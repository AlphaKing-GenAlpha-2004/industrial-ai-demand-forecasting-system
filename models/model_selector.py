from sklearn.ensemble import (

    ExtraTreesRegressor,
    ExtraTreesClassifier,

    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.metrics import (

    mean_absolute_error,
    accuracy_score
)

# ==========================================
# FAST INDUSTRIAL MODEL SELECTOR
# ==========================================

def select_best_model(

    X_train,
    X_test,

    y_train,
    y_test,

    problem_type
):

    # ======================================
    # REGRESSION
    # ======================================

    if problem_type == "regression":

        models = {

            "Extra Trees":

                ExtraTreesRegressor(

                    n_estimators=80,

                    max_depth=10,

                    random_state=42,

                    n_jobs=-1
                ),

            "Random Forest":

                RandomForestRegressor(

                    n_estimators=60,

                    max_depth=10,

                    random_state=42,

                    n_jobs=-1
                )
        }

        best_model = None
        best_error = float("inf")
        best_model_name = None

        # ==================================
        # TRAIN MODELS
        # ==================================

        for name, model in models.items():

            print(
                f"\nTraining {name}..."
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            mae = mean_absolute_error(

                y_test,

                predictions
            )

            print(
                f"{name} MAE: "
                f"{mae:.2f}"
            )

            # ------------------------------
            # BEST MODEL
            # ------------------------------

            if mae < best_error:

                best_error = mae

                best_model = model

                best_model_name = name

        print(
            f"\nBEST MODEL: "
            f"{best_model_name}"
        )

        print(
            f"LOWEST FORECAST ERROR: "
            f"{best_error:.2f}"
        )

        return (

            best_model,
            best_model_name
        )

    # ======================================
    # CLASSIFICATION
    # ======================================

    else:

        models = {

            "Extra Trees":

                ExtraTreesClassifier(

                    n_estimators=80,

                    max_depth=10,

                    random_state=42,

                    n_jobs=-1
                ),

            "Random Forest":

                RandomForestClassifier(

                    n_estimators=60,

                    max_depth=10,

                    random_state=42,

                    n_jobs=-1
                )
        }

        best_model = None
        best_accuracy = 0
        best_model_name = None

        # ==================================
        # TRAIN MODELS
        # ==================================

        for name, model in models.items():

            print(
                f"\nTraining {name}..."
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            accuracy = accuracy_score(

                y_test,

                predictions
            )

            print(
                f"{name} Accuracy: "
                f"{accuracy:.4f}"
            )

            # ------------------------------
            # BEST MODEL
            # ------------------------------

            if accuracy > best_accuracy:

                best_accuracy = accuracy

                best_model = model

                best_model_name = name

        print(
            f"\nBEST MODEL: "
            f"{best_model_name}"
        )

        return (

            best_model,
            best_model_name
        )