from sklearn.ensemble import (

    RandomForestRegressor,
    RandomForestClassifier,

    GradientBoostingRegressor,
    GradientBoostingClassifier,

    ExtraTreesRegressor,
    ExtraTreesClassifier
)

from sklearn.metrics import (

    mean_absolute_error,
    accuracy_score
)

# ==========================================
# INDUSTRIAL MODEL SELECTOR
# ==========================================

def select_best_model(

    X_train,
    X_test,
    y_train,
    y_test,
    problem_type
):

    # ======================================
    # REGRESSION MODELS
    # ======================================

    if problem_type == "regression":

        models = {

            "Random Forest":

                RandomForestRegressor(

                    n_estimators=300,

                    max_depth=20,

                    random_state=42,

                    n_jobs=-1
                ),

            "Gradient Boosting":

                GradientBoostingRegressor(

                    n_estimators=300,

                    learning_rate=0.05,

                    max_depth=6,

                    random_state=42
                ),

            "Extra Trees":

                ExtraTreesRegressor(

                    n_estimators=300,

                    max_depth=20,

                    random_state=42,

                    n_jobs=-1
                )
        }

        best_model = None
        best_error = float("inf")
        best_name = None

        # ==================================
        # TRAIN ALL MODELS
        # ==================================

        for name, model in models.items():

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
                f"{name} MAE: {mae:.2f}"
            )

            # ------------------------------
            # BEST MODEL SELECTION
            # ------------------------------

            if mae < best_error:

                best_error = mae

                best_model = model

                best_name = name

        print(
            f"\nBEST MODEL: {best_name}"
        )

        print(
            f"LOWEST FORECAST ERROR: {best_error:.2f}"
        )

        return best_model

    # ======================================
    # CLASSIFICATION MODELS
    # ======================================

    else:

        models = {

            "Random Forest":

                RandomForestClassifier(

                    n_estimators=300,

                    max_depth=20,

                    random_state=42,

                    n_jobs=-1
                ),

            "Gradient Boosting":

                GradientBoostingClassifier(

                    n_estimators=300,

                    learning_rate=0.05,

                    random_state=42
                ),

            "Extra Trees":

                ExtraTreesClassifier(

                    n_estimators=300,

                    max_depth=20,

                    random_state=42,

                    n_jobs=-1
                )
        }

        best_model = None
        best_accuracy = 0
        best_name = None

        # ==================================
        # TRAIN ALL MODELS
        # ==================================

        for name, model in models.items():

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
                f"{name} Accuracy: {accuracy:.4f}"
            )

            # ------------------------------
            # BEST MODEL SELECTION
            # ------------------------------

            if accuracy > best_accuracy:

                best_accuracy = accuracy

                best_model = model

                best_name = name

        print(
            f"\nBEST MODEL: {best_name}"
        )

        print(
            f"HIGHEST ACCURACY: {best_accuracy:.4f}"
        )

        return best_model