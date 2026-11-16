from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_FEATURES = ("eTIV", "nWBV", "ASF", "Age", "SES", "MMSE", "M/F")
IMAGING_FEATURES = ("eTIV", "nWBV", "ASF")
CLINICAL_FEATURES = ("Age", "SES", "MMSE", "M/F")
INPUT_RANGE_FEATURES = ("eTIV", "nWBV", "Age", "MMSE")
TRAINING_COLUMNS = ("Subject ID", "Group_Binary_Encoded", *MODEL_FEATURES)


class ModelBundleError(ValueError):
    """Raised when a serialized model does not match the app's input contract."""


class InputValidationError(ValueError):
    """Raised when a prediction input is invalid or outside the training domain."""


class ResultDataError(ValueError):
    """Raised when a displayed evaluation table is missing or malformed."""


class TrainingDataError(ValueError):
    """Raised when source data cannot safely be used to train the model."""


RESULT_TABLES = {
    "final_model_summary.csv": {
        "columns": ("Feature set", "Mean ROC-AUC", "Lowest seed", "Highest seed"),
        "scores": ("Mean ROC-AUC", "Lowest seed", "Highest seed"),
    },
    "model_comparison_results.csv": {
        "columns": (
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "AUC lowest seed",
            "AUC highest seed",
        ),
        "scores": (
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "AUC lowest seed",
            "AUC highest seed",
        ),
    },
    "oasis1_test_results.csv": {
        "columns": ("Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"),
        "scores": ("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"),
    },
}
BATCH_REQUIRED_COLUMNS = ("eTIV", "nWBV", "Age", "Sex", "SES", "MMSE")
BATCH_RECORD_ID_COLUMN = "record_id"
MAX_BATCH_ROWS = 500


def validate_training_frame(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise TrainingDataError("The training dataset is empty or is not tabular data.")
    if frame.columns.has_duplicates:
        raise TrainingDataError("The training dataset contains duplicate columns.")
    missing_columns = set(TRAINING_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise TrainingDataError(
            f"The training dataset is missing required columns: {', '.join(sorted(missing_columns))}."
        )

    validated = frame.loc[:, list(TRAINING_COLUMNS)].copy()
    subject_ids = validated["Subject ID"]
    if subject_ids.isna().any() or subject_ids.astype(str).str.strip().eq("").any():
        raise TrainingDataError("Subject ID values must be present for grouped evaluation.")

    numeric_columns = ("Group_Binary_Encoded", *MODEL_FEATURES)
    for column in numeric_columns:
        values = pd.to_numeric(validated[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise TrainingDataError(f"The training column {column} contains invalid values.")
        validated[column] = values

    labels = set(validated["Group_Binary_Encoded"].unique())
    if labels != {0, 1}:
        raise TrainingDataError("The encoded target must contain both labels 0 and 1 only.")
    if (validated["eTIV"] <= 0).any() or (validated["ASF"] <= 0).any():
        raise TrainingDataError("eTIV and ASF values must be positive.")
    if ((validated["nWBV"] <= 0) | (validated["nWBV"] >= 1)).any():
        raise TrainingDataError("nWBV values must be greater than 0 and less than 1.")
    if ((validated["Age"] < 60) | (validated["Age"] > 120)).any():
        raise TrainingDataError("Age values must be between 60 and 120 for this study.")
    for column, low, high in (("SES", 1, 5), ("MMSE", 0, 30), ("M/F", 0, 1)):
        values = validated[column]
        if ((values < low) | (values > high) | (values % 1 != 0)).any():
            raise TrainingDataError(f"{column} values must be integers from {low} to {high}.")

    if (validated["Group_Binary_Encoded"] % 1 != 0).any():
        raise TrainingDataError("The encoded target must contain integer labels.")
    subjects_per_class = validated.groupby("Group_Binary_Encoded")["Subject ID"].nunique()
    if len(subjects_per_class) != 2 or (subjects_per_class < 5).any():
        raise TrainingDataError(
            "At least five distinct subjects from each target class are required for grouped five-fold evaluation."
        )
    return validated


def validate_model_bundle(bundle):
    if not isinstance(bundle, Mapping):
        raise ModelBundleError("The model artifact must contain a mapping.")

    required_keys = {"model", "features", "imaging", "clinical", "ranges", "asf_const"}
    missing_keys = required_keys.difference(bundle)
    if missing_keys:
        raise ModelBundleError(
            f"The model artifact is missing required fields: {', '.join(sorted(missing_keys))}."
        )

    for key, expected, message in (
        ("features", MODEL_FEATURES, "The model feature names or order do not match this app."),
        ("imaging", IMAGING_FEATURES, "The model imaging feature group is invalid."),
        ("clinical", CLINICAL_FEATURES, "The model clinical feature group is invalid."),
    ):
        names = bundle[key]
        if not isinstance(names, (list, tuple)) or tuple(names) != expected:
            raise ModelBundleError(message)

    model = bundle["model"]
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelBundleError("The model must implement predict_proba.")
    try:
        classes = np.asarray(model.classes_, dtype=float)
        feature_count = int(model.n_features_in_)
    except (AttributeError, TypeError, ValueError) as error:
        raise ModelBundleError("The model is missing fitted class or feature metadata.") from error
    if classes.shape != (2,) or not np.array_equal(np.sort(classes), np.array([0.0, 1.0])):
        raise ModelBundleError("The model must be fitted for the binary labels 0 and 1.")
    if feature_count != len(MODEL_FEATURES):
        raise ModelBundleError("The model's fitted feature count does not match this app.")

    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None and tuple(model_feature_names) != MODEL_FEATURES:
        raise ModelBundleError("The model's fitted feature order does not match this app.")

    ranges = bundle["ranges"]
    if not isinstance(ranges, Mapping) or set(ranges) != set(INPUT_RANGE_FEATURES):
        raise ModelBundleError("The model's input ranges are missing or unexpected.")
    for feature in INPUT_RANGE_FEATURES:
        try:
            bounds = np.asarray(ranges[feature], dtype=float)
        except (TypeError, ValueError) as error:
            raise ModelBundleError(f"The model range for {feature} is invalid.") from error
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ModelBundleError(f"The model range for {feature} must be finite and increasing.")

        low, high = bounds
        domain_low, domain_high = {
            "eTIV": (0, np.inf),
            "nWBV": (0, 1),
            "Age": (60, 120),
            "MMSE": (0, 30),
        }[feature]
        if (
            low < domain_low
            or high > domain_high
            or (feature == "eTIV" and low <= 0)
            or (feature == "nWBV" and (low <= 0 or high >= 1))
        ):
            raise ModelBundleError(f"The model range for {feature} is outside its valid domain.")

    try:
        asf_constant = float(bundle["asf_const"])
    except (TypeError, ValueError) as error:
        raise ModelBundleError("The model's ASF constant must be a positive finite number.") from error
    if not np.isfinite(asf_constant) or asf_constant <= 0:
        raise ModelBundleError("The model's ASF constant must be a positive finite number.")


def validate_prediction_input(values, ranges):
    if not isinstance(values, Mapping) or set(values) != set(MODEL_FEATURES):
        raise InputValidationError("Prediction inputs do not match the model feature schema.")
    try:
        numeric_values = {name: float(values[name]) for name in MODEL_FEATURES}
    except (TypeError, ValueError) as error:
        raise InputValidationError("Every prediction input must be numeric.") from error
    if not all(np.isfinite(value) for value in numeric_values.values()):
        raise InputValidationError("Prediction inputs must all be finite numbers.")
    if numeric_values["eTIV"] <= 0:
        raise InputValidationError("eTIV must be positive.")
    if not 0 < numeric_values["nWBV"] < 1:
        raise InputValidationError("nWBV must be greater than 0 and less than 1.")
    if not 60 <= numeric_values["Age"] <= 120:
        raise InputValidationError("Age must be between 60 and 120 for this study.")
    if not 0 <= numeric_values["MMSE"] <= 30:
        raise InputValidationError("MMSE must be between 0 and 30.")

    try:
        for feature in INPUT_RANGE_FEATURES:
            low, high = (float(bound) for bound in ranges[feature])
            value = numeric_values[feature]
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError
            if not low <= value <= high:
                raise InputValidationError(
                    f"{feature} must be between {low:g} and {high:g}, matching the training data."
                )
    except InputValidationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise InputValidationError("The model's input ranges are invalid.") from error
    if not numeric_values["ASF"] > 0:
        raise InputValidationError("Derived ASF must be positive.")
    for feature in ("Age", "MMSE", "SES", "M/F"):
        if numeric_values[feature] % 1 != 0:
            raise InputValidationError(f"{feature} must be a whole number.")
    if numeric_values["SES"] not in (1, 2, 3, 4, 5):
        raise InputValidationError("Socioeconomic status must be an integer from 1 to 5.")
    if numeric_values["M/F"] not in (0, 1):
        raise InputValidationError("Sex encoding must be 0 (female) or 1 (male).")
    return numeric_values


def validate_batch_frame(frame, ranges, asf_const, max_rows=MAX_BATCH_ROWS):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise InputValidationError("The CSV must contain at least one data row.")
    if frame.columns.has_duplicates:
        raise InputValidationError("The CSV contains duplicate column names.")
    if len(frame) > max_rows:
        raise InputValidationError(f"The CSV may contain at most {max_rows} rows.")

    columns = set(frame.columns)
    sex_columns = columns.intersection({"Sex", "M/F"})
    if not sex_columns:
        raise InputValidationError("The CSV must include a Sex or M/F column.")
    if len(sex_columns) != 1:
        raise InputValidationError("Use either Sex or M/F, not both.")
    sex_column = next(iter(sex_columns))
    required = (set(BATCH_REQUIRED_COLUMNS) - {"Sex"}) | {sex_column}
    missing = required.difference(columns)
    if missing:
        raise InputValidationError(
            f"The CSV is missing required columns: {', '.join(sorted(missing))}."
        )
    unexpected = columns.difference(required | {BATCH_RECORD_ID_COLUMN})
    if unexpected:
        raise InputValidationError(
            "The CSV has unsupported columns: "
            f"{', '.join(sorted(str(column) for column in unexpected))}. "
            "Remove identifiers and other fields except the optional record_id."
        )

    checked_columns = ("eTIV", "nWBV", "Age", sex_column, "SES", "MMSE")
    checked = frame.loc[:, list(checked_columns)].copy()
    if BATCH_RECORD_ID_COLUMN in columns:
        record_ids = frame[BATCH_RECORD_ID_COLUMN]
        if (
            record_ids.isna().any()
            or record_ids.astype(str).str.strip().eq("").any()
            or record_ids.astype(str).str.strip().duplicated().any()
        ):
            raise InputValidationError(
                "Optional record_id values must be non-empty and unique. "
                "Use pseudonymous identifiers only."
            )
        checked[BATCH_RECORD_ID_COLUMN] = record_ids.astype(str).str.strip().to_numpy()

    for feature in ("eTIV", "nWBV", "Age", "SES", "MMSE"):
        values = pd.to_numeric(checked[feature], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise InputValidationError(
                f"The CSV column {feature} must contain finite numeric values."
            )
        checked[feature] = values

    if (checked["eTIV"] <= 0).any():
        raise InputValidationError("CSV eTIV values must be positive.")
    if sex_column == "Sex":
        sex_values = checked["Sex"].astype("string").str.strip().str.casefold()
        sex_encoding = {"female": 0, "f": 0, "male": 1, "m": 1}
        encoded_sex = sex_values.map(sex_encoding)
        if encoded_sex.isna().any():
            raise InputValidationError(
                "Sex must be Female, F, Male, or M in every CSV row."
            )
    else:
        encoded_sex = pd.to_numeric(checked["M/F"], errors="coerce")
        if (
            encoded_sex.isna().any()
            or not np.isfinite(encoded_sex.to_numpy(dtype=float)).all()
            or not encoded_sex.isin((0, 1)).all()
        ):
            raise InputValidationError("M/F values must be 0 (female) or 1 (male).")

    try:
        asf_constant = float(asf_const)
    except (TypeError, ValueError) as error:
        raise InputValidationError("The model's ASF constant is invalid.") from error
    if not np.isfinite(asf_constant) or asf_constant <= 0:
        raise InputValidationError("The model's ASF constant must be positive and finite.")
    checked["ASF"] = asf_constant / checked["eTIV"]
    checked["M/F"] = encoded_sex.astype(int)
    model_rows = checked.loc[:, list(MODEL_FEATURES)]
    validated_rows = []
    for row_number, (_, row) in enumerate(model_rows.iterrows(), start=1):
        try:
            validated_rows.append(validate_prediction_input(row.to_dict(), ranges))
        except InputValidationError as error:
            raise InputValidationError(f"CSV row {row_number}: {error}") from error

    validated = pd.DataFrame(validated_rows, columns=MODEL_FEATURES)
    if BATCH_RECORD_ID_COLUMN in checked:
        validated[BATCH_RECORD_ID_COLUMN] = checked[BATCH_RECORD_ID_COLUMN].to_numpy()
    return validated


def positive_class_probability(model, rows):
    probabilities = np.asarray(model.predict_proba(rows), dtype=float)
    classes = np.asarray(model.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if positive_indices.size != 1 or probabilities.shape != (len(rows), len(classes)):
        raise ValueError("The model returned an invalid probability result.")
    scores = probabilities[:, positive_indices[0]]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("The model returned a non-finite or out-of-range score.")
    return scores


def positive_class_shap_values(shap_values, feature_count):
    if isinstance(shap_values, list):
        if len(shap_values) != 2:
            raise ValueError("The explanation has an unexpected number of classes.")
        values = np.asarray(shap_values[1], dtype=float)
    else:
        values = np.asarray(shap_values, dtype=float)
        if values.ndim == 3:
            if values.shape[0] != 1 or values.shape[2] != 2:
                raise ValueError("The explanation has an unexpected shape.")
            values = values[0, :, 1]
        elif values.ndim == 2:
            if values.shape[0] != 1:
                raise ValueError("The explanation must describe exactly one input.")
            values = values[0]
        else:
            raise ValueError("The explanation has an unexpected shape.")

    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError("The explanation must describe exactly one input.")
        values = values[0]
    if values.shape != (feature_count,) or not np.isfinite(values).all():
        raise ValueError("The explanation values do not match the model features.")
    return values


def validate_result_table(frame, expected_columns, score_columns):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ResultDataError("The results table is empty or is not tabular data.")
    if frame.columns.has_duplicates or set(frame.columns) != set(expected_columns):
        raise ResultDataError(
            "The results table columns do not match the expected published summary."
        )

    validated = frame.loc[:, list(expected_columns)].copy()
    text_columns = [column for column in expected_columns if column not in score_columns]
    for column in text_columns:
        if validated[column].isna().any() or validated[column].astype(str).str.strip().eq("").any():
            raise ResultDataError(f"The results table contains a blank value in {column}.")
        validated[column] = validated[column].astype(str)

    for column in score_columns:
        values = pd.to_numeric(validated[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ResultDataError(f"The results table contains an invalid score in {column}.")
        if ((values < 0) | (values > 1)).any():
            raise ResultDataError(f"Scores in {column} must be between 0 and 1.")
        validated[column] = values
    return validated


def read_result_table(path, expected_columns, score_columns):
    try:
        frame = pd.read_csv(Path(path))
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ResultDataError(f"Could not read results file: {Path(path).name}.") from error
    return validate_result_table(frame, expected_columns, score_columns)
