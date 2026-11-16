import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app_support import (
    MODEL_FEATURES,
    RESULT_TABLES,
    InputValidationError,
    ModelBundleError,
    ResultDataError,
    positive_class_probability,
    positive_class_shap_values,
    read_result_table,
    validate_batch_frame,
    validate_model_bundle,
    validate_prediction_input,
    validate_result_table,
    validate_training_frame,
    TrainingDataError,
)


ROOT = Path(__file__).resolve().parents[1]


class ReversedClassModel:
    classes_ = np.array([1, 0])

    def predict_proba(self, rows):
        return np.tile([0.8, 0.2], (len(rows), 1))


class AppSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(ROOT / "model.pkl")

    def test_included_model_matches_app_contract_and_predicts(self):
        validate_model_bundle(self.bundle)
        values = {
            "eTIV": 1474,
            "nWBV": 0.739,
            "ASF": self.bundle["asf_const"] / 1474,
            "Age": 77,
            "SES": 2,
            "MMSE": 29,
            "M/F": 0,
        }
        checked = validate_prediction_input(values, self.bundle["ranges"])
        row = pd.DataFrame([[checked[name] for name in MODEL_FEATURES]], columns=MODEL_FEATURES)
        probability = positive_class_probability(self.bundle["model"], row)
        self.assertEqual(probability.shape, (1,))
        self.assertTrue(0 <= probability[0] <= 1)

    def test_streamlit_app_renders_with_included_artifacts(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(
            any("Dementia-group model score" in metric.label for metric in app.metric)
        )
        app.slider(key="etiv").set_value(1600)
        app.button(key="update_estimate").click().run(timeout=20)
        self.assertEqual(app.session_state["etiv"], 1600)
        app.button(key="reset_inputs").click().run(timeout=20)
        self.assertEqual(app.session_state["etiv"], 1469)
        app.segmented_control(key="dashboard_view").set_value("Batch CSV").run(
            timeout=20
        )
        self.assertFalse(app.exception)
        self.assertTrue(
            any(uploader.label == "Upload a CSV file" for uploader in app.file_uploader)
        )
        for section in ("How it works", "Results and limits"):
            app.segmented_control(key="dashboard_view").set_value(section).run(
                timeout=20
            )
            self.assertFalse(app.exception, f"Dashboard section failed: {section}")

    def test_rejects_mismatched_model_features(self):
        bundle = dict(self.bundle)
        bundle["features"] = list(reversed(bundle["features"]))
        with self.assertRaises(ModelBundleError):
            validate_model_bundle(bundle)

    def test_training_data_is_validated_for_schema_ranges_and_group_counts(self):
        frame = pd.DataFrame(
            [
                {
                    "Subject ID": f"subject-{index}",
                    "Group_Binary_Encoded": index % 2,
                    "eTIV": 1400 + index,
                    "nWBV": 0.7,
                    "ASF": 1.2,
                    "Age": 65 + index,
                    "SES": 2,
                    "MMSE": 25,
                    "M/F": index % 2,
                }
                for index in range(10)
            ]
        )
        self.assertEqual(len(validate_training_frame(frame)), 10)

        invalid = frame.drop(columns=["MMSE"])
        with self.assertRaises(TrainingDataError):
            validate_training_frame(invalid)

        invalid = frame.copy()
        invalid.loc[0, "nWBV"] = np.nan
        with self.assertRaises(TrainingDataError):
            validate_training_frame(invalid)

        invalid = frame.copy()
        invalid["Subject ID"] = "one-subject"
        with self.assertRaises(TrainingDataError):
            validate_training_frame(invalid)

    def test_rejects_invalid_input_values_and_out_of_training_range(self):
        values = {
            "eTIV": 1474,
            "nWBV": 0.739,
            "ASF": self.bundle["asf_const"] / 1474,
            "Age": 77,
            "SES": 2,
            "MMSE": 29,
            "M/F": 0,
        }
        invalid_values = dict(values, MMSE=float("nan"))
        with self.assertRaises(InputValidationError):
            validate_prediction_input(invalid_values, self.bundle["ranges"])
        invalid_values = dict(values, Age=59)
        with self.assertRaises(InputValidationError):
            validate_prediction_input(invalid_values, self.bundle["ranges"])

    def test_batch_csv_is_normalized_and_uses_pseudonymous_ids(self):
        frame = pd.DataFrame(
            {
                "record_id": ["case-001", "case-002"],
                "MMSE": [29, 27],
                "Sex": ["Female", "M"],
                "Age": [77, 76],
                "SES": [2, 2],
                "nWBV": [0.739, 0.713],
                "eTIV": [1474, 1463],
            }
        )
        checked = validate_batch_frame(
            frame, self.bundle["ranges"], self.bundle["asf_const"]
        )
        self.assertEqual(list(checked.columns), [*MODEL_FEATURES, "record_id"])
        self.assertEqual(checked["M/F"].tolist(), [0, 1])
        self.assertEqual(checked["record_id"].tolist(), ["case-001", "case-002"])
        self.assertTrue(np.isfinite(checked["ASF"]).all())
        batch_scores = positive_class_probability(
            self.bundle["model"], checked[list(MODEL_FEATURES)]
        )
        self.assertEqual(batch_scores.shape, (2,))
        self.assertTrue(np.isfinite(batch_scores).all())

        model_columns = frame.drop(columns=["record_id", "Sex"]).assign(
            **{"M/F": [0, 1]}
        )
        model_columns = model_columns[
            ["eTIV", "nWBV", "Age", "M/F", "SES", "MMSE"]
        ]
        from_model_schema = validate_batch_frame(
            model_columns, self.bundle["ranges"], self.bundle["asf_const"]
        )
        self.assertEqual(from_model_schema["M/F"].tolist(), [0, 1])

    def test_batch_csv_rejects_missing_extra_and_invalid_values(self):
        frame = pd.DataFrame(
            {
                "eTIV": [1474],
                "nWBV": [0.739],
                "Age": [77],
                "Sex": ["Female"],
                "SES": [2],
                "MMSE": [29],
            }
        )
        with self.assertRaises(InputValidationError):
            validate_batch_frame(
                frame.drop(columns=["MMSE"]),
                self.bundle["ranges"],
                self.bundle["asf_const"],
            )
        with self.assertRaises(InputValidationError):
            validate_batch_frame(
                frame.assign(patient_name=["Not allowed"]),
                self.bundle["ranges"],
                self.bundle["asf_const"],
            )
        with self.assertRaisesRegex(InputValidationError, "Age must be a whole number"):
            validate_batch_frame(
                frame.assign(Age=[77.5]),
                self.bundle["ranges"],
                self.bundle["asf_const"],
            )
        with self.assertRaisesRegex(InputValidationError, "CSV row 1"):
            validate_batch_frame(
                frame.assign(MMSE=[31]),
                self.bundle["ranges"],
                self.bundle["asf_const"],
            )
        with self.assertRaises(InputValidationError):
            validate_batch_frame(
                frame.assign(Sex=["unknown"]),
                self.bundle["ranges"],
                self.bundle["asf_const"],
            )

    def test_batch_csv_rejects_duplicate_ids_and_oversized_batches(self):
        frame = pd.DataFrame(
            {
                "record_id": ["same", "same"],
                "eTIV": [1474, 1463],
                "nWBV": [0.739, 0.713],
                "Age": [77, 76],
                "Sex": ["Female", "Male"],
                "SES": [2, 2],
                "MMSE": [29, 27],
            }
        )
        with self.assertRaisesRegex(InputValidationError, "unique"):
            validate_batch_frame(
                frame, self.bundle["ranges"], self.bundle["asf_const"]
            )
        frame = frame.drop(columns=["record_id"])
        with self.assertRaisesRegex(InputValidationError, "at most 1 rows"):
            validate_batch_frame(
                frame,
                self.bundle["ranges"],
                self.bundle["asf_const"],
                max_rows=1,
            )

    def test_probability_uses_positive_label_not_column_position(self):
        score = positive_class_probability(
            ReversedClassModel(), pd.DataFrame({"value": [1, 2]})
        )
        np.testing.assert_array_equal(score, [0.8, 0.8])

    def test_shap_output_shapes_are_checked(self):
        np.testing.assert_array_equal(
            positive_class_shap_values(np.zeros((1, len(MODEL_FEATURES), 2)), 7),
            np.zeros(7),
        )
        np.testing.assert_array_equal(
            positive_class_shap_values([np.zeros((1, 7)), np.ones((1, 7))], 7),
            np.ones(7),
        )
        with self.assertRaises(ValueError):
            positive_class_shap_values(np.zeros((1, 6)), 7)

    def test_published_result_tables_load_with_expected_schema(self):
        for filename, schema in RESULT_TABLES.items():
            with self.subTest(filename=filename):
                result = read_result_table(
                    ROOT / filename, schema["columns"], schema["scores"]
                )
                self.assertFalse(result.empty)

    def test_result_table_rejects_invalid_scores(self):
        frame = pd.DataFrame(
            {"Model": ["RF"], "Accuracy": [1.2], "Precision": [0.8]}
        )
        with self.assertRaises(ResultDataError):
            validate_result_table(frame, ("Model", "Accuracy", "Precision"), ("Accuracy", "Precision"))


if __name__ == "__main__":
    unittest.main()
