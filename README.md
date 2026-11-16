# Dementia Risk Predictor

An educational Streamlit dashboard for exploring a Random Forest trained on
OASIS-2 brain-scan measurements and patient details. This is a research and
educational decision-support prototype only. It has **not** been clinically
validated and must not be used for screening, diagnosis, triage, treatment, or
other real-world care. Its output is a model score, not a calibrated individual
clinical probability.

Do not enter or upload identifiable patient information or protected health
information (PHI). The dashboard does not connect to a clinical record system
or call an external prediction service. Inputs and uploaded CSV rows are
processed in the active app session and are not deliberately written to
persistent storage by the application. Hosting providers may have separate
logging, retention, or backup behavior. The deployment has no authentication,
authorization, or clinical-system integration. Obtain privacy, security, and
regulatory approval for the specific hosting environment before considering
any real patient data.

## Local setup

Use Python 3.12 (the version is recorded in `.python-version`) and install the
pinned app dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application requires the trusted `model.pkl` artifact and three published
summary CSV files in the project root. It validates their schema and values at
load time and reports a clear message if an artifact is missing or malformed.
The source OASIS data is not needed to run the dashboard.

The app supports light and dark appearance. In Streamlit, open the top-right
menu, choose **Settings**, then select **Light** or **Dark** under **Appearance**.
The browser remembers that preference. The dashboard theme, charts, and panels
adapt to the selected mode.

### Single-case and batch inputs

The Predict view supports the existing single-case form and CSV batch scoring.
Batch uploads are processed in the current app session, are not cached by the
app, and are limited to 5 MB and 500 rows. Use the in-app synthetic example to
see the CSV format. Required columns are `eTIV`, `nWBV`, `Age`, `SES`, and
`MMSE`, plus either `Sex` (`Female`/`F` or `Male`/`M`) or `M/F` (0 = female,
1 = male). An optional `record_id` must be unique and should be pseudonymous.
Other columns are rejected to reduce accidental exposure of unrelated data.
Blank, non-numeric, non-finite, out-of-domain, and out-of-training-range values
are rejected with a row number. The app derives `ASF` and normalizes the sex
encoding.

The downloaded batch file contains row numbers, optional pseudonymous IDs, and
the model score only. Scores are uncalibrated research outputs, not diagnoses,
clinical probabilities, or recommendations. The model was trained on a small
research cohort and is not validated for patients from other populations or
for clinical use.

## Deployment

For a non-sensitive demonstration on Streamlit Community Cloud, deploy this
repository's `main` branch, select `app.py` as the entry point, and use Python
3.12. `requirements.txt` pins the runtime libraries; the scikit-learn version
must remain `1.6.1` for the included model artifact. Keep `model.pkl` and the
displayed result CSV files in the deployed repository. The app uses
`.streamlit/config.toml` for its switchable clinical blue/teal theme, upload limit,
headless server mode, and disabled usage-stat collection. Never commit
`.streamlit/secrets.toml` or other secrets.

Community Cloud is not approved by this project for PHI. For any organization
deployment, select an environment only after its owner has reviewed
authentication, access controls, network boundaries, logging, retention,
encryption, and applicable privacy/regulatory requirements. No hosting target
for protected health data has been selected or certified here.

`joblib` model files are executable serialized artifacts: deploy only a model
you generated or obtained from a trusted source. The app checks its expected
features, class labels, ranges, and fitted input count; these checks do not
make an untrusted pickle safe to load.

## Data and model provenance

The modeling work uses OASIS-1 and OASIS-2 datasets from
[OASIS Brains](https://www.oasis-brains.org/), subject to their access terms.
The cleaned person-level data is not included. The analysis in this project
used a Kaggle copy of the OASIS CSV files while access was pending; follow the
dataset provider's terms and use authorized data for any new analysis.

`oasis_final_model.py` trains the dashboard model from
`oasis_longitudinal_cleaned.csv`, which must be prepared from authorized
OASIS-2 data. It writes the model artifact and analysis outputs; it no longer
rewrites the deployment dependency manifest. Before fitting, it checks required
columns, finite numeric values, target labels, clinical-value domains, subject
IDs, and per-class subject counts needed for grouped evaluation. Rebuild and
validate the model with the pinned environment before replacing the deployed
artifact. The app limits inputs to the saved training ranges and checks all
submitted values before inference.

## Project files

- `app.py`: Streamlit dashboard and user interface.
- `app_support.py`: model, single-case, batch CSV, and evaluation-table validation.
- `model.pkl`: fitted Random Forest and its feature/range metadata.
- `oasis_final_model.py`: model training and analysis script.
- `final_model_summary.csv`, `model_comparison_results.csv`,
  `oasis1_test_results.csv`: published evaluation summaries displayed by the app.
- `tests/test_app_support.py`: validation and artifact contract tests.
- `.github/workflows/tests.yml`: Python 3.12 CI test workflow.

## Evaluation and limitations

The included analyses report grouped cross-validation on a small OASIS-2
sample and an external OASIS-1 test. The final dashboard model is trained on all
OASIS-2 rows and has no prospective or clinical validation. The dataset is
limited to older adults, MMSE is closely related to the group label, and model
scores can reflect dataset bias or leakage. SHAP describes how the fitted
model uses its inputs; it does not establish causality. Reported results and
feature explanations are research artifacts, not evidence of clinical
performance or utility.
