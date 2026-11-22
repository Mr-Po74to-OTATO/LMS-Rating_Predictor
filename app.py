"""
LMS Student Rating Predictor (Streamlit app)

Predicts whether a student's rating is High or Low from midterm LMS activity counts,
using the Random Forest trained in Task 2 (Version B) and saved as rf_model.joblib.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "rf_model.joblib"

FEATURE_HELP = {
    "Submission.Actions": "Number of submission-related actions",
    "Quiz.Actions": "Number of quiz-related actions",
    "Forum.Actions": "Number of forum-related actions",
    "Create.Actions": "Number of create-related actions",
    "Read.Actions": "Number of read-related actions",
    "Update.Actions": "Number of update-related actions",
    "Delete.Actions": "Number of delete-related actions",
    "Days.Logged.In": "Number of days the student logged into the LMS",
}

st.set_page_config(page_title="LMS Rating Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


def predict(model, features, df):
    """Return the input rows with predicted rating and probability of High."""
    X = df[features].astype(float)
    prob_high = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["Predicted rating"] = ["High" if p >= 0.5 else "Low" for p in prob_high]
    out["P(High)"] = prob_high.round(3)
    out["Needs support"] = out["Predicted rating"].eq("Low").map({True: "Yes", False: "No"})
    return out


try:
    model, features = load_model()
except FileNotFoundError:
    st.error("Model file `rf_model.joblib` not found. Place it in the same folder as `app.py`.")
    st.stop()

st.title("🎓 LMS Student Rating Predictor")
st.write(
    "Predicts whether a student is in the **High** or **Low** rating category from "
    "midterm LMS activity. Students predicted as **Low** may need additional academic support."
)
st.caption(
    "Model: Random Forest (bootstrap=False), 91.1% test accuracy. "
    "Predictions support instructor judgment and should not be used as the only basis for decisions."
)

tab_single, tab_batch = st.tabs(["Single student", "Batch (CSV upload)"])

# ---------------- Single student ----------------
with tab_single:
    st.subheader("Enter the student's activity counts")
    with st.form("single"):
        cols = st.columns(2)
        values = {}
        for i, f in enumerate(features):
            values[f] = cols[i % 2].number_input(
                f, min_value=0, value=0, step=1, help=FEATURE_HELP.get(f, "")
            )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        result = predict(model, features, pd.DataFrame([values])).iloc[0]
        p_high = float(result["P(High)"])
        if result["Predicted rating"] == "High":
            st.success(f"Predicted rating: **High** (probability {p_high:.1%})")
        else:
            st.warning(
                f"Predicted rating: **Low** (probability of Low {1 - p_high:.1%}). "
                "Consider reaching out to this student."
            )
        st.progress(p_high, text=f"P(High) = {p_high:.1%}")

# ---------------- Batch ----------------
with tab_batch:
    st.subheader("Upload a CSV of students")
    st.write("The file must contain these columns (extra columns such as a student ID are kept):")
    st.code(", ".join(features))

    template = pd.DataFrame(columns=["Student.ID"] + features)
    st.download_button(
        "Download CSV template", template.to_csv(index=False), "template.csv", "text/csv"
    )

    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

        missing = [f for f in features if f not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()

        bad = data[features].apply(pd.to_numeric, errors="coerce")
        if bad.isna().any().any() or (bad < 0).any().any():
            st.error("All activity columns must be non-negative numbers with no blanks.")
            st.stop()

        results = predict(model, features, data.drop(columns=["rating"], errors="ignore"))
        n_low = int(results["Predicted rating"].eq("Low").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Students", len(results))
        c2.metric("Predicted High", len(results) - n_low)
        c3.metric("Predicted Low", n_low)

        only_low = st.checkbox("Show only students predicted Low")
        view = results[results["Predicted rating"] == "Low"] if only_low else results
        st.dataframe(view.sort_values("P(High)"), use_container_width=True, hide_index=True)

        st.download_button(
            "Download predictions",
            results.to_csv(index=False),
            "predictions.csv",
            "text/csv",
            type="primary",
        )
