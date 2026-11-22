# LMS Student Rating Predictor (Streamlit)

Web app that predicts whether a student's rating is **High** or **Low** from midterm LMS activity counts, using the Random Forest from Task 2 (Version B).

## Files
| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `rf_model.joblib` | Trained model (saved by the Task 2 notebook) |
| `requirements.txt` | Python packages, pinned to the versions used for training |
| `sample_students.csv` | 20 example students for testing the batch upload |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

## Deploy on Streamlit Community Cloud (free)
1. Create a public GitHub repository and upload all files in this folder.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository, branch `main`, and main file `app.py`.
4. Click **Deploy**. The app gets a public `*.streamlit.app` link.

## Important: package versions
A joblib model must be loaded with the same scikit-learn version it was trained with.
If you re-run the notebook with a different scikit-learn version, update the version in
`requirements.txt` to match (`python -c "import sklearn; print(sklearn.__version__)"`).
