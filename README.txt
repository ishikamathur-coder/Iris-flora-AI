# IRIS.AI — Iris Species Classification Website

## Run locally (Windows / VS Code)

1. Install Python 3.10 or newer.
2. Extract this ZIP folder.
3. Open the extracted folder in VS Code.
4. Open Terminal in VS Code.
5. (Recommended) Create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

6. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

7. Start the website:

   ```bash
   streamlit run app.py
   ```

8. Streamlit will open the website in your browser (usually at http://localhost:8501).

## What is included?

- Dark luxury visual theme: Jet Black `#050505`, Royal Amethyst `#3B0A6B`, Electric Cyan `#00E5FF`
- Iris species information cards
- Interactive four-measurement classifier
- Random Forest model trained from scikit-learn's built-in Iris dataset
- Prediction probabilities
- Test accuracy, classification report, confusion matrix and dataset preview

## Important project note

This app trains the Random Forest model automatically the first time it runs. It uses an 80/20 stratified train-test split and `random_state=42`, following the supplied project documentation. It does not need a separate `.joblib` file for this all-in-one starter version.

The flower photos are loaded from Wikimedia Commons and the hero background is loaded from Unsplash, so those images require an internet connection. The ML prediction itself runs locally after the Python packages are installed.
