# ARP Spoofing Detection — Streamlit Demo

Run the Streamlit web app to test the saved MLP model on packet features.

Setup

1. Create a virtual environment (recommended) and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app

```bash
streamlit run app.py
```

Notes
- The repo includes `model/mlp_model.pkl` and `model/scaler.pkl` which are loaded by the app.
- The app uses `dataset/ARP-dataset.csv` to fit label encoders for categorical fields so manual inputs match training encodings.
- Use "Upload CSV" to run batch predictions; uploaded CSV must contain the same feature columns (excluding `Label`).
