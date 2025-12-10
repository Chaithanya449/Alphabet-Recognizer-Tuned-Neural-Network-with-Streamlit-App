📘 Alphabet Recognizer – Tuned Neural Network Classification with Streamlit

A complete end-to-end Machine Learning project that predicts English alphabets (A–Z) using a feature-based Neural Network model, with an interactive Streamlit web app for real-time predictions.
The project includes model training, preprocessing, saving artifacts, and a fully functional UI for deployment.
🏗️ Architecture Diagram  


                    ┌──────────────────────────┐
                    │        Raw Dataset        │
                    │  (16 Numerical Features)  │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │    Data Preprocessing     │
                    │ - Scaling (StandardScaler)│
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   Neural Network Model    │
                    │ - Keras Sequential Model  │
                    │ - 26-class classification │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   Saved Artifacts         │
                    │ - Tuned_model.keras       │
                    │ - scaler.pkl              │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │     Streamlit App         │
                    │ - app.py                  │
                    │ - Loads model & scaler    │
                    │ - Predicts alphabet A–Z   │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Real-Time Predictions UI │
                    └──────────────────────────┘



This project demonstrates strong real-world ML engineering capabilities:

🔥 Neural Network development using TensorFlow/Keras

🔄 Full ML pipeline: preprocessing → training → tuning → evaluation

💾 Model packaging using .keras and .pkl

🌐 Interactive Streamlit web interface

📈 Real-time inference with confidence scores

📘 Jupyter Notebook with complete end-to-end workflow

Perfect to add under:
👉 Machine Learning Projects / AI Portfolio / End-to-End ML Systems

📂 Repository Structure (Actual Repo)
├── Neural_Network.ipynb        # Model training + preprocessing notebook
├── README.md                   # Project documentation
├── Tuned_model.keras           # Saved neural network model
├── app.py                      # Streamlit application
├── app_screenshot.png          # Screenshot of UI
├── requirements.txt            # Dependencies
├── scaler.pkl                  # Preprocessing scaler
└── test_data_for_app.csv       # Sample test dataset

🔧 How It Works

Dataset includes 16 engineered numerical features per alphabet

Neural Network predicts 26 classes (A to Z)

Model outputs:

Predicted class index

Mapped alphabet

Confidence score

Streamlit app:

Loads model & scaler

Selects a random test sample

Displays prediction + confidence

Shows probability distribution chart

▶️ Run the Project Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the Streamlit app
streamlit run app.py


App opens at:

http://localhost:8501

🧪 Model Training Workflow (Neural_Network.ipynb)

The notebook includes:

Data loading

Preprocessing using StandardScaler

Model architecture (Dense layers)

Training & validation

Hyperparameter tuning

Saving:

Tuned_model.keras

scaler.pkl

📈 Example Prediction Output

Predicted Letter: G

Confidence: 92.56%

Actual Label: G

Probability Chart: Displayed inside Streamlit UI

🐞 Common Issues
❌ KeyError: actual_label_index

Cause: CSV uses different column naming.

✔ Fix: Ensure the CSV contains the correct label column or adjust column name in app.py.

🌐 Deployment Ready

This project can be deployed easily using:

Streamlit Cloud (free & simplest)

Render

Railway

Heroku

Requirements for deployment:

app.py
Tuned_model.keras
scaler.pkl
test_data_for_app.csv
requirements.txt

🛠 Tech Stack

Python

TensorFlow / Keras

Pandas & NumPy

Scikit-Learn

Streamlit

Matplotlib / Seaborn


👤 Author

Chaitanya Krishna
Open to collaborations and improvements!
