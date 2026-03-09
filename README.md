# 🔤 Alphabet Recognizer – Tuned Neural Network with Streamlit

An **end-to-end Machine Learning system** that predicts English alphabets (A–Z) using a **tuned Neural Network model** and provides **real-time predictions through an interactive Streamlit web application**.

The project demonstrates the full ML pipeline including **data preprocessing, model training, hyperparameter tuning, artifact packaging, and deployment**.

🔗 **Live Demo**  
https://alphabet-recognizer-tuned-neural-network-with-app-app-m7du4dl9.streamlit.app/

---

# 🚀 Project Highlights

- Neural Network classifier for **26 alphabet classes**
- **Feature scaling using StandardScaler**
- End-to-end ML pipeline: preprocessing → training → tuning → evaluation
- Model artifact packaging (`.keras` + `.pkl`)
- **Interactive Streamlit web interface**
- **Real-time predictions with confidence scores**
- Cloud deployment using **Streamlit Community Cloud**

---




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



---

# 📊 Dataset

- Each record contains **16 engineered numerical features**
- Target variable represents **alphabet classes (A–Z)**
- Total classes: **26**

The dataset is used to train a **multi-class neural network classifier**.

---

# 🧠 Model Details

Neural Network built using **TensorFlow / Keras**

Architecture:
Input Layer (16 features)
↓
Dense Layer (ReLU)
↓
Dense Layer (ReLU)
↓
Output Layer (Softmax – 26 classes)

Model outputs:

- Predicted alphabet
- Confidence score
- Class probability distribution

---

# 📂 Repository Structure

Alphabet-Recognizer-Tuned-Neural-Network-with-Streamlit-App/

│
├── Neural_Network.ipynb
├── app.py
├── Tuned_model.keras
├── scaler.pkl
├── test_data_for_app.csv
├── requirements.txt
├── app_screenshot.png
└── README.md

---

# ⚙️ Running the Project Locally

### 1️⃣ Install dependencies

### 2️⃣ Start the Streamlit app

The application will open at:
http://localhost:8501

---

# 📈 Example Prediction
Predicted Letter: G
Confidence: 92.56%
Actual Label: G

The Streamlit interface also displays a **probability distribution chart** for all classes.

---

# 🛠 Tech Stack

**Languages**
- Python

**Machine Learning**
- TensorFlow / Keras
- Scikit-learn

**Data Processing**
- Pandas
- NumPy

**Visualization**
- Matplotlib
- Seaborn

**Deployment**
- Streamlit
- Streamlit Community Cloud

---

# 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**, demonstrating the ability to move a machine learning model from development to a **publicly accessible web application**.

Requirements for deployment:
app.py
Tuned_model.keras
scaler.pkl
test_data_for_app.csv
requirements.txt

---

# 👨‍💻 Author

**Chaitanya Krishna**

GitHub  
https://github.com/Chaithanya449

LinkedIn  
https://www.linkedin.com/in/chaitanyakrishna-profile

---

⭐ If you found this project useful, consider **starring the repository**.
