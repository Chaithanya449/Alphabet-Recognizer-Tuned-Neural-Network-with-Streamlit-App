import streamlit as st
import pandas as pd
import numpy as np 
import tensorflow as tf
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import string

#---Page Configuration---
st.set_page_config(page_title = 'Alphabet recognizer',layout = 'wide')
st.title('Neural Networks Alphabet Recognizer')
st.markdown("""
This app demonstrates a **Tuned Neural Network** (91.45% Accuracy).
Since the dataset uses **16 mathematical features** (not raw images), 
we test the model by picking random samples from the unseen test set.
""")

#---Load Artifacts---
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('Tuned_model.keras')
    with open('scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    data = pd.read_csv('test_data_for_app.csv')
    return model,scaler,data

#---Error Handling---
try:
    model,scaler,test_data= load_artifacts()
    st.success("✅The tuned neural network model is loaded Successfully!")
except Exception as e:
    st.error(f"Error loading files :{e} make sure the .keras, .pkl, .csv files are in the same folder")    

# ---Mapping Labels---
class_names = list(string.ascii_uppercase)

# --- SIDEBAR ---
st.sidebar.header("User Controls")
if st.sidebar.button("🎲 Pick Random Sample"):
    # 1. Select a random row
    random_row = test_data.sample(1)

    # 2. Separate features and actual label
    features_only = random_row.drop('Actual_label_index', axis=1)
    actual_index = int(random_row['Actual_label_index'].values[0])
    actual_letter = class_names[actual_index]

    # 3. Make Prediction
    prediction_probs = model.predict(features_only)
    predicted_index = np.argmax(prediction_probs)
    predicted_letter = class_names[predicted_index]
    confidence = np.max(prediction_probs) * 100

    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Prediction Result")
        st.metric(label="Actual Letter", value=actual_letter)
        st.metric(label="Predicted Letter", value=predicted_letter, 
                 delta=f"{confidence:.2f}% Confidence")

        if actual_letter == predicted_letter:
            st.success("Correct Prediction! 🎉")
        else:
            st.error("Incorrect Prediction ❌")

        st.caption("Input Features (First 5):")
        st.write(features_only.iloc[0, :5])

    with col2:
        st.subheader("Model 'Thinking' Process")
        prob_df = pd.DataFrame({
            'Letter': class_names,
            'Probability': prediction_probs[0]
        })

        top_3 = prob_df.nlargest(3, 'Probability')

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Letter', y='Probability', data=prob_df, color = "lightgray", ax=ax)
        sns.barplot(x='Letter', y='Probability', data=top_3, palette="viridis", ax=ax)

        ax.set_ylabel("Confidence")
        ax.set_xlabel("Letter")
        st.pyplot(fig)

else:
    st.info("👈 Click the button in the sidebar to test the model!")