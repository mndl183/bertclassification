import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import requests
import os
import tempfile
import zipfile
import shutil

# Set page configuration
st.set_page_config(
    page_title="Text Classification with BERT",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS to improve appearance and responsiveness
st.markdown("""
<style>
    /* Base styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .spam {
        background-color: rgba(255, 107, 107, 0.2);
        border: 1px solid rgba(255, 107, 107, 0.4);
    }
    .ham {
        background-color: rgba(107, 255, 107, 0.2);
        border: 1px solid rgba(107, 255, 107, 0.4);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    
    /* Responsive styles for mobile */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        .sub-header {
            font-size: 1.2rem;
        }
        .result-box {
            padding: 1rem;
        }
    }
    
    /* Make buttons more visible */
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        height: auto;
        padding: 10px 5px;
        font-weight: 500;
    }
    
    /* Make text areas and inputs larger */
    textarea, input {
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model_from_zip():
    """Download and load the saved BERT model from URL silently"""
    model_url = "https://filedn.eu/l7s80SJp4rmpWYtuEpH57df/bertmodel/textguard_bert.zip"
    model_dir = "./downloaded_model"
    
    try:
        # Create a temporary directory to download the model
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "model.zip")
            
            # Download the model without displaying progress
            response = requests.get(model_url, stream=True)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Extract the model silently
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the model directory (it might be in a subdirectory after extraction)
            extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            model_extracted_path = os.path.join(temp_dir, "textguard_bert")
            
            # If the model is in a subdirectory with a different name
            if "textguard_bert" not in extracted_dirs and len(extracted_dirs) > 0:
                model_extracted_path = os.path.join(temp_dir, extracted_dirs[0])
            
            # Load the model
            model = tf.saved_model.load(model_extracted_path)
            return model
                
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_message(model, message):
    """Make prediction using the loaded model"""
    if not message:
        return None
    
    result = model(tf.constant([message]))
    score = tf.sigmoid(result).numpy()[0][0]
    
    return {
        "probability": float(score),
        "is_spam": score > 0.5,
        "confidence": max(score, 1 - score)
    }

def main():
    # Header
    st.markdown('<p class="main-header">Text Classification with BERT</p>', unsafe_allow_html=True)
    st.markdown("""
    This app uses a fine-tuned BERT model to classify text as either normal (0) or spam/suspicious (1).
    Enter any text to see how the model classifies it.
    """)
    
    # Load model automatically without showing a button or progress
    if 'model' not in st.session_state:
        with st.spinner("Setting up the model... (this will only happen once)"):
            model = load_model_from_zip()
            if model:
                st.session_state.model = model
            else:
                st.error("Failed to load the model. Please refresh the page or try again later.")
                st.stop()
    else:
        model = st.session_state.model
    
    # Input area
    st.markdown('<p class="sub-header">Enter Text to Classify</p>', unsafe_allow_html=True)
    
    # Example texts
    st.markdown("**Try some examples:**")
    
    example_texts = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "Meeting rescheduled to 3pm tomorrow",
        "URGENT: Your bank account has been suspended. Verify now:",
        "Don't forget to pick up milk on your way home"
    ]
    
    # Create a row of buttons for examples
    # Use a more responsive layout with multiple rows for mobile
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"Example 1", key="example_0", help=example_texts[0]):
            st.session_state.example_text = example_texts[0]
        
        if st.button(f"Example 3", key="example_2", help=example_texts[2]):
            st.session_state.example_text = example_texts[2]
    
    with col2:
        if st.button(f"Example 2", key="example_1", help=example_texts[1]):
            st.session_state.example_text = example_texts[1]
        
        if st.button(f"Example 4", key="example_3", help=example_texts[3]):
            st.session_state.example_text = example_texts[3]
    
    # Get example text from session state if available
    initial_text = st.session_state.get("example_text", "")
    message = st.text_area("Type or paste text here:", value=initial_text, height=150, key="text_input")
    
    # Make prediction when text is present
    if message:
        result = predict_message(model, message)
        
        if result:
            # Display results
            st.markdown('<p class="sub-header">Classification Result</p>', unsafe_allow_html=True)
            
            if result["is_spam"]:
                st.markdown(f'<div class="result-box spam"><h3>Suspicious/Spam Text</h3>'
                            f'<p>Confidence: {result["confidence"]*100:.2f}%</p>'
                            f'<p>Probability Score: {result["probability"]:.4f}</p></div>', 
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box ham"><h3>Normal Text</h3>'
                            f'<p>Confidence: {result["confidence"]*100:.2f}%</p>'
                            f'<p>Probability Score: {result["probability"]:.4f}</p></div>',
                            unsafe_allow_html=True)
            
            # Visualize confidence with a progress bar
            st.progress(float(result["probability"]))
            
            st.markdown("**Interpretation:**")
            st.markdown("- Score close to 0: Likely normal message")
            st.markdown("- Score close to 1: Likely suspicious/spam message")

if __name__ == "__main__":
    main()
