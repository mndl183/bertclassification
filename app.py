import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Text Classification with BERT",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the saved BERT model"""
    try:
        model = tf.saved_model.load('./textguard_bert')
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
    
    # Load model
    with st.spinner("Loading model... This might take a moment."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model files exist in the correct location.")
        st.stop()
    
    st.success("Model loaded successfully!")
    
    # Input area
    st.markdown('<p class="sub-header">Enter Text to Classify</p>', unsafe_allow_html=True)
    message = st.text_area("Type or paste text here:", height=150)
    
    # Example texts
    with st.expander("Try some example messages"):
        example_texts = [
            "Congratulations! You've won a $1000 gift card. Click here to claim now!",
            "Meeting rescheduled to 3pm tomorrow",
            "URGENT: Your bank account has been suspended. Verify now:",
            "Don't forget to pick up milk on your way home"
        ]
        
        for i, example in enumerate(example_texts):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                message = example
                st.session_state.message = example
    
    # Make prediction
    if message:
        with st.spinner("Analyzing text..."):
            result = predict_message(model, message)
        
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
        
        # Visualize confidence
        st.progress(float(result["probability"]))
        
        st.markdown("**Interpretation:**")
        st.markdown("- Score close to 0: Likely normal message")
        st.markdown("- Score close to 1: Likely suspicious/spam message")

if __name__ == "__main__":
    main()