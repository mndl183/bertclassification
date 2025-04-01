# BERT Text Classification App

This is a Streamlit web application that uses a fine-tuned BERT model to classify text messages as either normal or spam/suspicious.

## Setup Instructions

### 1. Prerequisites
- Python 3.8 or newer
- The trained BERT model (`textguard_bert` directory)

### 2. Installation

Clone this repository and navigate to the project directory:

```bash
git clone <your-repository-url>
cd <repository-directory>
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Model Setup

Make sure your trained BERT model is in the correct location. The app expects the model to be in a directory named `textguard_bert` in the same directory as the app.py file.

### 4. Running Locally

To run the application locally:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.

### 5. Deploying to Streamlit Cloud

To deploy to Streamlit Cloud:

1. Push your code to a GitHub repository
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and select your repository
4. Set the main file path to `app.py`
5. Deploy the app

## Important Notes for Deployment

- **Model Size**: BERT models can be large. Ensure your model is under the size limits of Streamlit Cloud or consider using a smaller model.
- **Requirements**: All required packages are listed in `requirements.txt`. Streamlit Cloud will install these automatically.
- **Secrets**: If your app requires API keys or secrets, use Streamlit's secrets management.

## Using the App

1. Enter text in the provided text area
2. The model will classify the text and display the result
3. You can also try one of the example messages by clicking the example buttons

## Model Information

The model used in this app is a fine-tuned BERT model based on `small_bert/bert_en_uncased_L-4_H-512_A-8`. It was trained to classify text messages as either normal (0) or spam/suspicious (1).