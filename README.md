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

