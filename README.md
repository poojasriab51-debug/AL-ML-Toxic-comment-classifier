# Toxic Comment Detection AI/ML

A machine learning project to detect and classify toxic comments across multiple categories using deep learning.

## Features

- **Multi-label classification**: Detects 6 types of toxicity:
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate
- **LSTM-based model**: Uses a Bidirectional LSTM neural network for text classification
- **Interactive web app**: Built with Streamlit for easy testing

## Project Structure

```
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
├── data/
│   ├── train.csv              # Training data
│   ├── test.csv               # Test data
│   └── train_cleaned.csv      # Preprocessed training data
├── models/
│   ├── toxic_comment_model.keras  # Trained model
│   └── tokenizer.pkl              # Saved tokenizer
├── notebooks/
│   └── Toxicity_Detection.ipynb   # Jupyter notebook with EDA & training
├── outputs/                   # Output files
└── src/
    └── preprocessing.py       # Text preprocessing utilities
```

## Setup

1. **Clone the repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook** (optional - to train the model)
   ```bash
   jupyter notebook notebooks/Toxicity_Detection.ipynb
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the Streamlit app
2. Enter a comment in the text area
3. Click **Analyze Comment**
4. View predictions across all 6 toxicity categories

## Tech Stack

- **TensorFlow/Keras** - Deep learning framework
- **scikit-learn** - Data preprocessing & evaluation
- **NLTK** - Natural language processing
- **Pandas/NumPy** - Data manipulation
- **Streamlit** - Web application framework

