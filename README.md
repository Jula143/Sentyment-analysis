# Sentiment analysis package
Package allows you to create and compare different models for sentiment analysis. It also contains a simple app that uses one of the models to clasify the sentiment of the review written by user.
Used dataset: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

Available classification methods:
- Naive Bayes
- Logistic Regression
- SVM
- K-Nearest Neighbours

Available vectorization methods:
- Count Vectorizer
- Yes/No Vector
- TF-IDF
- Word2Vec
- FastText

Methods for models comparison:
- confusion matrix
- accuracy
- precision
- recall
- F1 value
- specificity
- ROC curve

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Jula143/Sentyment-analysis
    cd Sentyment-analysis
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. To create new models and compare them write in your console command:
    ```bash
    python main.py
    ```
2. To see how one of the models handles the sentiment analysis deploy the app using:
    ```bash
   cd app
   python app.py
    ```


