Mental Health Text Classification
This project focuses on classifying text data related to mental health using various machine learning models, including Naive Bayes, Random Forest, and LSTM-based neural networks. The dataset used contains text labeled with binary labels indicating whether the text suggests a mental health concern (1) or not (0).

Dataset
The dataset consists of 27,977 entries with the following columns:

text: The textual content to classify.
label: The binary label indicating the presence (1) or absence (0) of mental health concerns.
Project Workflow
Data Preprocessing

Removal of duplicates and irrelevant characters (e.g., punctuation, URLs).
Tokenization, stemming, and stopword removal.
Conversion of text to numeric format using TF-IDF and Word2Vec.
Model Training

Naive Bayes: TF-IDF features used to train a Multinomial Naive Bayes model.
Random Forest: Word2Vec embeddings used to train a Random Forest model.
LSTM Neural Network: A sequential model with Bidirectional LSTM layers and embedding for text sequences.
Evaluation

Models are evaluated using the following metrics:
Accuracy: The percentage of correctly classified instances.
Precision: The ratio of true positive predictions to the total positive predictions (i.e., the relevance of the positive results).
Recall: The ratio of true positive predictions to the total actual positives (i.e., the ability to find all relevant positive results).
F1-Score: The harmonic mean of precision and recall, providing a single metric that balances the two.
Confusion Matrix: A summary of prediction results on a classification problem, showing the counts of true positives, true negatives, false positives, and false negatives.
AUC-ROC: The Area Under the Receiver Operating Characteristic Curve, which provides a measure of the model's ability to distinguish between classes.
Results
Naive Bayes:

Accuracy: 89.4%
Precision: 0.85 (class 1), 0.95 (class 0)
Recall: 0.96 (class 1), 0.83 (class 0)
F1-Score: 0.90 (class 1), 0.89 (class 0)
AUC-ROC: 0.92
Random Forest:

Accuracy: 61%
Precision: 0.78 (class 1), 0.57 (class 0)
Recall: 0.30 (class 1), 0.92 (class 0)
F1-Score: 0.44 (class 1), 0.70 (class 0)
AUC-ROC: 0.66
LSTM Model:

Accuracy: 92.4%
Precision: 0.94 (class 1), 0.92 (class 0)
Recall: 0.92 (class 1), 0.93 (class 0)
F1-Score: 0.93 (class 1), 0.93 (class 0)
AUC-ROC: 0.95
Dependencies
Python 3.x
Pandas
Numpy
Matplotlib
Seaborn
Sklearn
NLTK
Gensim
TensorFlow
Keras
Skimpy
Install dependencies using:

bash
Copy code
pip install -r requirements.txt
How to Run
Preprocess the data and extract features by running the data_preprocessing.py script.
Train the models using the respective scripts:
train_naive_bayes.py
train_random_forest.py
train_lstm.py
Evaluate the models and compare their performance using evaluate_models.py.
Conclusion
The LSTM model showed the best performance in detecting mental health-related text, achieving the highest accuracy and robust performance across all evaluation metrics. It is recommended as the most effective model for this task.

