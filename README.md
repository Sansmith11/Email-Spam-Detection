# EMAIL SPAM DETECTION WITH MACHINE LEARNING


1. Project Overview

Email Spam Detection is a machine learning–based classification problem where incoming messages are categorized as Spam or Ham (Non-Spam). 
This project uses Natural Language Processing (NLP) and supervised learning algorithms to detect spam messages based on their textual content.

2. Dataset Description

Dataset Name: SMS Spam Collection Dataset

Source: Kaggle

Link:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/code

Total Records: 5,572 SMS messages

Classes:

ham → Non-spam

spam → Spam

The dataset contains two main columns:

v1 → Label (spam/ham)

v2 → Message text

3. Libraries Used

NumPy – Numerical computations

Pandas – Data manipulation

Matplotlib & Seaborn – Data visualization

NLTK – Text preprocessing and NLP

Scikit-learn – Machine learning models and evaluation

Pickle – Saving trained models

4. Data Preprocessing

The following preprocessing steps were applied to clean and prepare text data:

Removal of special characters using Regular Expressions

Conversion of text to lowercase

Tokenization of words

Removal of stopwords using NLTK

Stemming using Porter Stemmer

Creation of a Bag of Words model using CountVectorizer

This helps convert raw text into numerical form suitable for machine learning models.

5. Feature Extraction

  Technique Used: Bag of Words (BoW)

  Vectorizer: CountVectorizer(max_features=4000)

  Converts text messages into numerical feature vectors.

6. Train–Test Split

   Training Data: 80%

   Testing Data: 20%

   Random State: 42

This ensures reproducibility and fair evaluation.

7. Machine Learning Models Used
• Random Forest Classifier

An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.

• Decision Tree Classifier

A tree-based model that splits data based on feature conditions for classification.

• Multinomial Naïve Bayes

A probabilistic classifier especially effective for text classification problems such as spam detection.

8. Model Evaluation

Each model was evaluated using:

  Confusion Matrix

  Accuracy Score

  Classification Report

  Heatmap Visualization

 Observation:
Multinomial Naïve Bayes achieved the highest accuracy and showed better precision and recall for spam detection.

9. Best Model

 Multinomial Naïve Bayes
It performs best because:

It works efficiently with high-dimensional sparse data

It is well-suited for NLP-based problems

It gives stable results with less computational cost

10. Model Saving

All trained models were saved using Pickle for future deployment:

RFC.pkl

DTC.pkl

MNB.pkl

