# Toxic Comment Classification Project

## Introduction
This project aims to build a multi-headed model in Python to identify various forms of toxicity in user comments, including insults, obscenity, threats, and identity-based hate. The dataset, derived from the Wiki corpus, includes labeled categories such as Insult, Identity Hate, Threat, Toxic, Obscene, and Severe Toxic.

## Objective
The goal is to accurately predict these toxicity categories for each comment using a model built without transformers or pre-trained LLMs.

## Technology and Tools Used
- Python
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for data visualization
- NLTK and SpaCy for text processing
- TensorFlow and Keras for model building
- Jupyter Notebook for development

## Steps
1. **Data Loading and Exploration**
    - Load the training and test datasets.
    - Display the first few rows of the dataset to understand its structure.

2. **Data Understanding and Cleaning**
    - Check the structure of the dataset and summary statistics.
    - Identify and handle missing values.
    - Remove unnecessary columns and clean the text data.

3. **Text Preprocessing**
    - Remove stopwords using NLTK.
    - Apply lemmatization using SpaCy.
    - Remove HTML tags, links, special characters, digits, and multiple spaces.
    - Convert text to lowercase and apply the preprocessing function to the comment text.

4. **Exploratory Data Analysis (EDA)**
    - Visualize the distribution of toxicity categories.
    - Generate word clouds for different toxicity categories.

5. **Model Building**
    - Use TF-IDF Vectorizer for feature extraction.
    - Split the data into training and testing sets.
    - Build a multi-headed neural network using TensorFlow and Keras.
    - Compile and train the model.

6. **Model Evaluation**
    - Evaluate the model using classification metrics like precision, recall, and F1-score.
    - Generate a classification report.

7. **Saving and Exporting the Model**
    - Save the trained model for future use.
    - Export the preprocessed data for future data imports.

## Conclusion
This notebook provides a comprehensive approach to building a multi-headed model for classifying toxic comments. The steps include data loading, preprocessing, EDA, model building, and evaluation. The project leverages various data science and machine learning techniques to achieve the desired outcomes.

## Data Source
The data for this project is sourced from the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview) on Kaggle.
