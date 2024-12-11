# Spam-or-Not-Spam-Email-Classifier
This project implements a simple web application using **Streamlit** that classifies emails as **Spam** or **Not Spam** (Ham) based on their content. The classification model is built using a **Logistic Regression** classifier and trained on a dataset of labeled email messages. Users can input an email message and receive a prediction of whether it is spam or not. 
## Project Description

This project consists of:
- A dataset containing labeled email messages (`spam` or `not spam`).
- Preprocessing the dataset to convert categorical labels into numerical values.
- Feature extraction using **TF-IDF Vectorizer** to convert email text into numerical features.
- A **Logistic Regression** classifier to predict whether an email is spam or not.
- A **Streamlit** web app where users can input email messages and get predictions, along with examples of spam and not spam emails.

## Steps Involved

1. **Data Preprocessing**:
    - Load the dataset from a CSV file.
    - Clean the data by handling missing values.
    - Map categories to numeric values (`0` for spam, `1` for not spam).
    
2. **Feature Extraction**:
    - Use **TF-IDF Vectorizer** to convert the email content (text) into numerical features.

3. **Model Training**:
    - Split the dataset into training and test sets.
    - Train the **Logistic Regression** model on the training data.

4. **Streamlit Web App**:
    - Create a user interface where users can input an email message.
    - Display the classification result (Spam or Not Spam).
    - Show examples of spam and not spam messages.

5. **Deployment**:
    - Deploy the app using **Streamlit** to allow users to interact with it.
    - Optionally, set up **CI/CD** for automatic deployment and updates.
