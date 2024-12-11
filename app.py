import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and preprocess the data
raw_mail_data = pd.read_csv(r'E:\Data analysis\Projects\learning\17\mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Map categories to 0 and 1 (spam=0, not spam=1)
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Check for any unexpected or missing categories
print(mail_data['Category'].value_counts())

X = mail_data['Message']
Y = mail_data['Category']

# Ensure Y is of integer type
Y = Y.astype(int)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction using TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Ensure Y_train is of integer type
Y_train = Y_train.astype(int)

# Train the model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Streamlit UI
st.title('Spam or Not Spam Email Classifier')

# Add examples of spam and not spam
spam_example = "Congratulations! You've won a $1000 gift card, click here to claim it!"
not_spam_example = "Hello, I wanted to check in on the status of the project we discussed."

st.subheader('Examples:')
st.write(f"Spam Example: {spam_example}")
st.write(f"Not Spam Example: {not_spam_example}")

# Get user input for email message
user_message = st.text_area("Enter email message:")

# Predict when the user inputs a message
if user_message:
    # Transform the user input
    user_input_features = feature_extraction.transform([user_message])
    
    # Predict the category
    prediction = model.predict(user_input_features)
    
    # Display the prediction result
    if prediction == 0:
        st.write("This message is **Spam**.")
    else:
        st.write("This message is **Not Spam**.")
