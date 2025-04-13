# spam_detector_with_input.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Step 2: Preprocessing
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# Step 4: Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Take user input
print("\n📝 Enter your message below to check if it's Spam or Not Spam:")
user_input = input("Your Message: ")

# Step 8: Predict user message
user_vec = vectorizer.transform([user_input])
prediction = model.predict(user_vec)

print("\n🔍 Prediction Result:")
print("👉 This message is:", "Spam" if prediction[0] == 1 else "Not Spam")
